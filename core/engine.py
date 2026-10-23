"""Single-process multi-symbol trading engine.

One MT5 connection, one loop — scans all enabled symbols every POLL_INTERVAL.
Each symbol has its own independent SymbolContext with config, state, and specs.

Usage:
    python -m core.engine                        # all configured symbols
    python -m core.engine --symbols BTCUSD XAUUSD  # specific symbols
"""

import datetime
import logging
import time
from typing import List, Optional

import MetaTrader5 as mt5

from config import (
    HTF, HTF_CANDLES_LOOKBACK, LTF, LTF_CANDLES_LOOKBACK,
    LIQUIDITY_TF, LIQUIDITY_CANDLES, MAX_LTF_WAIT_CANDLES,
    POLL_INTERVAL, SWING_STRENGTH,
    USE_CLOSED_CANDLES_ONLY,
)
from core.symbol_context import SymbolContext
from mt5_connector import MT5Connector
from risk.risk_manager import (
    calculate_lot_size, validate_entry,
)
from chart import BotChart
from core.persistence import BotDatabase
from strategy import SMCStrategy
import telegram_bot

logger = logging.getLogger("smc_bot")

RECOVERY_SLEEP = 30

# All symbols the engine can trade.  Add more here.
# Backtest 2026-10-23: EURUSD PF 0.61, avgR -0.188 — consistently losing.
# Removed from live trading.  Recoverable from git.
DEFAULT_SYMBOLS = ["BTCUSD", "GBPUSD"]

# ICT macro time windows (UTC hours).  Only trade during these.
MACRO_WINDOWS_UTC = [
    (2, 0, 5, 0),    # London killzone
    (7, 0, 11, 0),   # NY AM killzone
    (12, 0, 13, 0),  # NY lunch macro
    (13, 30, 16, 0),  # NY PM
    (19, 0, 20, 0),  # Asian (crypto)
]


class MultiSymbolEngine:
    """Drives the SMC strategy across multiple symbols from one process."""

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbol_names = symbols or DEFAULT_SYMBOLS
        self.contexts: dict[str, SymbolContext] = {}
        self.connectors: dict[str, MT5Connector] = {}
        self.db = BotDatabase()
        self.chart = BotChart()
        self._running = False
        self._tg_listener = None
        self._cycle = 0
        self._last_status = 0.0
        # Global daily trade counter (across all symbols)
        self._global_trades_today = 0
        self._global_trade_date = ""

    # ---------------------------------------------------------------- init
    def initialize(self) -> bool:
        """Connect to MT5 and set up all symbol contexts."""
        if not mt5.initialize():
            logger.error("MT5 init failed: %s", mt5.last_error())
            return False

        from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logger.error("MT5 login failed: %s", mt5.last_error())
            return False

        logger.info("Connected to MT5.")

        for sym in self.symbol_names:
            ctx = SymbolContext(sym)
            if not ctx.initialize():
                logger.warning("Skipping %s — not available.", sym)
                continue
            connector = MT5Connector(symbol=sym, magic_number=ctx.cfg.magic_number)
            # Connector is already init'd via the global mt5.initialize()
            connector._initialized = True
            self.contexts[sym] = ctx
            self.connectors[sym] = connector
            logger.info(
                "Enabled %s: lot=%s stop=%s risk=%.1f%% night=%d-%d blocked=%s",
                sym, ctx.cfg.lot_size, ctx.cfg.stop_mode, ctx.cfg.risk_percent,
                ctx.cfg.night_start, ctx.cfg.night_end, ctx.cfg.blocked_days or "none")

        if not self.contexts:
            logger.error("No symbols available — cannot start.")
            return False

        # Restore persisted state (consecutive losses, paused, etc.)
        for sym, ctx in self.contexts.items():
            saved = self.db.load_symbol_state(sym)
            if saved:
                ctx.state.consecutive_losses = saved["consecutive_losses"]
                ctx.state.paused = bool(saved["paused"])
                ctx.state.pause_reason = saved["pause_reason"] or ""
                logger.info("Restored %s state: consec_losses=%d paused=%s",
                            sym, ctx.state.consecutive_losses, ctx.state.paused)

        # Telegram
        if telegram_bot.test_connection():
            account = mt5.account_info()
            bal = account.balance if account else 0
            symbols_list = ", ".join(self.contexts.keys())
            telegram_bot.send_message(
                f"🚀 <b>SMC Bot Started</b>\n\n"
                f"Symbols: {symbols_list}\n"
                f"Balance: ${bal:.2f}")

            # Start command listener with close handlers
            self._tg_listener = telegram_bot.TelegramCommandListener(
                on_close_symbol=self._tg_close_symbol,
                on_close_all=self._tg_close_all,
            )
            self._tg_listener.start()
        else:
            self._tg_listener = None

        logger.info("Engine ready: %d symbols.", len(self.contexts))
        return True

    def _tg_close_symbol(self, symbol: str):
        """Close position on a symbol (triggered by Telegram)."""
        if symbol in self.connectors:
            conn = self.connectors[symbol]
            positions = conn.get_open_positions()
            if positions:
                for pos in positions:
                    close_type = (mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY
                                  else mt5.ORDER_TYPE_BUY)
                    price = conn.entry_price(close_type)
                    if price:
                        req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": pos.volume,
                            "type": close_type,
                            "position": pos.ticket,
                            "price": price,
                            "deviation": 20,
                            "magic": conn.magic_number,
                            "comment": "TG_close",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(req)
                        if result and result.retcode == 10009:
                            telegram_bot.send_message(f"✅ {symbol} position closed.")
                        else:
                            telegram_bot.send_message(f"❌ Failed to close {symbol}.")
            else:
                telegram_bot.send_message(f"No open position on {symbol}.")

    def _tg_close_all(self):
        """Close all bot positions (triggered by Telegram)."""
        for sym in self.connectors:
            self._tg_close_symbol(sym)

    # ---------------------------------------------------------------- main loop
    def run(self):
        """Main poll loop — scans all symbols every POLL_INTERVAL seconds."""
        self._running = True
        logger.info("=" * 50)
        logger.info("Multi-symbol SMC engine started (%d symbols).",
                     len(self.contexts))
        logger.info("Symbols: %s", ", ".join(self.contexts.keys()))
        logger.info("=" * 50)

        self._cycle = 0
        self._last_status = 0

        try:
            while self._running:
                try:
                    if not mt5.terminal_info():
                        logger.warning("MT5 connection lost — reconnecting...")
                        if not mt5.initialize():
                            time.sleep(RECOVERY_SLEEP)
                            continue

                    for sym, ctx in self.contexts.items():
                        try:
                            self._scan_symbol(ctx)
                        except Exception:
                            logger.exception("Error scanning %s — skipping.", sym)

                    # Periodic status log every 5 minutes
                    self._cycle += 1
                    now = time.time()
                    if now - self._last_status >= 300:
                        self._last_status = now
                        parts = []
                        tg_statuses = []
                        for sym, ctx in self.contexts.items():
                            s = ctx.state
                            poi_str = f"POI={'Y' if s.poi else 'N'}"
                            trend_str = s.trend[:4]
                            pos = self.connectors[sym].get_open_positions()
                            pos_str = f"POS={len(pos)}" if pos else "POS=0"
                            parts.append(f"{sym}({trend_str} {poi_str} {pos_str})")

                            # Build Telegram status entry
                            entry = {
                                "symbol": sym,
                                "trend": s.trend,
                                "poi": s.poi is not None,
                                "position": bool(pos),
                                "rejections": dict(s.rejections),
                            }
                            if s.poi:
                                entry["poi_top"] = float(s.poi.top)
                                entry["poi_bottom"] = float(s.poi.bottom)
                                entry["direction"] = s.poi.type
                                entry["zone_active"] = s.watch_key is not None and not s.abandoned
                            if hasattr(s, '_last_dol') and s._last_dol is not None:
                                entry["dol"] = s._last_dol
                            tg_statuses.append(entry)
                            # Reset rejection counters after reporting
                            s.rejections = {}

                        logger.info("Status: %s", " | ".join(parts))

                        # Send to Telegram every 30 minutes (6th cycle)
                        if self._cycle % 6 == 1:
                            telegram_bot.notify_scan_status(tg_statuses)

                        # Generate HTF overview charts for each symbol
                        for sym, ctx in self.contexts.items():
                            try:
                                df_htf = self.connectors[sym].fetch_rates(HTF, HTF_CANDLES_LOOKBACK)
                                if df_htf is not None:
                                    self.chart.plot_htf_only(
                                        symbol=sym, df_htf=df_htf,
                                        poi=ctx.state.poi, trend=ctx.state.trend,
                                        ema_period=ctx.cfg.trend_ema_period)
                            except Exception:
                                pass

                    self._handle_tg_status()
                    time.sleep(POLL_INTERVAL)

                except KeyboardInterrupt:
                    raise
                except Exception:
                    logger.exception("Engine error — recovering.")
                    time.sleep(RECOVERY_SLEEP)

        except KeyboardInterrupt:
            logger.info("Engine stopped by user.")
        finally:
            # Persist all symbol states before exit
            for sym, ctx in self.contexts.items():
                self.db.save_symbol_state(
                    sym, ctx.state.consecutive_losses,
                    ctx.state.paused, ctx.state.pause_reason)
            self.db.close()
            mt5.shutdown()
            logger.info("MT5 disconnected. State persisted.")

    # ---------------------------------------------------------------- per-symbol
    def _scan_symbol(self, ctx: SymbolContext):
        """Unified trading plan — one strategy flow per the discretionary plan.

        Flow:
          0. Manage existing / detect closed
          1. Global 1-trade-per-day gate
          2. is_trading_allowed (news, risk, caps, blocked days, etc.)
          3. TIME gate — only trade during macro/killzone windows
          4. HTF NARRATIVE — 1H trend + POI (FVG+OB pd-array)
          5. DOL — Draw on Liquidity (nearest swing target on 15m/1H)
          6. LTF EXECUTION — 5m FVG confirmation inside POI
          7. SMT divergence confirmation
          8. Risk, sizing, liquidity resistance → execute
        """
        sym = ctx.symbol
        connector = self.connectors[sym]
        now = datetime.datetime.now(datetime.timezone.utc)

        # Daily resets
        today = now.strftime("%Y-%m-%d")
        ctx.state.reset_daily(today)
        ctx.state.reset_periodic(now)
        if self._global_trade_date != today:
            self._global_trade_date = today
            self._global_trades_today = 0

        # 0. Detect closed positions — update risk state
        positions = connector.get_open_positions()
        if ctx.state.last_position_ticket is not None and not positions:
            self._on_position_closed(ctx, connector)

        # 0b. Manage existing positions (always, even during blocked hours)
        if positions:
            ctx.state.last_position_ticket = positions[0].ticket
            self._manage_positions(ctx, connector, positions)
            return  # one position at a time per symbol

        # === PLAN RULE 10: Global 1 trade per day ===
        if self._global_trades_today >= ctx.cfg.max_trades_per_day:
            ctx.state.record_rejection("GLOBAL_MAX_DAILY_TRADES")
            return

        # === PLAN RULES 7,9,10,11,13: is_trading_allowed ===
        # (news, blocked days, daily loss, consecutive losses, weekly/monthly cap)
        block = ctx.is_trading_allowed(now)
        if block:
            ctx.state.record_rejection(block)
            return

        # === PLAN RULE 3: TIME — only during macro/killzone windows ===
        if not self._in_macro_window(now):
            return  # silent — happens every 10 seconds, don't log

        # === PLAN RULE 2: HTF NARRATIVE — 1H trend + POI ===
        df_htf = connector.fetch_rates(HTF, HTF_CANDLES_LOOKBACK)
        df_ltf = connector.fetch_rates(LTF, LTF_CANDLES_LOOKBACK)
        if df_htf is None or df_ltf is None:
            return

        # 4a. HTF trend (EMA 100)
        trend = (
            SMCStrategy.get_htf_trend(
                df_htf, ctx.cfg.trend_ema_period,
                use_closed_candles=USE_CLOSED_CANDLES_ONLY)
            if ctx.cfg.use_trend_filter else "NEUTRAL"
        )
        ctx.state.trend = trend

        if trend in ctx.cfg.blocked_trends:
            ctx.state.record_rejection("BLOCKED_TREND")
            return

        if trend == "NEUTRAL":
            return  # no narrative = no trade

        # 4b. HTF POI (1H FVG + OB = pd-array)
        poi = SMCStrategy.detect_htf_poi(
            df_htf=df_htf,
            use_trend_filter=ctx.cfg.use_trend_filter,
            ema_period=ctx.cfg.trend_ema_period,
            use_closed_candles=USE_CLOSED_CANDLES_ONLY,
        )
        ctx.state.poi = poi

        if poi is None:
            return  # no HTF pd-array = no narrative = no trade

        # === PLAN RULE 1: DOL — Draw on Liquidity ===
        df_liq = connector.fetch_rates(LIQUIDITY_TF, LIQUIDITY_CANDLES)
        tick = connector.get_tick()
        if tick is None:
            return
        current_price = (tick.bid + tick.ask) / 2

        # Find the DOL target (nearest swing on 15m)
        dol_target = None
        if df_liq is not None:
            dol_target = SMCStrategy.find_nearest_liquidity(
                df_liq,
                "BUY" if poi.type == "BULLISH" else "SELL",
                current_price,
                strength=SWING_STRENGTH,
                use_closed_candles=True,
            )

        if dol_target is None:
            ctx.state.record_rejection("NO_DOL")
            return  # no draw on liquidity = no reason for price to move
        ctx.state._last_dol = dol_target

        # === PLAN RULE 5: LTF EXECUTION — 5m FVG inside POI ===
        # Check zone patience (price must be in/near the POI)
        if not self._check_zone(ctx, df_ltf, poi, current_price):
            return

        # 5m FVG confirmation with SL at swing high/low
        setup = SMCStrategy.check_ltf_confirmation(
            df_ltf, poi, ctx.cfg.rrr,
            use_closed_candles=USE_CLOSED_CANDLES_ONLY,
            stop_mode=ctx.cfg.stop_mode, buffer_atr=0.5)

        if setup is None:
            return

        # Override TP with DOL target (Plan Rule 4: 15m/1H target)
        risk = abs(setup["entry"] - setup["sl"])
        dol_dist = abs(dol_target - setup["entry"])
        dol_rr = dol_dist / risk if risk > 0 else 0
        if dol_rr >= ctx.cfg.min_rr_liquidity:
            setup["tp"] = dol_target
        # else keep the fixed RRR tp — DOL too close

        # Notify: a valid setup was found, about to validate
        logger.info("[%s] SETUP FOUND: %s entry=%.5f sl=%.5f tp=%.5f model=%s",
                    sym, setup["direction"], setup["entry"], setup["sl"],
                    setup["tp"], setup.get("model", "PLAN"))
        telegram_bot.notify_setup_found(
            sym, setup["direction"], setup["entry"],
            setup["sl"], setup["tp"], setup.get("model", "PLAN"))

        # === PLAN RULE 6: SMT divergence gate ===
        if ctx.cfg.require_smt and ctx.cfg.smt_correlated_symbol:
            smt_block = self._check_smt(ctx, connector, df_ltf, setup)
            if smt_block:
                ctx.state.record_rejection(smt_block)
                return

        # === CHART: visualize what the bot sees ===
        try:
            self.chart.plot_analysis(
                symbol=sym,
                df_htf=df_htf,
                df_ltf=df_ltf,
                poi=poi,
                dol_target=dol_target,
                setup=setup,
                trend=trend,
                ema_period=ctx.cfg.trend_ema_period,
            )
        except Exception:
            logger.warning("[%s] Chart generation failed — continuing.", sym)

        # === PLAN RULES 8,9: Execute with risk/resistance checks ===
        self._execute_setup(ctx, connector, setup, "PLAN", df_liq, now)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _in_macro_window(now) -> bool:
        """Check if current UTC time is inside an ICT macro/killzone window."""
        h, m = now.hour, now.minute
        current_mins = h * 60 + m
        for start_h, start_m, end_h, end_m in MACRO_WINDOWS_UTC:
            window_start = start_h * 60 + start_m
            window_end = end_h * 60 + end_m
            if window_start <= current_mins < window_end:
                return True
        return False

    def _check_zone(self, ctx, df_ltf, poi, current_price):
        """Zone patience: price must be in/near the POI, with timeout."""
        if not SMCStrategy.is_zone_in_play(poi, df_ltf, current_price=current_price):
            return False

        zone_key = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
        if zone_key != ctx.state.watch_key:
            ctx.state.watch_key = zone_key
            ctx.state.watch_start = df_ltf.iloc[-1]["time"]
            ctx.state.abandoned = False
            telegram_bot.notify_waiting_for_trade(
                ctx.symbol, poi.type,
                round(float(poi.top), 5), round(float(poi.bottom), 5),
                ctx.state.trend, "Waiting for 5m FVG")

        if ctx.state.abandoned:
            return False

        if ctx.state.watch_start is not None:
            waited = int((df_ltf["time"] >= ctx.state.watch_start).sum())
            if waited > MAX_LTF_WAIT_CANDLES:
                ctx.state.abandoned = True
                return False

        return True

    def _check_smt(self, ctx, connector, df_ltf, setup) -> Optional[str]:
        """Check SMT divergence with correlated symbol. Returns rejection reason or None."""
        corr_sym = ctx.cfg.smt_correlated_symbol
        if not corr_sym:
            return None

        # Fetch correlated symbol data
        df_corr = None
        if corr_sym in self.connectors:
            df_corr = self.connectors[corr_sym].fetch_rates(LTF, LTF_CANDLES_LOOKBACK)
        else:
            # Not in our trading list — fetch directly from MT5
            rates = mt5.copy_rates_from_pos(corr_sym, LTF, 0, LTF_CANDLES_LOOKBACK)
            if rates is not None and len(rates) > 0:
                import pandas as pd
                df_corr = pd.DataFrame(rates)

        if df_corr is None:
            logger.warning("[%s] Cannot fetch SMT data for %s — allowing trade.",
                          ctx.symbol, corr_sym)
            return None  # fail-open

        smt = SMCStrategy.detect_smt_divergence(
            df_ltf, df_corr,
            swing_strength=SWING_STRENGTH,
            use_closed_candles=USE_CLOSED_CANDLES_ONLY,
        )

        if smt is None:
            return "NO_SMT_DIVERGENCE"

        # SMT must align with trade direction
        direction = setup["direction"]
        if direction == "BUY" and smt != "BULLISH":
            return f"SMT_WRONG_DIRECTION ({smt} vs BUY)"
        if direction == "SELL" and smt != "BEARISH":
            return f"SMT_WRONG_DIRECTION ({smt} vs SELL)"

        logger.info("[%s] SMT confirmed: %s divergence with %s",
                    ctx.symbol, smt, corr_sym)
        return None

    # ---------------------------------------------------------------- execution

    def _execute_setup(self, ctx, connector, setup, strategy_name, df_liq, now):
        """Validate, size, and execute a setup from any strategy."""
        sym = ctx.symbol

        # TP comes from the DOL target set in _scan_symbol, or fixed RRR fallback.
        tp = setup["tp"]

        # Liquidity resistance filter: classify target, adjust risk or skip
        liq_resistance = None
        if ctx.cfg.use_liquidity_resistance and df_liq is not None:
            liq_resistance = SMCStrategy.classify_liquidity_resistance(
                df_liq, tp, setup["direction"],
                swing_strength=SWING_STRENGTH,
                use_closed_candles=True)
            if liq_resistance == "HIGH":
                if ctx.cfg.high_resistance_risk_factor <= 0:
                    ctx.state.record_rejection("HIGH_RESISTANCE_TARGET")
                    logger.info("[%s] %s REJECTED: only high resistance liquidity target.",
                                sym, strategy_name)
                    telegram_bot.notify_setup_rejected(
                        sym, setup["direction"], "HIGH_RESISTANCE_TARGET",
                        setup.get("entry", 0), setup["sl"])
                    return
                logger.info("[%s] %s: high resistance target — risk reduced to %.0f%%.",
                            sym, strategy_name, ctx.cfg.high_resistance_risk_factor * 100)

        # Max OB risk width filter (backtest 2026-09-16: XAUUSD outlier OBs)
        if ctx.cfg.max_ob_risk_px > 0:
            risk_px = abs(setup.get("entry", 0) - setup["sl"])
            if risk_px > ctx.cfg.max_ob_risk_px:
                ctx.state.record_rejection("OB_TOO_WIDE")
                logger.info("[%s] %s REJECTED: OB risk %.2f > max %.2f",
                            sym, strategy_name, risk_px, ctx.cfg.max_ob_risk_px)
                telegram_bot.notify_setup_rejected(
                    sym, setup["direction"], f"OB_TOO_WIDE ({risk_px:.2f} > {ctx.cfg.max_ob_risk_px:.2f})",
                    setup.get("entry", 0), setup["sl"])
                return

        # Risk validation
        entry_price = connector.entry_price(
            mt5.ORDER_TYPE_BUY if setup["direction"] == "BUY"
            else mt5.ORDER_TYPE_SELL)
        if entry_price is None:
            return

        # Premium/Discount filter: BUY only in discount, SELL only in premium
        if ctx.cfg.use_premium_discount:
            df_ltf = connector.fetch_rates(LTF, LTF_CANDLES_LOOKBACK)
            if df_ltf is not None:
                pd_zone = SMCStrategy.get_premium_discount(
                    df_ltf, lookback=50, use_closed_candles=USE_CLOSED_CANDLES_ONLY)
                if not SMCStrategy.is_premium_discount_aligned(pd_zone, setup["direction"]):
                    ctx.state.record_rejection("PREMIUM_DISCOUNT")
                    zone_name = pd_zone["zone"] if pd_zone else "N/A"
                    logger.info("[%s] %s REJECTED: %s not in correct P/D zone (%s)",
                                sym, strategy_name, setup["direction"], zone_name)
                    telegram_bot.notify_setup_rejected(
                        sym, setup["direction"], f"PREMIUM_DISCOUNT ({zone_name})",
                        setup.get("entry", 0), setup["sl"])
                    return

        # Guard: if market drifted too far from the strategy's intended entry,
        # the SL/TP were sized for a different price and the real RR is wrong.
        intended_entry = setup.get("entry")
        if intended_entry is not None:
            intended_risk = abs(intended_entry - setup["sl"])
            if intended_risk > 0:
                slippage_pct = abs(entry_price - intended_entry) / intended_risk
                if slippage_pct > 0.20:
                    ctx.state.record_rejection("ENTRY_DRIFT")
                    logger.info(
                        "[%s] %s REJECTED: market %.5f drifted %.0f%% from "
                        "intended %.5f (max 20%%)",
                        sym, strategy_name, entry_price,
                        slippage_pct * 100, intended_entry)
                    telegram_bot.notify_setup_rejected(
                        sym, setup["direction"],
                        f"ENTRY_DRIFT ({slippage_pct*100:.0f}% from {intended_entry:.5f})",
                        entry_price, setup["sl"])
                    return

        rejection = validate_entry(ctx, setup["direction"],
                                    entry_price, setup["sl"], tp)
        if rejection:
            ctx.state.record_rejection(rejection)
            logger.info("[%s] %s REJECTED: %s", sym, strategy_name, rejection)
            telegram_bot.notify_setup_rejected(
                sym, setup["direction"], rejection, entry_price, setup["sl"])
            return

        # Lot sizing
        lots = calculate_lot_size(ctx, entry_price, setup["sl"])
        if lots is None:
            ctx.state.record_rejection("SIZING_FAILED")
            telegram_bot.notify_setup_rejected(
                sym, setup["direction"], "SIZING_FAILED", entry_price, setup["sl"])
            return

        # Apply liquidity resistance risk reduction
        if liq_resistance == "HIGH" and ctx.cfg.high_resistance_risk_factor < 1.0:
            from decimal import Decimal, ROUND_DOWN
            reduced = lots * ctx.cfg.high_resistance_risk_factor
            step = Decimal(str(ctx.spec.volume_step))
            lots = float(Decimal(str(reduced)).quantize(step, rounding=ROUND_DOWN))
            lots = max(lots, ctx.spec.volume_min)
            logger.info("[%s] Lots reduced to %.2f (high resistance target).", sym, lots)

        # Execute
        logger.info(
            "[%s] %s SETUP: %s entry=%.5f sl=%.5f tp=%.5f lots=%.2f",
            sym, strategy_name, setup["direction"],
            entry_price, setup["sl"], tp, lots)

        order_type = (mt5.ORDER_TYPE_BUY if setup["direction"] == "BUY"
                      else mt5.ORDER_TYPE_SELL)

        risk = abs(entry_price - setup["sl"])
        if setup["direction"] == "BUY":
            actual_tp = tp if tp > entry_price else entry_price + risk * ctx.cfg.rrr
        else:
            actual_tp = tp if tp < entry_price else entry_price - risk * ctx.cfg.rrr

        levels = connector.normalize_levels(order_type, entry_price,
                                             setup["sl"], actual_tp)

        success = connector.send_order(
            order_type=order_type,
            sl=levels.sl,
            tp=levels.tp,
            volume=lots,
            comment=f"SMC_{strategy_name}_{setup['direction']}",
        )

        if success:
            logger.info("[%s] %s ORDER PLACED: %s %.2f lots",
                        sym, strategy_name, setup["direction"], lots)
            # Track the new position for close detection
            new_positions = connector.get_open_positions()
            if new_positions:
                ctx.state.last_position_ticket = new_positions[0].ticket
            account = mt5.account_info()
            bal = account.balance if account else 0
            telegram_bot.notify_trade_opened(
                sym, setup["direction"], lots,
                entry_price, setup["sl"], actual_tp,
                risk * lots * ctx.spec.contract_size, bal)
            # Journal: record trade open
            ticket = new_positions[0].ticket if new_positions else 0
            self.db.record_trade(
                symbol=sym, ticket=ticket, direction=setup["direction"],
                entry=entry_price, sl=setup["sl"], tp=actual_tp, volume=lots,
                entry_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                strategy=strategy_name,
                smt_confirmed=bool(ctx.cfg.require_smt and ctx.cfg.smt_correlated_symbol),
                liq_resistance=liq_resistance or "")
            self._reset_watch(ctx)
            self._global_trades_today += 1
        else:
            ctx.state.record_rejection("ORDER_FAILED")

    def _manage_positions(self, ctx: SymbolContext, connector: MT5Connector,
                          positions):
        """Manage open positions: breakeven, partial close."""
        if not ctx.cfg.use_partial_close and not ctx.cfg.use_breakeven:
            return

        tick = connector.get_tick()
        if tick is None:
            return

        for pos in positions:
            if not pos.sl or not pos.tp:
                continue

            entry = pos.price_open
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            current = tick.bid if is_buy else tick.ask

            # Already a runner (SL at entry)?
            tol = ctx.spec.point * 10  # symbol-aware tolerance
            at_entry = (pos.sl >= entry - tol if is_buy
                        else pos.sl <= entry + tol)
            if at_entry:
                continue

            # Breakeven at N×R (independent of partial close)
            if ctx.cfg.use_breakeven and not ctx.cfg.use_partial_close:
                risk = abs(entry - pos.sl)
                be_dist = risk * ctx.cfg.breakeven_r
                if is_buy:
                    if current >= entry + be_dist:
                        connector.modify_position_sl(pos, entry)
                        logger.info("[%s] Breakeven at %.1fR — SL moved to entry.",
                                    ctx.symbol, ctx.cfg.breakeven_r)
                else:
                    if current <= entry - be_dist:
                        connector.modify_position_sl(pos, entry)
                        logger.info("[%s] Breakeven at %.1fR — SL moved to entry.",
                                    ctx.symbol, ctx.cfg.breakeven_r)
                continue

            if not ctx.cfg.use_partial_close:
                continue

            tp_dist = abs(pos.tp - entry)
            trigger_price = (entry + tp_dist * ctx.cfg.partial_trigger_pct if is_buy
                             else entry - tp_dist * ctx.cfg.partial_trigger_pct)
            triggered = current >= trigger_price if is_buy else current <= trigger_price

            if triggered:
                vol_min = ctx.spec.volume_min
                can_split = pos.volume > vol_min

                if can_split:
                    # Enough volume to split: close half, keep runner
                    close_vol = connector.normalize_volume(pos.volume * ctx.cfg.partial_close_pct)
                    logger.info("[%s] Partial close %.2f lots at 80%% TP.",
                                ctx.symbol, close_vol)

                    if connector.partial_close(pos, close_vol):
                        # After partial close MT5 creates a new ticket for the
                        # remaining volume — the old pos.ticket is now closed.
                        remaining = connector.get_open_positions()
                        if not remaining:
                            logger.warning("[%s] No remaining position after partial close.", ctx.symbol)
                            continue
                        pos = remaining[0]

                        # Find next liquidity for runner TP
                        df_liq = connector.fetch_rates(LIQUIDITY_TF, LIQUIDITY_CANDLES)
                        new_tp = pos.tp
                        if df_liq is not None:
                            direction = "BUY" if is_buy else "SELL"
                            next_liq = SMCStrategy.find_next_liquidity(
                                df_liq, direction, pos.tp,
                                strength=SWING_STRENGTH, use_closed_candles=True)
                            if next_liq is not None:
                                new_tp = next_liq

                        connector.modify_position_sl_tp(pos, entry, new_tp)
                        logger.info("[%s] Runner: SL->entry, TP->%.5f",
                                    ctx.symbol, new_tp)
                        telegram_bot.notify_partial_close(
                            ctx.symbol, close_vol,
                            (current - entry) * close_vol * ctx.spec.contract_size
                            if is_buy else
                            (entry - current) * close_vol * ctx.spec.contract_size,
                            new_tp)
                else:
                    # Minimum lot — can't split, just move SL to entry
                    connector.modify_position_sl(pos, entry)
                    logger.info("[%s] Min lot (%.2f) — SL moved to entry (breakeven).",
                                ctx.symbol, pos.volume)

    def _handle_tg_status(self):
        """Respond to /status and /balance commands."""
        # Check if there are unprocessed status requests
        # This is handled by the TelegramCommandListener callback
        pass

    def _on_position_closed(self, ctx: SymbolContext, connector: MT5Connector):
        """Detect how the last position closed and update risk state."""
        ticket = ctx.state.last_position_ticket
        ctx.state.last_position_ticket = None

        # Look up the closing deal in history
        from_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2)
        to_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        deals = mt5.history_deals_get(from_date, to_date, position=ticket)

        pnl = 0.0
        if deals:
            # Sum profit from all deals on this position (entry + exit + partials)
            pnl = sum(d.profit + d.commission + d.swap for d in deals)

        ctx.state.record_trade_result(pnl)

        # Cooldown: wait N candles before the next trade on this symbol
        if ctx.cfg.cooldown_candles > 0:
            cooldown_minutes = 5 * ctx.cfg.cooldown_candles  # 5m per LTF candle
            now = datetime.datetime.now(datetime.timezone.utc)
            ctx.state.cooldown_until = now + datetime.timedelta(minutes=cooldown_minutes)

        logger.info(
            "[%s] Position #%s closed: P&L=$%.2f | "
            "consec_losses=%d trades_today=%d daily_pnl=$%.2f",
            ctx.symbol, ticket, pnl,
            ctx.state.consecutive_losses, ctx.state.trades_today,
            ctx.state.daily_pnl)

        # Persist to journal and DB
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        self.db.update_daily(date_str, ctx.symbol, pnl, pnl > 0)
        self.db.save_symbol_state(
            ctx.symbol, ctx.state.consecutive_losses,
            ctx.state.paused, ctx.state.pause_reason)

        # Update the trade record with exit info
        try:
            self.db.conn.execute(
                "UPDATE trades SET pnl=?, exit_time=?, exit_reason=? "
                "WHERE ticket=? AND pnl=0",
                (pnl, now_str, "SL" if pnl < 0 else "TP", ticket))
            self.db.conn.commit()
        except Exception:
            logger.warning("[%s] Failed to update trade record for #%s.", ctx.symbol, ticket)

        telegram_bot.send_message(
            f"{'✅' if pnl >= 0 else '❌'} <b>{ctx.symbol}</b> closed: "
            f"${pnl:+.2f} | "
            f"Day: ${ctx.state.daily_pnl:+.2f} | "
            f"Week: ${ctx.state.weekly_pnl:+.2f} | "
            f"Trades today: {ctx.state.trades_today}")

    def _reset_watch(self, ctx: SymbolContext):
        ctx.state.watch_key = None
        ctx.state.watch_start = None
        ctx.state.abandoned = False


# ---------------------------------------------------------------- CLI
def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="Multi-symbol SMC trading engine")
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="Symbols to trade (default: BTCUSD XAUUSD GBPUSD)")
    args = ap.parse_args()

    engine = MultiSymbolEngine(symbols=args.symbols)
    if engine.initialize():
        engine.run()


if __name__ == "__main__":
    main()
