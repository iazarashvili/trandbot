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
import pandas as pd

from config import (
    HTF, HTF_CANDLES_LOOKBACK, LTF, LTF_CANDLES_LOOKBACK,
    LIQUIDITY_TF, LIQUIDITY_CANDLES, MAX_LTF_WAIT_CANDLES,
    POLL_INTERVAL, SWING_STRENGTH,
    USE_CLOSED_CANDLES_ONLY,
)
from core.symbol_context import SymbolContext
from mt5_connector import MT5Connector
from risk.risk_manager import (
    calculate_lot_size, check_portfolio_risk, validate_entry,
)
from strategy import SMCStrategy
import telegram_bot

logger = logging.getLogger("smc_bot")

RECOVERY_SLEEP = 30

# All symbols the engine can trade.  Add more here.
DEFAULT_SYMBOLS = ["BTCUSD", "XAUUSD", "GBPUSD"]


class MultiSymbolEngine:
    """Drives the SMC strategy across multiple symbols from one process."""

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbol_names = symbols or DEFAULT_SYMBOLS
        self.contexts: dict[str, SymbolContext] = {}
        self.connectors: dict[str, MT5Connector] = {}
        self._running = False

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

                    # Periodic status check for Telegram /status and /balance
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
            mt5.shutdown()
            logger.info("MT5 disconnected.")

    # ---------------------------------------------------------------- per-symbol
    def _scan_symbol(self, ctx: SymbolContext):
        """One full cycle for one symbol."""
        sym = ctx.symbol
        connector = self.connectors[sym]
        now = datetime.datetime.now(datetime.timezone.utc)
        ctx.state.reset_daily(now.strftime("%Y-%m-%d"))

        # 1. Manage existing positions (always, even during blocked hours)
        positions = connector.get_open_positions()
        if positions:
            self._manage_positions(ctx, connector, positions)
            return  # one position at a time per symbol

        # 2. Check if trading is allowed
        block = ctx.is_trading_allowed(now)
        if block:
            ctx.state.record_rejection(block)
            return

        # 3. Fetch market data
        df_htf = connector.fetch_rates(HTF, HTF_CANDLES_LOOKBACK)
        df_ltf = connector.fetch_rates(LTF, LTF_CANDLES_LOOKBACK)
        if df_htf is None or df_ltf is None:
            return

        tick = connector.get_tick()
        if tick is None:
            return
        current_price = (tick.bid + tick.ask) / 2

        # 4. HTF trend
        trend = (
            SMCStrategy.get_htf_trend(
                df_htf, ctx.cfg.trend_ema_period,
                use_closed_candles=USE_CLOSED_CANDLES_ONLY)
            if ctx.cfg.use_trend_filter else "NEUTRAL"
        )
        ctx.state.trend = trend

        # 5. HTF POI
        poi = SMCStrategy.detect_htf_poi(
            df_htf=df_htf,
            use_trend_filter=ctx.cfg.use_trend_filter,
            ema_period=ctx.cfg.trend_ema_period,
            use_closed_candles=USE_CLOSED_CANDLES_ONLY,
        )
        ctx.state.poi = poi

        if poi is None:
            self._reset_watch(ctx)
            return

        # 6. Check if price is in zone
        if not SMCStrategy.is_zone_in_play(poi, df_ltf, current_price=current_price):
            self._reset_watch(ctx)
            return

        # 7. Watch timer
        zone_key = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
        if zone_key != ctx.state.watch_key:
            ctx.state.watch_key = zone_key
            ctx.state.watch_start = df_ltf.iloc[-1]["time"]
            ctx.state.abandoned = False

        if ctx.state.abandoned:
            return

        if ctx.state.watch_start is not None:
            waited = int((df_ltf["time"] >= ctx.state.watch_start).sum())
            if waited > MAX_LTF_WAIT_CANDLES:
                ctx.state.abandoned = True
                return

        # 7b. Sweep filter (XAUUSD uses this)
        if ctx.cfg.use_sweep_filter:
            sweep = SMCStrategy.detect_liquidity_sweep(
                df_ltf, swing_strength=SWING_STRENGTH,
                use_closed_candles=USE_CLOSED_CANDLES_ONLY)
            if sweep is None:
                return
            if poi.type == "BULLISH" and sweep.direction != "BULLISH":
                return
            if poi.type == "BEARISH" and sweep.direction != "BEARISH":
                return

        # 8. LTF confirmation (FVG)
        setup = SMCStrategy.check_ltf_confirmation(
            df_ltf, poi, ctx.cfg.rrr,
            use_closed_candles=USE_CLOSED_CANDLES_ONLY,
            stop_mode=ctx.cfg.stop_mode,
            buffer_atr=0.5,
        )
        if setup is None:
            return

        # 9. Invert if configured
        if ctx.cfg.invert_signals:
            setup = SMCStrategy.invert(setup, ctx.cfg.rrr)

        # 10. Find liquidity TP (skip if disabled per symbol)
        df_liq = connector.fetch_rates(LIQUIDITY_TF, LIQUIDITY_CANDLES)
        tp = setup["tp"]  # fallback
        if ctx.cfg.use_liquidity_tp and df_liq is not None:
            entry_price = connector.entry_price(
                mt5.ORDER_TYPE_BUY if setup["direction"] == "BUY"
                else mt5.ORDER_TYPE_SELL)
            if entry_price:
                liq = SMCStrategy.find_nearest_liquidity(
                    df_liq, setup["direction"], entry_price,
                    strength=SWING_STRENGTH, use_closed_candles=True)
                if liq is not None:
                    risk = abs(entry_price - setup["sl"])
                    liq_dist = abs(liq - entry_price)
                    liq_rr = liq_dist / risk if risk > 0 else 0
                    if liq_rr >= ctx.cfg.min_rr_liquidity:
                        tp = liq

        # 11. Risk validation
        entry_price = connector.entry_price(
            mt5.ORDER_TYPE_BUY if setup["direction"] == "BUY"
            else mt5.ORDER_TYPE_SELL)
        if entry_price is None:
            return

        rejection = validate_entry(ctx, setup["direction"],
                                    entry_price, setup["sl"], tp)
        if rejection:
            ctx.state.record_rejection(rejection)
            logger.info("[%s] REJECTED: %s", sym, rejection)
            return

        # 12. Calculate lot size
        lots = calculate_lot_size(ctx, entry_price, setup["sl"])
        if lots is None:
            ctx.state.record_rejection("SIZING_FAILED")
            return

        # 13. Execute
        logger.info(
            "[%s] SETUP: %s entry=%.5f sl=%.5f tp=%.5f lots=%.2f",
            sym, setup["direction"], entry_price, setup["sl"], tp, lots)

        order_type = (mt5.ORDER_TYPE_BUY if setup["direction"] == "BUY"
                      else mt5.ORDER_TYPE_SELL)

        # Re-anchor SL/TP on actual entry price
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
            comment=f"SMC_{setup['direction']}",
        )

        if success:
            logger.info("[%s] ORDER PLACED: %s %.2f lots", sym, setup["direction"], lots)
            account = mt5.account_info()
            bal = account.balance if account else 0
            telegram_bot.notify_trade_opened(
                sym, setup["direction"], lots,
                entry_price, setup["sl"], actual_tp,
                risk * lots * ctx.spec.contract_size, bal)
            self._reset_watch(ctx)
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
            at_entry = (pos.sl >= entry - 0.01 if is_buy
                        else pos.sl <= entry + 0.01)
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
