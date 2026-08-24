"""Main Trading Bot Loop with Detailed Diagnostic Logging."""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import MetaTrader5 as mt5
import pandas as pd

from config import (
    BREAKEVEN_TRIGGER_R,
    HTF,
    HTF_CANDLES_LOOKBACK,
    INVERT_SIGNALS,
    LIQUIDITY_CANDLES,
    LIQUIDITY_TF,
    LOT_SIZE,
    NIGHT_END_HOUR,
    NIGHT_START_HOUR,
    LTF,
    LTF_CANDLES_LOOKBACK,
    MAGIC_NUMBER,
    MAX_LTF_WAIT_CANDLES,
    MAX_RISK_PCT,
    MAX_RISK_USD,
    MAX_SPREAD_POINTS,
    MIN_RRR_LIQUIDITY,
    PARTIAL_CLOSE_PCT,
    PARTIAL_TRIGGER_PCT,
    POLL_INTERVAL,
    RISK_PERCENT,
    RRR,
    SKIP_WEEKENDS,
    STOP_BUFFER_ATR,
    STOP_MODE,
    SWING_STRENGTH,
    SYMBOL,
    TREND_EMA_PERIOD,
    USE_BREAKEVEN,
    USE_CLOSED_CANDLES_ONLY,
    USE_PARTIAL_CLOSE,
    USE_RISK_BASED_LOT,
    USE_TREND_FILTER,
)
from mt5_connector import MT5Connector
from strategy import SMCStrategy, ZonePOI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smc_bot")

# Wait this long before retrying after a connection loss or an unexpected error.
RECOVERY_SLEEP = 30


@dataclass
class PoiWatch:
    """Tracks how long price has been sitting inside the current POI.

    Gives MAX_LTF_WAIT_CANDLES its meaning: a zone that price has been
    grinding inside for a long time without producing a shift has lost its
    edge, so the bot stops trying until price leaves and comes back.
    """

    key: Optional[tuple] = None
    entered_at: Optional[pd.Timestamp] = None
    abandoned: bool = False

    def enter(self, poi: ZonePOI, candle_time) -> None:
        key = (poi.type, poi.top, poi.bottom)
        if key != self.key:
            self.key = key
            self.entered_at = candle_time
            self.abandoned = False

    def candles_waited(self, df_ltf: pd.DataFrame) -> int:
        if self.entered_at is None:
            return 0
        return int((df_ltf["time"] >= self.entered_at).sum())

    def leave(self) -> None:
        self.key = None
        self.entered_at = None
        self.abandoned = False


def manage_open_positions(connector: MT5Connector, positions: Sequence) -> None:
    """Manages open positions: breakeven stops and partial closes."""
    tick = connector.get_tick()
    if tick is None:
        return

    for position in positions:
        if not position.sl or not position.tp:
            continue

        entry = position.price_open
        risk = abs(entry - position.sl)
        if risk <= 0:
            continue

        is_buy = position.type == mt5.POSITION_TYPE_BUY
        current = tick.bid if is_buy else tick.ask

        # ---- Partial close at 80% of TP ----
        already_runner = (position.sl >= entry - 0.01 if is_buy
                          else position.sl <= entry + 0.01)
        if USE_PARTIAL_CLOSE and not already_runner:
            tp_dist = abs(position.tp - entry)
            trigger_price = (entry + tp_dist * PARTIAL_TRIGGER_PCT if is_buy
                             else entry - tp_dist * PARTIAL_TRIGGER_PCT)

            triggered = (current >= trigger_price if is_buy
                         else current <= trigger_price)

            if triggered:
                info = connector.symbol_info()
                vol_min = info.volume_min if info else 0.01
                can_split = position.volume > vol_min

                if can_split:
                    # Enough volume to split: close half, keep runner
                    close_vol = connector.normalize_volume(position.volume * PARTIAL_CLOSE_PCT)
                    logger.info(
                        "📊 Position #%s reached 80%% of TP — partial close %.2f lots.",
                        position.ticket, close_vol)

                    if connector.partial_close(position, close_vol):
                        # After partial close MT5 creates a new ticket for the
                        # remaining volume — re-fetch to get the valid ticket.
                        remaining = connector.get_open_positions()
                        if not remaining:
                            logger.warning("No remaining position after partial close.")
                            break
                        position = remaining[0]

                        # Find next liquidity level for the runner
                        df_liq = connector.fetch_rates(LIQUIDITY_TF, LIQUIDITY_CANDLES)
                        new_tp = position.tp  # fallback
                        if df_liq is not None:
                            direction = "BUY" if is_buy else "SELL"
                            next_liq = SMCStrategy.find_next_liquidity(
                                df_liq, direction, position.tp,
                                strength=SWING_STRENGTH, use_closed_candles=True)
                            if next_liq is not None:
                                new_tp = next_liq
                                logger.info(
                                    "🎯 Runner TP moved to next liquidity: %.2f", new_tp)
                            else:
                                logger.info(
                                    "🎯 No next liquidity found — keeping original TP.")

                        # Move stop to entry, TP to next liquidity
                        connector.modify_position_sl_tp(position, entry, new_tp)
                        logger.info(
                            "🏃 Runner: SL→%.2f (entry), TP→%.2f", entry, new_tp)
                else:
                    # Minimum lot — can't split, just move SL to entry
                    connector.modify_position_sl(position, entry)
                    logger.info(
                        "📊 Position #%s min lot (%.2f) — SL moved to entry (breakeven).",
                        position.ticket, position.volume)
                continue

        # ---- Breakeven stop (if enabled separately) ----
        if USE_BREAKEVEN:
            target = risk * BREAKEVEN_TRIGGER_R
            at_or_beyond_be = (position.sl >= entry if is_buy
                               else position.sl <= entry)
            if at_or_beyond_be:
                continue
            if is_buy and current >= entry + target:
                logger.info(
                    "🛡️ Position #%s reached %.1fR — moving stop to break-even.",
                    position.ticket, BREAKEVEN_TRIGGER_R)
                connector.modify_position_sl(position, entry)
            elif not is_buy and current <= entry - target:
                logger.info(
                    "🛡️ Position #%s reached %.1fR — moving stop to break-even.",
                    position.ticket, BREAKEVEN_TRIGGER_R)
                connector.modify_position_sl(position, entry)


def spread_is_acceptable(connector: MT5Connector) -> bool:
    """Blocks entries while the spread is wider than configured."""
    spread = connector.get_spread_points()
    if spread is None:
        return False
    if MAX_SPREAD_POINTS > 0 and spread > MAX_SPREAD_POINTS:
        logger.warning(
            "🚫 Spread too wide (%.0f > %s points) — skipping entry.",
            spread,
            MAX_SPREAD_POINTS,
        )
        return False
    return True


def place_order(connector: MT5Connector, setup: dict,
                df_liq: Optional[pd.DataFrame] = None) -> bool:
    """Turns a confirmed setup into a live market order."""
    order_type = (
        mt5.ORDER_TYPE_BUY if setup["direction"] == "BUY" else mt5.ORDER_TYPE_SELL
    )

    price = connector.entry_price(order_type)
    if price is None:
        logger.error("No live price available — skipping entry.")
        return False

    risk = abs(price - setup["sl"])
    if risk <= 0:
        logger.warning(
            "Price has already passed the stop (%.2f vs SL %.2f) — setup void.",
            price,
            setup["sl"],
        )
        return False

    risk_usd = risk * LOT_SIZE
    # Percentage-based risk cap (uses account balance)
    if MAX_RISK_PCT > 0:
        account = mt5.account_info()
        if account:
            max_allowed = account.balance * MAX_RISK_PCT / 100
            if risk_usd > max_allowed:
                logger.warning(
                    "🚫 Risk $%.2f exceeds %.1f%% of balance ($%.2f) — skipping.",
                    risk_usd, MAX_RISK_PCT, max_allowed)
                return False
    # Hard dollar cap fallback
    elif MAX_RISK_USD > 0 and risk_usd > MAX_RISK_USD:
        logger.warning(
            "🚫 Risk $%.2f exceeds MAX_RISK_USD $%.0f — skipping entry.",
            risk_usd, MAX_RISK_USD)
        return False

    # Try liquidity-based TP first, fall back to fixed RRR.
    tp = None
    if df_liq is not None:
        liq = SMCStrategy.find_nearest_liquidity(
            df_liq, setup["direction"], price,
            strength=SWING_STRENGTH, use_closed_candles=True)
        if liq is not None:
            liq_dist = abs(liq - price)
            liq_rrr = liq_dist / risk if risk > 0 else 0
            if liq_rrr >= MIN_RRR_LIQUIDITY:
                tp = liq
                logger.info(
                    "🎯 TP at 15m liquidity level %.2f (%.1fR from entry).",
                    tp, liq_rrr)
            else:
                logger.info(
                    "🎯 Nearest liquidity %.2f is only %.2fR — using fixed RRR.",
                    liq, liq_rrr)

    if tp is None:
        tp = price + risk * RRR if order_type == mt5.ORDER_TYPE_BUY else price - risk * RRR
        logger.info("🎯 No liquidity level found — using fixed RRR %.1f, TP %.2f.", RRR, tp)

    levels = connector.normalize_levels(order_type, price, setup["sl"], tp)

    volume = LOT_SIZE
    if USE_RISK_BASED_LOT:
        volume = connector.calculate_volume(
            order_type, levels.price, levels.sl, RISK_PERCENT
        )
        if volume is None:
            logger.error("Position sizing failed — skipping entry.")
            return False
        logger.info(
            "📐 Risking %.2f%% of balance → %s lots.", RISK_PERCENT, volume
        )

    return connector.send_order(
        order_type=order_type,
        sl=levels.sl,
        tp=levels.tp,
        volume=volume,
        comment=f"SMC_{setup['direction']}",
    )


def run_cycle(connector: MT5Connector, watch: PoiWatch) -> None:
    """Executes one poll of the market. Raises nothing the caller must handle."""
    # 0. Weekend & night filter — no new entries outside trading hours.
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)

    skip_reason = None
    if SKIP_WEEKENDS and now.weekday() >= 5:
        skip_reason = "Weekend (UTC %s)" % now.strftime("%A")
    elif NIGHT_START_HOUR > NIGHT_END_HOUR:  # wraps midnight: e.g. 23-04
        if now.hour >= NIGHT_START_HOUR or now.hour < NIGHT_END_HOUR:
            skip_reason = "Night session (UTC %02d:%02d)" % (now.hour, now.minute)
    elif NIGHT_START_HOUR <= now.hour < NIGHT_END_HOUR:
        skip_reason = "Night session (UTC %02d:%02d)" % (now.hour, now.minute)

    if skip_reason:
        positions = connector.get_open_positions()
        if positions:
            manage_open_positions(connector, positions)
        logger.info("🌙 %s — skipping new entries.", skip_reason)
        return

    # 1. Existing positions take priority over new entries.
    positions = connector.get_open_positions()
    if positions:
        manage_open_positions(connector, positions)
        watch.leave()
        logger.info("⏳ Waiting... Existing open position detected (%s).", len(positions))
        return

    # 2. Market data.
    df_htf = connector.fetch_rates(HTF, HTF_CANDLES_LOOKBACK)
    df_ltf = connector.fetch_rates(LTF, LTF_CANDLES_LOOKBACK)
    if df_htf is None or df_ltf is None:
        logger.warning("⚠️ Failed to fetch price data from MT5. Retrying...")
        return

    tick = connector.get_tick()
    if tick is None:
        logger.warning("⚠️ No live quote available. Retrying...")
        return
    current_price = (tick.bid + tick.ask) / 2

    # 3. Macro trend.
    trend = (
        SMCStrategy.get_htf_trend(
            df_htf, TREND_EMA_PERIOD, use_closed_candles=USE_CLOSED_CANDLES_ONLY
        )
        if USE_TREND_FILTER
        else "NEUTRAL"
    )

    # 4. HTF point of interest.
    poi = SMCStrategy.detect_htf_poi(
        df_htf=df_htf,
        use_trend_filter=USE_TREND_FILTER,
        ema_period=TREND_EMA_PERIOD,
        use_closed_candles=USE_CLOSED_CANDLES_ONLY,
    )

    if poi is None:
        watch.leave()
        logger.info(
            "🔍 Price: %.2f | Trend: %s | ❌ No active 1H POI (OB+FVG) found "
            "in trend direction.",
            current_price,
            trend,
        )
        return

    if not SMCStrategy.is_zone_in_play(poi, df_ltf, current_price=current_price):
        watch.leave()
        logger.info(
            "🎯 1H POI Found (%s: %.2f - %.2f) | Price: %.2f | ⏳ Waiting for "
            "price to enter zone...",
            poi.type,
            poi.bottom,
            poi.top,
            current_price,
        )
        return

    # 5. Price is inside the zone — start (or continue) the LTF countdown.
    watch.enter(poi, df_ltf.iloc[-1]["time"])
    waited = watch.candles_waited(df_ltf)

    if watch.abandoned:
        logger.info(
            "💤 1H %s zone abandoned after %s 5m candles without FVG.",
            poi.type,
            MAX_LTF_WAIT_CANDLES,
        )
        return

    if waited > MAX_LTF_WAIT_CANDLES:
        watch.abandoned = True
        logger.info(
            "⌛ Giving up on the 1H %s zone: %s/%s 5m candles elapsed with no "
            "FVG.",
            poi.type,
            waited,
            MAX_LTF_WAIT_CANDLES,
        )
        return

    logger.info(
        "⚡ Price inside 1H %s POI (%s/%s candles) — checking 5m for FVG...",
        poi.type,
        waited,
        MAX_LTF_WAIT_CANDLES,
    )

    # 6. LTF confirmation.
    setup = SMCStrategy.check_ltf_confirmation(
        df_ltf,
        poi,
        RRR,
        use_closed_candles=USE_CLOSED_CANDLES_ONLY,
        stop_mode=STOP_MODE,
        buffer_atr=STOP_BUFFER_ATR,
    )
    if setup is None:
        logger.info("⌛ Inside 1H %s zone, but no 5m FVG confirmation yet.", poi.type)
        return

    if INVERT_SIGNALS:
        original = setup["direction"]
        setup = SMCStrategy.invert(setup, RRR)
        logger.info(
            "🔄 INVERT_SIGNALS is on — trading the %s signal as a %s.",
            original,
            setup["direction"],
        )

    logger.info(
        "🔥 SETUP CONFIRMED! Direction: %s | Signal close: %.2f | SL: %.2f | "
        "TP (at signal): %.2f",
        setup["direction"],
        setup["entry"],
        setup["sl"],
        setup["tp"],
    )

    if not spread_is_acceptable(connector):
        return

    # Fetch 15m data for liquidity-based TP.
    df_liq = connector.fetch_rates(LIQUIDITY_TF, LIQUIDITY_CANDLES)

    if place_order(connector, setup, df_liq=df_liq):
        logger.info("🎉 ORDER PLACED SUCCESSFULLY ON METATRADER 5!")
        watch.leave()


def main() -> None:
    import argparse
    import importlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default=None,
                    help="Override SYMBOL from config (e.g. XAUUSD)")
    ap.add_argument("--config", type=str, default=None,
                    help="Symbol-specific config module (e.g. config_xauusd)")
    args = ap.parse_args()

    # Load symbol-specific config if provided
    if args.config:
        cfg = importlib.import_module(args.config)
    elif args.symbol and args.symbol != SYMBOL:
        try:
            cfg = importlib.import_module(f"config_{args.symbol.lower()}")
        except ModuleNotFoundError:
            cfg = None
    else:
        cfg = None

    # Pull values from symbol config or defaults
    if cfg:
        symbol = getattr(cfg, "SYMBOL", SYMBOL)
        magic = getattr(cfg, "MAGIC_NUMBER", MAGIC_NUMBER)
        # Update globals used by this module
        global LOT_SIZE, MAX_RISK_USD, MAX_RISK_PCT, NIGHT_START_HOUR, NIGHT_END_HOUR
        LOT_SIZE = getattr(cfg, "LOT_SIZE", LOT_SIZE)
        MAX_RISK_USD = getattr(cfg, "MAX_RISK_USD", MAX_RISK_USD)
        MAX_RISK_PCT = getattr(cfg, "MAX_RISK_PCT", MAX_RISK_PCT)
        NIGHT_START_HOUR = getattr(cfg, "NIGHT_START_HOUR", NIGHT_START_HOUR)
        NIGHT_END_HOUR = getattr(cfg, "NIGHT_END_HOUR", NIGHT_END_HOUR)
        logger.info("Loaded config: %s", args.config or f"config_{symbol.lower()}")
    else:
        symbol = args.symbol or SYMBOL
        magic = MAGIC_NUMBER

    connector = MT5Connector(symbol=symbol, magic_number=magic)
    if not connector.initialize():
        return

    sizing = (
        f"{RISK_PERCENT}% risk" if USE_RISK_BASED_LOT else f"{LOT_SIZE} lots"
    )
    logger.info("==========================================")
    logger.info("🚀 SMC Bot Event Loop Started...")
    logger.info(
        "Target: %s | Size: %s | RRR: 1:%s | Trend Filter: %s (%s EMA) | "
        "Break-even: %s",
        symbol,
        sizing,
        RRR,
        USE_TREND_FILTER,
        TREND_EMA_PERIOD,
        USE_BREAKEVEN,
    )
    if INVERT_SIGNALS:
        logger.info(
            "🔄 INVERT_SIGNALS: ON — every signal is traded in the OPPOSITE "
            "direction. Set INVERT_SIGNALS = False in config.py to revert."
        )
    logger.info("==========================================")

    watch = PoiWatch()

    try:
        while True:
            try:
                if not connector.is_connected():
                    if not connector.reconnect():
                        logger.error(
                            "Reconnect failed — retrying in %ss.", RECOVERY_SLEEP
                        )
                        time.sleep(RECOVERY_SLEEP)
                        continue
                    logger.info("✅ Reconnected to MT5.")

                run_cycle(connector, watch)
                time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001 - the loop must outlive any single cycle
                # A dropped connection or a malformed tick must not take the
                # bot down while a position is open and unmanaged.
                logger.exception("Unexpected error in trading cycle — recovering.")
                time.sleep(RECOVERY_SLEEP)

    except KeyboardInterrupt:
        logger.info("🛑 Bot execution stopped manually.")
    finally:
        connector.shutdown()


if __name__ == "__main__":
    main()
