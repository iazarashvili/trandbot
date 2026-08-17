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
    LOT_SIZE,
    LTF,
    LTF_CANDLES_LOOKBACK,
    MAGIC_NUMBER,
    MAX_LTF_WAIT_CANDLES,
    MAX_SPREAD_POINTS,
    POLL_INTERVAL,
    RISK_PERCENT,
    RRR,
    STOP_BUFFER_ATR,
    STOP_MODE,
    SYMBOL,
    TREND_EMA_PERIOD,
    USE_BREAKEVEN,
    USE_CLOSED_CANDLES_ONLY,
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
    """Moves the stop to break-even once a position is far enough in profit."""
    if not USE_BREAKEVEN or not positions:
        return

    tick = connector.get_tick()
    if tick is None:
        return

    for position in positions:
        if not position.sl:
            continue

        risk = abs(position.price_open - position.sl)
        if risk <= 0:
            continue
        target = risk * BREAKEVEN_TRIGGER_R

        if position.type == mt5.POSITION_TYPE_BUY:
            if position.sl >= position.price_open:
                continue  # already at or beyond break-even
            if tick.bid >= position.price_open + target:
                logger.info(
                    "🛡️ Position #%s reached %.1fR — moving stop to break-even.",
                    position.ticket,
                    BREAKEVEN_TRIGGER_R,
                )
                connector.modify_position_sl(position, position.price_open)
        else:
            if position.sl <= position.price_open:
                continue
            if tick.ask <= position.price_open - target:
                logger.info(
                    "🛡️ Position #%s reached %.1fR — moving stop to break-even.",
                    position.ticket,
                    BREAKEVEN_TRIGGER_R,
                )
                connector.modify_position_sl(position, position.price_open)


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


def place_order(connector: MT5Connector, setup: dict) -> bool:
    """Turns a confirmed setup into a live market order."""
    order_type = (
        mt5.ORDER_TYPE_BUY if setup["direction"] == "BUY" else mt5.ORDER_TYPE_SELL
    )

    price = connector.entry_price(order_type)
    if price is None:
        logger.error("No live price available — skipping entry.")
        return False

    # Re-anchor the target on the price we will actually pay, otherwise the
    # configured RRR silently drifts with every tick since the signal candle.
    risk = abs(price - setup["sl"])
    if risk <= 0:
        logger.warning(
            "Price has already passed the stop (%.2f vs SL %.2f) — setup void.",
            price,
            setup["sl"],
        )
        return False

    tp = price + risk * RRR if order_type == mt5.ORDER_TYPE_BUY else price - risk * RRR
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
            "🔍 Price: %.2f | Trend: %s | ❌ No active 15m POI (OB+FVG) found "
            "in trend direction.",
            current_price,
            trend,
        )
        return

    if not SMCStrategy.is_zone_in_play(poi, df_ltf, current_price=current_price):
        watch.leave()
        logger.info(
            "🎯 15m POI Found (%s: %.2f - %.2f) | Price: %.2f | ⏳ Waiting for "
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
            "💤 15m %s zone abandoned after %s 1m candles without a shift.",
            poi.type,
            MAX_LTF_WAIT_CANDLES,
        )
        return

    if waited > MAX_LTF_WAIT_CANDLES:
        watch.abandoned = True
        logger.info(
            "⌛ Giving up on the 15m %s zone: %s/%s 1m candles elapsed with no "
            "MSS.",
            poi.type,
            waited,
            MAX_LTF_WAIT_CANDLES,
        )
        return

    logger.info(
        "⚡ Price inside 15m %s POI (%s/%s candles) — checking 1m for MSS + FVG...",
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
        logger.info("⌛ Inside 15m %s zone, but no 1m MSS/FVG confirmation yet.", poi.type)
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

    if place_order(connector, setup):
        logger.info("🎉 ORDER PLACED SUCCESSFULLY ON METATRADER 5!")
        watch.leave()


def main() -> None:
    connector = MT5Connector(symbol=SYMBOL, magic_number=MAGIC_NUMBER)
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
        SYMBOL,
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
