"""Smart Money Concepts (SMC) Strategy Analyzer with Trend Filter.

Pure functions over price DataFrames — no broker calls, no I/O — so the whole
strategy is unit-testable and back-testable offline.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

# Candles inspected on the LTF for the market structure shift and the stop.
_LTF_STRUCTURE_LOOKBACK = 10
# The 1m FVG is formed by the last three candles, so the MSS reference high /
# low is taken from the candles before it.
_LTF_FVG_CANDLES = 3


@dataclass(frozen=True)
class ZonePOI:
    type: Literal["BULLISH", "BEARISH"]
    top: float
    bottom: float
    index: int = -1  # position of the FVG's third candle in the HTF frame


class SMCStrategy:
    """Implements 1H HTF OB zone + 5m LTF displacement & FVG execution logic."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def closed_candles(
        df: pd.DataFrame, use_closed_candles: bool = True
    ) -> pd.DataFrame:
        """Drops the still-forming last candle.

        The live candle repaints: its close, high and low keep moving until the
        bar completes, so a signal read from it can appear and vanish within a
        single polling interval.
        """
        if use_closed_candles and len(df) > 1:
            return df.iloc[:-1]
        return df

    @staticmethod
    def is_zone_in_play(
        poi: ZonePOI,
        df: Optional[pd.DataFrame] = None,
        current_price: Optional[float] = None,
        lookback: int = 1,
    ) -> bool:
        """Single definition of "price is touching the POI".

        True when the live price sits inside the zone, or when any of the last
        `lookback` candles overlapped it.
        """
        if current_price is not None and poi.bottom <= current_price <= poi.top:
            return True

        if df is None or len(df) == 0 or lookback <= 0:
            return False

        window = df.iloc[-lookback:]
        overlaps = (window["low"] <= poi.top) & (window["high"] >= poi.bottom)
        return bool(overlaps.any())

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int = 100) -> pd.Series:
        """Calculates the Exponential Moving Average (EMA)."""
        return df["close"].ewm(span=period, adjust=False).mean()

    @staticmethod
    def get_htf_trend(
        df_htf: pd.DataFrame,
        period: int = 100,
        use_closed_candles: bool = True,
    ) -> str:
        """Determines the macro trend from price versus the EMA."""
        df = SMCStrategy.closed_candles(df_htf, use_closed_candles)
        if len(df) < period:
            return "NEUTRAL"

        ema = SMCStrategy.calculate_ema(df, period)
        current_close = df.iloc[-1]["close"]
        current_ema = ema.iloc[-1]

        if current_close > current_ema:
            return "BULLISH"
        if current_close < current_ema:
            return "BEARISH"
        return "NEUTRAL"

    # ------------------------------------------------------------------
    # HTF point of interest
    # ------------------------------------------------------------------
    @staticmethod
    def _is_fvg_mitigated(
        df: pd.DataFrame,
        index: int,
        poi_type: str,
        fvg_top: float,
        fvg_bottom: float,
    ) -> bool:
        """True once price has traded all the way back through the gap.

        A partially filled gap is still a valid point of interest — that is
        precisely the tap the bot is waiting for — but a gap price has fully
        closed is dead, and entering it means entering a level the market has
        already resolved.
        """
        future = df.iloc[index + 1 :]
        if len(future) == 0:
            return False

        if poi_type == "BULLISH":
            return bool((future["low"] <= fvg_bottom).any())
        return bool((future["high"] >= fvg_top).any())

    @staticmethod
    def detect_htf_poi(
        df_htf: pd.DataFrame,
        use_trend_filter: bool = True,
        ema_period: int = 100,
        use_closed_candles: bool = True,
    ) -> Optional[ZonePOI]:
        """Finds the most recent unmitigated 15m FVG + Order Block in trend."""
        df = SMCStrategy.closed_candles(df_htf, use_closed_candles)
        if len(df) < 4:
            return None

        trend = (
            SMCStrategy.get_htf_trend(df, ema_period, use_closed_candles=False)
            if use_trend_filter
            else "NEUTRAL"
        )

        # Walk backwards so the newest qualifying zone wins.  The range starts
        # at the last candle: an FVG that completed on the most recent bar is
        # the one most likely to still be unmitigated.
        for i in range(len(df) - 1, 2, -1):
            c1, c3 = df.iloc[i - 2], df.iloc[i]
            ob_candle = df.iloc[i - 3]

            # ----------------------------------------------------------
            # BULLISH POI (only if the trend is BULLISH or the filter is off)
            # ----------------------------------------------------------
            if trend in ("BULLISH", "NEUTRAL") and c3["low"] > c1["high"]:
                fvg_bottom = c1["high"]
                fvg_top = c3["low"]

                if ob_candle["close"] < ob_candle["open"]:  # last down candle
                    ob_top = max(ob_candle["high"], ob_candle["open"])
                    ob_bottom = ob_candle["low"]

                    if fvg_bottom <= ob_top and not SMCStrategy._is_fvg_mitigated(
                        df, i, "BULLISH", fvg_top, fvg_bottom
                    ):
                        return ZonePOI(
                            type="BULLISH",
                            top=max(fvg_top, ob_top),
                            bottom=min(fvg_bottom, ob_bottom),
                            index=i,
                        )

            # ----------------------------------------------------------
            # BEARISH POI (only if the trend is BEARISH or the filter is off)
            # ----------------------------------------------------------
            if trend in ("BEARISH", "NEUTRAL") and c3["high"] < c1["low"]:
                fvg_top = c1["low"]
                fvg_bottom = c3["high"]

                if ob_candle["close"] > ob_candle["open"]:  # last up candle
                    ob_top = ob_candle["high"]
                    ob_bottom = min(ob_candle["low"], ob_candle["close"])

                    if fvg_top >= ob_bottom and not SMCStrategy._is_fvg_mitigated(
                        df, i, "BEARISH", fvg_top, fvg_bottom
                    ):
                        return ZonePOI(
                            type="BEARISH",
                            top=max(fvg_top, ob_top),
                            bottom=min(fvg_bottom, ob_bottom),
                            index=i,
                        )

        return None

    @staticmethod
    def invert(setup: dict, rrr: float) -> dict:
        """Mirrors a setup: same entry, same 1R distance, opposite direction.

        The stop and target are reflected around the entry rather than rebuilt
        from the opposite side's structure, so risk per trade is identical to
        the setup being inverted and the two modes can be compared directly.

        Note what this does *not* do: inverting a 1:R system does not invert
        its win rate.  A losing long only tells us price reached -1R before
        +3R; the mirrored short still has to reach -3R before +1R, which is a
        different and stricter condition.
        """
        entry = setup["entry"]
        risk = abs(entry - setup["sl"])

        if setup["direction"] == "BUY":
            return {
                "direction": "SELL",
                "entry": entry,
                "sl": entry + risk,
                "tp": entry - risk * rrr,
                "inverted_from": "BUY",
            }
        return {
            "direction": "BUY",
            "entry": entry,
            "sl": entry - risk,
            "tp": entry + risk * rrr,
            "inverted_from": "SELL",
        }

    # ------------------------------------------------------------------
    # Liquidity levels (swing highs / swing lows)
    # ------------------------------------------------------------------
    @staticmethod
    def find_swing_levels(
        df: pd.DataFrame,
        strength: int = 3,
        use_closed_candles: bool = True,
    ) -> tuple[List[float], List[float]]:
        """Returns (swing_highs, swing_lows) from a price DataFrame.

        A swing high is a bar whose high is the highest of the surrounding
        `strength` bars on each side.  Swing lows are the mirror.
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < 2 * strength + 1:
            return [], []

        highs = data["high"].to_numpy(float)
        lows = data["low"].to_numpy(float)
        swing_highs: List[float] = []
        swing_lows: List[float] = []

        for i in range(strength, len(data) - strength):
            if highs[i] == np.max(highs[i - strength: i + strength + 1]):
                swing_highs.append(float(highs[i]))
            if lows[i] == np.min(lows[i - strength: i + strength + 1]):
                swing_lows.append(float(lows[i]))

        return swing_highs, swing_lows

    @staticmethod
    def find_nearest_liquidity(
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        strength: int = 3,
        use_closed_candles: bool = True,
    ) -> Optional[float]:
        """Finds the nearest liquidity target for a trade.

        BUY  -> nearest swing high ABOVE entry (buyside liquidity)
        SELL -> nearest swing low BELOW entry (sellside liquidity)
        """
        swing_highs, swing_lows = SMCStrategy.find_swing_levels(
            df, strength, use_closed_candles)

        if direction == "BUY":
            above = [h for h in swing_highs if h > entry_price]
            return min(above) if above else None

        below = [l for l in swing_lows if l < entry_price]
        return max(below) if below else None

    @staticmethod
    def find_next_liquidity(
        df: pd.DataFrame,
        direction: str,
        beyond_price: float,
        strength: int = 3,
        use_closed_candles: bool = True,
    ) -> Optional[float]:
        """Finds the next liquidity level BEYOND a given price.

        Used for the extended TP after partial close — the remaining runner
        targets the next structural level past the original TP.

        BUY  -> next swing high ABOVE beyond_price
        SELL -> next swing low BELOW beyond_price
        """
        swing_highs, swing_lows = SMCStrategy.find_swing_levels(
            df, strength, use_closed_candles)

        if direction == "BUY":
            above = [h for h in swing_highs if h > beyond_price]
            return min(above) if above else None

        below = [l for l in swing_lows if l < beyond_price]
        return max(below) if below else None

    # ------------------------------------------------------------------
    # LTF confirmation
    # ------------------------------------------------------------------
    @staticmethod
    def _stop_level(
        window: pd.DataFrame,
        structure: pd.DataFrame,
        poi: ZonePOI,
        bullish: bool,
        stop_mode: str,
        buffer_atr: float,
    ) -> float:
        """Where the setup is invalidated, under the chosen placement rule.

        "window"    the extreme of the whole lookback — the original rule.  It
                    is a *window* rule: the distance it produces depends on how
                    noisy the last N candles happened to be, not on where the
                    trade idea actually dies.
        "swing"     the extreme since the swing the MSS broke, plus a buffer.
                    This is the level that has to give way for the shift to be
                    wrong.
        "zone"      as "swing", but never tighter than the far side of the HTF
                    zone — the trade idea only really dies once price leaves
                    the point of interest altogether.
        """
        if stop_mode == "window":
            return float(window["low"].min() if bullish else window["high"].max())

        # Average range over the lookback, used as the wick buffer and ATR stop.
        atr = float((window["high"] - window["low"]).mean())

        if stop_mode == "atr":
            last_close = float(window.iloc[-1]["close"])
            return (last_close - atr * buffer_atr) if bullish else (last_close + atr * buffer_atr)

        if stop_mode == "ob":
            # Stop just beyond the OB zone boundary + small ATR buffer
            buffer = atr * buffer_atr
            if bullish:
                return float(poi.bottom) - buffer
            return float(poi.top) + buffer

        buffer = atr * buffer_atr

        if bullish:
            pivot = int(structure["high"].to_numpy().argmax())
            level = float(window["low"].to_numpy()[pivot:].min())
            if stop_mode == "zone":
                level = min(level, float(poi.bottom))
            return level - buffer

        pivot = int(structure["low"].to_numpy().argmin())
        level = float(window["high"].to_numpy()[pivot:].max())
        if stop_mode == "zone":
            level = max(level, float(poi.top))
        return level + buffer

    @staticmethod
    def check_ltf_confirmation(
        df_ltf: pd.DataFrame,
        poi: ZonePOI,
        rrr: float,
        use_closed_candles: bool = True,
        stop_mode: str = "window",
        buffer_atr: float = 0.5,
    ) -> Optional[dict]:
        """Checks for a displacement FVG while price sits inside the HTF OB zone."""
        df = SMCStrategy.closed_candles(df_ltf, use_closed_candles)
        if len(df) < _LTF_STRUCTURE_LOOKBACK:
            return None

        signal = df.iloc[-1]
        fvg_first = df.iloc[-_LTF_FVG_CANDLES]
        structure = df.iloc[-_LTF_STRUCTURE_LOOKBACK:-_LTF_FVG_CANDLES]
        window = df.iloc[-_LTF_STRUCTURE_LOOKBACK:]

        if not SMCStrategy.is_zone_in_play(poi, df, lookback=1):
            return None

        if poi.type == "BULLISH":
            has_fvg = signal["low"] > fvg_first["high"]
            if not has_fvg:
                return None

            entry_price = float(signal["close"])
            sl_price = SMCStrategy._stop_level(
                window, structure, poi, True, stop_mode, buffer_atr)
            risk = entry_price - sl_price
            if risk <= 0:
                return None

            return {
                "direction": "BUY",
                "entry": entry_price,
                "sl": sl_price,
                "tp": entry_price + risk * rrr,
            }

        if poi.type == "BEARISH":
            has_fvg = signal["high"] < fvg_first["low"]
            if not has_fvg:
                return None

            entry_price = float(signal["close"])
            sl_price = SMCStrategy._stop_level(
                window, structure, poi, False, stop_mode, buffer_atr)
            risk = sl_price - entry_price
            if risk <= 0:
                return None

            return {
                "direction": "SELL",
                "entry": entry_price,
                "sl": sl_price,
                "tp": entry_price - risk * rrr,
            }

        return None
