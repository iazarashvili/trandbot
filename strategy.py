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
class SessionRange:
    """High/low of a trading session (e.g. Asian session)."""
    high: float
    low: float
    start_time: object = None
    end_time: object = None


@dataclass(frozen=True)
class BreakerBlock:
    """A broken Order Block that flips to support/resistance."""
    type: Literal["BULLISH", "BEARISH"]  # BULLISH = old bearish OB broken upward
    top: float
    bottom: float
    break_index: int = -1


@dataclass(frozen=True)
class StructureShift:
    """A confirmed market structure shift (MSS) or break of structure (BOS)."""
    kind: Literal["BOS", "MSS"]
    direction: Literal["BULLISH", "BEARISH"]
    break_level: float
    break_index: int


@dataclass(frozen=True)
class EqualLevel:
    """Equal highs or equal lows — liquidity pools where stops accumulate."""
    type: Literal["EQH", "EQL"]  # EQH = equal highs, EQL = equal lows
    level: float
    count: int  # how many swing points at this level


@dataclass(frozen=True)
class LiquiditySweep:
    """A confirmed liquidity sweep — price breached a swing level and reversed."""
    direction: Literal["BULLISH", "BEARISH"]  # BULLISH = swept sellside, BEARISH = swept buyside
    liquidity_price: float      # the swing level that was swept
    sweep_low: float            # how far price went past (for bullish)
    sweep_high: float           # how far price went past (for bearish)
    sweep_depth: float          # how much past the level
    sweep_bar: int              # index in the DataFrame
    rejection: bool             # did price close back above/below


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
    # Liquidity Sweep
    # ------------------------------------------------------------------
    @staticmethod
    def detect_liquidity_sweep(
        df: pd.DataFrame,
        swing_strength: int = 3,
        sweep_tolerance_atr: float = 0.10,
        lookback: int = 50,
        use_closed_candles: bool = True,
    ) -> Optional[LiquiditySweep]:
        """Detects the most recent liquidity sweep on the given timeframe.

        BULLISH sweep: price dips below a swing low (sellside liquidity) and
        closes back above it — smart money grabbed the stops below.

        BEARISH sweep: price spikes above a swing high (buyside liquidity) and
        closes back below it — smart money grabbed the stops above.
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < lookback:
            return None

        window = data.iloc[-lookback:]
        highs = window["high"].to_numpy(float)
        lows = window["low"].to_numpy(float)
        closes = window["close"].to_numpy(float)
        opens = window["open"].to_numpy(float)

        atr = float((window["high"] - window["low"]).mean())
        min_breach = atr * sweep_tolerance_atr

        # Find swing highs and lows
        swing_highs_idx = []
        swing_lows_idx = []
        for i in range(swing_strength, len(window) - swing_strength):
            if highs[i] == np.max(highs[i - swing_strength: i + swing_strength + 1]):
                swing_highs_idx.append(i)
            if lows[i] == np.min(lows[i - swing_strength: i + swing_strength + 1]):
                swing_lows_idx.append(i)

        # Check recent candles for sweeps (last 10 candles)
        scan_from = max(0, len(window) - 10)

        # BULLISH sweep — check sellside (swing lows)
        for si in reversed(swing_lows_idx):
            level = lows[si]
            for t in range(max(si + 1, scan_from), len(window)):
                if lows[t] < level - min_breach and closes[t] > level:
                    return LiquiditySweep(
                        direction="BULLISH",
                        liquidity_price=float(level),
                        sweep_low=float(lows[t]),
                        sweep_high=0.0,
                        sweep_depth=float(level - lows[t]),
                        sweep_bar=t,
                        rejection=True,
                    )

        # BEARISH sweep — check buyside (swing highs)
        for si in reversed(swing_highs_idx):
            level = highs[si]
            for t in range(max(si + 1, scan_from), len(window)):
                if highs[t] > level + min_breach and closes[t] < level:
                    return LiquiditySweep(
                        direction="BEARISH",
                        liquidity_price=float(level),
                        sweep_low=0.0,
                        sweep_high=float(highs[t]),
                        sweep_depth=float(highs[t] - level),
                        sweep_bar=t,
                        rejection=True,
                    )

        return None

    # ------------------------------------------------------------------
    # Trend & Consolidation
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int = 100) -> pd.Series:
        """Calculates the Exponential Moving Average (EMA)."""
        return df["close"].ewm(span=period, adjust=False).mean()

    @staticmethod
    def is_consolidating(
        df: pd.DataFrame,
        atr_period: int = 14,
        threshold: float = 0.5,
        use_closed_candles: bool = True,
    ) -> bool:
        """True when the market is ranging / low volatility.

        Compares recent ATR (last 5 bars) to the longer ATR (atr_period bars).
        If recent ATR < threshold * longer ATR, the market is consolidating.
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < atr_period:
            return False

        ranges = (data["high"] - data["low"]).to_numpy(float)
        long_atr = float(ranges[-atr_period:].mean())
        short_atr = float(ranges[-5:].mean())

        if long_atr <= 0:
            return False
        return short_atr < long_atr * threshold

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

        # High-volatility filter (LuxAlgo): skip OB candles with range >= 2*ATR
        ranges = (df["high"] - df["low"]).to_numpy(float)
        atr = float(pd.Series(ranges).rolling(min(200, len(df))).mean().iloc[-1])

        # Walk backwards so the newest qualifying zone wins.
        for i in range(len(df) - 1, 2, -1):
            c1, c3 = df.iloc[i - 2], df.iloc[i]
            ob_candle = df.iloc[i - 3]

            # Skip high-volatility OB candles (LuxAlgo filter)
            ob_range = float(ob_candle["high"] - ob_candle["low"])
            if ob_range >= 2 * atr:
                continue

            # ----------------------------------------------------------
            # BULLISH POI (trend must be BULLISH, or filter is off)
            # ----------------------------------------------------------
            if (trend == "BULLISH" or not use_trend_filter) and c3["low"] > c1["high"]:
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
            # BEARISH POI (trend must be BEARISH, or filter is off)
            # ----------------------------------------------------------
            if (trend == "BEARISH" or not use_trend_filter) and c3["high"] < c1["low"]:
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
    # Setup Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def score_setup(
        df_htf: pd.DataFrame,
        df_ltf: pd.DataFrame,
        poi: Optional[ZonePOI],
        setup: Optional[dict],
        sweep: Optional[LiquiditySweep],
        trend: str,
        rrr: float = 3.0,
        use_closed_candles: bool = True,
    ) -> dict:
        """Scores a setup 0-100 based on confluence of SMC factors.

        Returns {"score": int, "factors": {name: points, ...}}.
        """
        factors = {}
        ltf = SMCStrategy.closed_candles(df_ltf, use_closed_candles)

        # 1. Liquidity sweep confirmed (+20)
        if sweep is not None:
            factors["liquidity_sweep"] = 20
        else:
            factors["liquidity_sweep"] = 0

        # 2. Displacement quality (+20)
        if setup and len(ltf) >= 3:
            signal = ltf.iloc[-1]
            atr = float((ltf["high"] - ltf["low"]).mean())
            body = abs(signal["close"] - signal["open"])
            full_range = signal["high"] - signal["low"]
            body_ratio = body / full_range if full_range > 0 else 0
            body_atr = body / atr if atr > 0 else 0

            disp_score = 0
            if body_atr >= 1.0:
                disp_score += 10
            elif body_atr >= 0.5:
                disp_score += 5
            if body_ratio >= 0.65:
                disp_score += 10
            elif body_ratio >= 0.50:
                disp_score += 5
            factors["displacement"] = disp_score
        else:
            factors["displacement"] = 0

        # 3. FVG exists (+15)
        if setup is not None:
            factors["fvg"] = 15
        else:
            factors["fvg"] = 0

        # 4. HTF alignment (+15)
        if poi and setup:
            aligned = ((poi.type == "BULLISH" and trend == "BULLISH") or
                       (poi.type == "BEARISH" and trend == "BEARISH"))
            factors["htf_alignment"] = 15 if aligned else 5  # neutral gets 5
        else:
            factors["htf_alignment"] = 0

        # 5. POI zone exists (+10)
        factors["poi_zone"] = 10 if poi else 0

        # 6. RR quality (+10)
        if setup:
            entry = setup["entry"]
            sl = setup["sl"]
            tp = setup["tp"]
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0
            if rr >= 3.0:
                factors["rr_quality"] = 10
            elif rr >= 2.5:
                factors["rr_quality"] = 7
            elif rr >= 2.0:
                factors["rr_quality"] = 5
            else:
                factors["rr_quality"] = 0
        else:
            factors["rr_quality"] = 0

        # 7. Premium/Discount (+10)
        if poi and setup and len(ltf) >= 20:
            recent = ltf.iloc[-20:]
            range_high = float(recent["high"].max())
            range_low = float(recent["low"].min())
            equilibrium = (range_high + range_low) / 2
            price = setup["entry"]

            if poi.type == "BULLISH" and price < equilibrium:
                factors["premium_discount"] = 10  # buying in discount
            elif poi.type == "BEARISH" and price > equilibrium:
                factors["premium_discount"] = 10  # selling in premium
            else:
                factors["premium_discount"] = 3
        else:
            factors["premium_discount"] = 0

        score = sum(factors.values())
        return {"score": score, "factors": factors}

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
            fvg_bottom = float(fvg_first["high"])  # candle 1 high
            fvg_top = float(signal["low"])          # candle 3 low
            has_fvg = fvg_top > fvg_bottom
            if not has_fvg:
                return None

            # FVG disrespect: if any candle after FVG creation closed below
            # fvg_bottom, the gap is filled — price disrespected it.
            mid_candle = df.iloc[-2]  # candle between c1 and c3
            if float(mid_candle["close"]) < fvg_bottom:
                return None  # disrespected

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
            fvg_top = float(fvg_first["low"])      # candle 1 low
            fvg_bottom = float(signal["high"])      # candle 3 high
            has_fvg = fvg_top > fvg_bottom
            if not has_fvg:
                return None

            # FVG disrespect: if any candle after FVG creation closed above
            # fvg_top, the gap is filled — price disrespected it.
            mid_candle = df.iloc[-2]
            if float(mid_candle["close"]) > fvg_top:
                return None  # disrespected

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

    # ------------------------------------------------------------------
    # ICT: Asian Session Range
    # ------------------------------------------------------------------
    @staticmethod
    def get_asian_range(
        df: pd.DataFrame,
        asian_start_hour: int = 0,
        asian_end_hour: int = 6,
        use_closed_candles: bool = True,
    ) -> Optional[SessionRange]:
        """Calculates the Asian session high/low from intraday data.

        Asian session: 00:00-06:00 UTC (20:00-02:00 EST).
        London/NY often sweeps the Asian range before reversing.
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < 10:
            return None

        times = pd.to_datetime(data["time"])
        hours = times.dt.hour
        today = times.iloc[-1].normalize()

        # Find today's Asian session candles
        asian_mask = (times >= today) & (hours >= asian_start_hour) & (hours < asian_end_hour)
        if not asian_mask.any():
            # Try yesterday's Asian session
            yesterday = today - pd.Timedelta(days=1)
            asian_mask = (times >= yesterday) & (times < today) & \
                         (hours >= asian_start_hour) & (hours < asian_end_hour)

        if not asian_mask.any():
            return None

        asian = data[asian_mask]
        return SessionRange(
            high=float(asian["high"].max()),
            low=float(asian["low"].min()),
            start_time=asian["time"].iloc[0],
            end_time=asian["time"].iloc[-1],
        )

    @staticmethod
    def is_asian_range_swept(
        df: pd.DataFrame,
        asian_range: SessionRange,
        direction: str,
    ) -> bool:
        """Checks if price swept the Asian session range.

        BULLISH: price dipped below Asian low then recovered (sellside sweep)
        BEARISH: price spiked above Asian high then recovered (buyside sweep)
        """
        if asian_range is None:
            return False

        recent = df.tail(5)
        if len(recent) == 0:
            return False

        if direction == "BULLISH":
            # Price went below Asian low and closed back above
            swept_below = (recent["low"] < asian_range.low).any()
            closed_above = float(recent.iloc[-1]["close"]) > asian_range.low
            return swept_below and closed_above

        if direction == "BEARISH":
            # Price went above Asian high and closed back below
            swept_above = (recent["high"] > asian_range.high).any()
            closed_below = float(recent.iloc[-1]["close"]) < asian_range.high
            return swept_above and closed_below

        return False

    # ------------------------------------------------------------------
    # ICT: Premium / Discount Zones
    # ------------------------------------------------------------------
    @staticmethod
    def get_premium_discount(
        df: pd.DataFrame,
        lookback: int = 50,
        swing_strength: int = 3,
        use_closed_candles: bool = True,
    ) -> Optional[dict]:
        """Determines premium/discount zone using swing structure (LuxAlgo-style).

        Uses trailing swing highs/lows as the range bounds, not a rolling
        window max/min. This gives structurally meaningful zones.

        Returns {"equilibrium": float, "zone": "PREMIUM"|"DISCOUNT"|"EQUILIBRIUM",
                 "range_high": float, "range_low": float}
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < lookback:
            return None

        window = data.iloc[-lookback:]
        highs = window["high"].to_numpy(float)
        lows = window["low"].to_numpy(float)

        # Find structural swing highs and lows
        swing_high = None
        swing_low = None
        for i in range(swing_strength, len(window) - swing_strength):
            if highs[i] == np.max(highs[i - swing_strength: i + swing_strength + 1]):
                if swing_high is None or highs[i] > swing_high:
                    swing_high = float(highs[i])
            if lows[i] == np.min(lows[i - swing_strength: i + swing_strength + 1]):
                if swing_low is None or lows[i] < swing_low:
                    swing_low = float(lows[i])

        if swing_high is None or swing_low is None or swing_high <= swing_low:
            return None

        # Track trailing extremes (update with current bars like LuxAlgo)
        range_high = max(swing_high, float(highs[-1]))
        range_low = min(swing_low, float(lows[-1]))
        equilibrium = (range_high + range_low) / 2.0
        current_price = float(data.iloc[-1]["close"])

        if current_price > equilibrium:
            zone = "PREMIUM"
        elif current_price < equilibrium:
            zone = "DISCOUNT"
        else:
            zone = "EQUILIBRIUM"

        return {
            "equilibrium": equilibrium,
            "zone": zone,
            "range_high": range_high,
            "range_low": range_low,
            "price": current_price,
        }

    @staticmethod
    def is_premium_discount_aligned(
        pd_zone: Optional[dict],
        direction: str,
    ) -> bool:
        """Checks if trade direction aligns with premium/discount zone.

        BUY should be in DISCOUNT, SELL should be in PREMIUM.
        """
        if pd_zone is None:
            return True  # no data, allow trade

        if direction == "BUY" and pd_zone["zone"] == "DISCOUNT":
            return True
        if direction == "SELL" and pd_zone["zone"] == "PREMIUM":
            return True
        return False

    # ------------------------------------------------------------------
    # ICT: Market Structure Shift (MSS) & Break of Structure (BOS)
    # ------------------------------------------------------------------
    @staticmethod
    def detect_structure_shift(
        df: pd.DataFrame,
        swing_strength: int = 3,
        lookback: int = 50,
        use_closed_candles: bool = True,
    ) -> Optional[StructureShift]:
        """Detects the most recent BOS or CHoCH using LuxAlgo-style trend bias.

        Tracks trend bias from swing structure. When price crosses a pivot:
        - BOS  = cross WITH current bias (trend continuation)
        - CHoCH = cross AGAINST current bias (trend reversal / MSS)

        The bias flips on every cross, just like LuxAlgo's displayStructure().
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < lookback:
            return None

        window = data.iloc[-lookback:]
        highs = window["high"].to_numpy(float)
        lows = window["low"].to_numpy(float)
        closes = window["close"].to_numpy(float)

        # Find swing pivots
        pivot_highs = []  # (index, price, crossed)
        pivot_lows = []
        for i in range(swing_strength, len(window) - swing_strength):
            if highs[i] == np.max(highs[i - swing_strength: i + swing_strength + 1]):
                pivot_highs.append([i, float(highs[i]), False])
            if lows[i] == np.min(lows[i - swing_strength: i + swing_strength + 1]):
                pivot_lows.append([i, float(lows[i]), False])

        if not pivot_highs or not pivot_lows:
            return None

        # Replay structure breaks to build trend bias (like LuxAlgo)
        bias = 0  # 0 = unknown, +1 = BULLISH, -1 = BEARISH
        last_shift = None
        ph_idx = 0  # pointer into pivot_highs
        pl_idx = 0  # pointer into pivot_lows

        for t in range(len(window)):
            # Update available pivots (only those formed before bar t)
            while ph_idx < len(pivot_highs) and pivot_highs[ph_idx][0] <= t - swing_strength:
                ph_idx += 1
            while pl_idx < len(pivot_lows) and pivot_lows[pl_idx][0] <= t - swing_strength:
                pl_idx += 1

            # Check most recent uncrossed pivot high
            for pi in range(ph_idx - 1, -1, -1):
                if not pivot_highs[pi][2]:
                    level = pivot_highs[pi][1]
                    if closes[t] > level:
                        pivot_highs[pi][2] = True
                        kind = "CHoCH" if bias == -1 else "BOS"
                        bias = 1  # flip to BULLISH
                        last_shift = StructureShift(
                            kind=kind if kind == "CHoCH" else "BOS",
                            direction="BULLISH",
                            break_level=level,
                            break_index=t)
                    break

            # Check most recent uncrossed pivot low
            for pi in range(pl_idx - 1, -1, -1):
                if not pivot_lows[pi][2]:
                    level = pivot_lows[pi][1]
                    if closes[t] < level:
                        pivot_lows[pi][2] = True
                        kind = "CHoCH" if bias == 1 else "BOS"
                        bias = -1  # flip to BEARISH
                        last_shift = StructureShift(
                            kind=kind if kind == "CHoCH" else "BOS",
                            direction="BEARISH",
                            break_level=level,
                            break_index=t)
                    break

        return last_shift

    # ------------------------------------------------------------------
    # ICT: Breaker Blocks
    # ------------------------------------------------------------------
    @staticmethod
    def detect_breaker_block(
        df: pd.DataFrame,
        lookback: int = 50,
        use_closed_candles: bool = True,
    ) -> Optional[BreakerBlock]:
        """Detects breaker blocks — failed Order Blocks that flip polarity.

        When a bullish OB is broken downward, it becomes a bearish breaker
        (new resistance). When a bearish OB is broken upward, it becomes
        a bullish breaker (new support).
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < lookback:
            return None

        window = data.iloc[-lookback:]
        highs = window["high"].to_numpy(float)
        lows = window["low"].to_numpy(float)
        opens = window["open"].to_numpy(float)
        closes = window["close"].to_numpy(float)

        # Scan for order blocks that were subsequently broken
        for i in range(len(window) - 4, 2, -1):
            # Look for bearish OB (last up candle before down move) broken upward
            if closes[i] > opens[i]:  # bullish candle (potential bearish OB)
                ob_top = highs[i]
                ob_bottom = min(lows[i], closes[i])

                # Was there a down move after it?
                down_move = any(closes[j] < ob_bottom for j in range(i + 1, min(i + 5, len(window))))
                if not down_move:
                    continue

                # Was it then broken upward (close above ob_top)?
                for j in range(i + 3, len(window)):
                    if closes[j] > ob_top:
                        # Breaker block: old bearish OB is now bullish support
                        # Check it hasn't been mitigated (retested and broken down)
                        mitigated = any(closes[k] < ob_bottom
                                        for k in range(j + 1, len(window)))
                        if not mitigated:
                            return BreakerBlock(
                                type="BULLISH",
                                top=float(ob_top),
                                bottom=float(ob_bottom),
                                break_index=j)
                        break

            # Look for bullish OB (last down candle before up move) broken downward
            if closes[i] < opens[i]:  # bearish candle (potential bullish OB)
                ob_top = max(highs[i], opens[i])
                ob_bottom = lows[i]

                # Was there an up move after it?
                up_move = any(closes[j] > ob_top for j in range(i + 1, min(i + 5, len(window))))
                if not up_move:
                    continue

                # Was it then broken downward (close below ob_bottom)?
                for j in range(i + 3, len(window)):
                    if closes[j] < ob_bottom:
                        # Breaker block: old bullish OB is now bearish resistance
                        mitigated = any(closes[k] > ob_top
                                        for k in range(j + 1, len(window)))
                        if not mitigated:
                            return BreakerBlock(
                                type="BEARISH",
                                top=float(ob_top),
                                bottom=float(ob_bottom),
                                break_index=j)
                        break

        return None

    # ------------------------------------------------------------------
    # ICT: Power of Three (PO3)
    # ------------------------------------------------------------------
    @staticmethod
    def detect_po3(
        df: pd.DataFrame,
        consolidation_bars: int = 20,
        atr_threshold: float = 0.4,
        sweep_bars: int = 5,
        use_closed_candles: bool = True,
    ) -> Optional[dict]:
        """Detects Power of Three: Accumulation -> Manipulation -> Distribution.

        1. Accumulation: low-volatility consolidation (range < threshold x ATR)
        2. Manipulation: false breakout / liquidity sweep from the range
        3. Distribution: reversal signal (strong rejection candle)

        Returns {"phase": "DISTRIBUTION", "direction": "BULLISH"|"BEARISH",
                 "range_high": float, "range_low": float, "sweep_price": float}
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < consolidation_bars + sweep_bars + 5:
            return None

        # 1. Accumulation: find consolidation range
        consol = data.iloc[-(consolidation_bars + sweep_bars + 5):-(sweep_bars + 5)]
        atr = float((data["high"] - data["low"]).iloc[-60:].mean()) if len(data) >= 60 else \
              float((data["high"] - data["low"]).mean())

        range_high = float(consol["high"].max())
        range_low = float(consol["low"].min())
        range_size = range_high - range_low

        if range_size > atr * consolidation_bars * atr_threshold:
            return None

        # 2. Manipulation: check if price swept above or below the range
        recent = data.iloc[-(sweep_bars + 5):]
        highs = recent["high"].to_numpy(float)
        lows = recent["low"].to_numpy(float)
        closes = recent["close"].to_numpy(float)

        # Bullish PO3: price dipped below range_low then reversed up
        swept_below = False
        sweep_price = 0.0
        for i in range(len(recent)):
            if lows[i] < range_low:
                swept_below = True
                sweep_price = float(lows[i])

        if swept_below and closes[-1] > range_low:
            last_body = closes[-1] - float(recent.iloc[-1]["open"])
            if last_body > 0:
                return {
                    "phase": "DISTRIBUTION",
                    "direction": "BULLISH",
                    "range_high": range_high,
                    "range_low": range_low,
                    "sweep_price": sweep_price,
                }

        # Bearish PO3: price spiked above range_high then reversed down
        swept_above = False
        for i in range(len(recent)):
            if highs[i] > range_high:
                swept_above = True
                sweep_price = float(highs[i])

        if swept_above and closes[-1] < range_high:
            last_body = float(recent.iloc[-1]["open"]) - closes[-1]
            if last_body > 0:
                return {
                    "phase": "DISTRIBUTION",
                    "direction": "BEARISH",
                    "range_high": range_high,
                    "range_low": range_low,
                    "sweep_price": sweep_price,
                }

        return None

    # ------------------------------------------------------------------
    # ICT: Inversion Fair Value Gap (IFVG)
    # ------------------------------------------------------------------
    @staticmethod
    def detect_ifvg(
        df: pd.DataFrame,
        lookback: int = 50,
        use_closed_candles: bool = True,
    ) -> Optional[dict]:
        """Detects Inversion Fair Value Gaps -- filled FVGs that flip polarity.

        When a bullish FVG gets filled (price trades through it), the zone
        becomes bearish resistance (IFVG). Vice versa for bearish FVGs.

        Returns {"type": "BULLISH"|"BEARISH", "top": float, "bottom": float}
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < lookback:
            return None

        window = data.iloc[-lookback:]
        highs = window["high"].to_numpy(float)
        lows = window["low"].to_numpy(float)
        closes = window["close"].to_numpy(float)

        for i in range(len(window) - 3, 2, -1):
            # Bullish FVG: candle[i-2] high < candle[i] low
            c1_high = highs[i - 2]
            c3_low = lows[i]
            if c3_low > c1_high:
                fvg_top = c3_low
                fvg_bottom = c1_high

                # Was it filled? (price closed below fvg_bottom)
                filled = False
                fill_idx = -1
                for j in range(i + 1, len(window)):
                    if closes[j] < fvg_bottom:
                        filled = True
                        fill_idx = j
                        break

                if filled:
                    # Zone still valid? (not traded through again)
                    still_valid = True
                    for k in range(fill_idx + 1, len(window)):
                        if closes[k] > fvg_top:
                            still_valid = False
                            break

                    if still_valid:
                        current = closes[-1]
                        zone_range = fvg_top - fvg_bottom
                        if fvg_bottom - zone_range <= current <= fvg_top:
                            return {
                                "type": "BEARISH",
                                "top": float(fvg_top),
                                "bottom": float(fvg_bottom),
                            }

            # Bearish FVG: candle[i-2] low > candle[i] high
            c1_low = lows[i - 2]
            c3_high = highs[i]
            if c1_low > c3_high:
                fvg_top = c1_low
                fvg_bottom = c3_high

                filled = False
                fill_idx = -1
                for j in range(i + 1, len(window)):
                    if closes[j] > fvg_top:
                        filled = True
                        fill_idx = j
                        break

                if filled:
                    still_valid = True
                    for k in range(fill_idx + 1, len(window)):
                        if closes[k] < fvg_bottom:
                            still_valid = False
                            break

                    if still_valid:
                        current = closes[-1]
                        zone_range = fvg_top - fvg_bottom
                        if fvg_bottom <= current <= fvg_top + zone_range:
                            return {
                                "type": "BULLISH",
                                "top": float(fvg_top),
                                "bottom": float(fvg_bottom),
                            }

        return None

    # ------------------------------------------------------------------
    # ICT: Equal Highs / Equal Lows (Liquidity Pools)
    # ------------------------------------------------------------------
    @staticmethod
    def detect_equal_levels(
        df: pd.DataFrame,
        swing_strength: int = 3,
        atr_threshold: float = 0.1,
        use_closed_candles: bool = True,
    ) -> tuple[list, list]:
        """Detects equal highs and equal lows (LuxAlgo-style).

        Equal highs/lows are swing points at approximately the same price
        level. They indicate liquidity pools where stop orders accumulate.

        atr_threshold: how close two swing points must be (as fraction of ATR)
        to be considered "equal". LuxAlgo default is 0.1.

        Returns ([EqualLevel, ...], [EqualLevel, ...]) for EQH and EQL.
        """
        data = SMCStrategy.closed_candles(df, use_closed_candles)
        if len(data) < 2 * swing_strength + 1:
            return [], []

        highs = data["high"].to_numpy(float)
        lows = data["low"].to_numpy(float)
        atr = float((data["high"] - data["low"]).mean())
        tolerance = atr * atr_threshold

        # Find swing highs and lows
        sh_prices = []
        sl_prices = []
        for i in range(swing_strength, len(data) - swing_strength):
            if highs[i] == np.max(highs[i - swing_strength: i + swing_strength + 1]):
                sh_prices.append(float(highs[i]))
            if lows[i] == np.min(lows[i - swing_strength: i + swing_strength + 1]):
                sl_prices.append(float(lows[i]))

        # Group equal highs
        eqh_list = []
        used = set()
        for i, p1 in enumerate(sh_prices):
            if i in used:
                continue
            group = [p1]
            for j in range(i + 1, len(sh_prices)):
                if j not in used and abs(sh_prices[j] - p1) < tolerance:
                    group.append(sh_prices[j])
                    used.add(j)
            if len(group) >= 2:
                avg_level = sum(group) / len(group)
                eqh_list.append(EqualLevel(
                    type="EQH", level=round(avg_level, 5), count=len(group)))

        # Group equal lows
        eql_list = []
        used = set()
        for i, p1 in enumerate(sl_prices):
            if i in used:
                continue
            group = [p1]
            for j in range(i + 1, len(sl_prices)):
                if j not in used and abs(sl_prices[j] - p1) < tolerance:
                    group.append(sl_prices[j])
                    used.add(j)
            if len(group) >= 2:
                avg_level = sum(group) / len(group)
                eql_list.append(EqualLevel(
                    type="EQL", level=round(avg_level, 5), count=len(group)))

        return eqh_list, eql_list

    @staticmethod
    def find_nearest_equal_level(
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        swing_strength: int = 3,
        atr_threshold: float = 0.1,
        use_closed_candles: bool = True,
    ) -> Optional[float]:
        """Finds nearest equal high/low as liquidity target.

        BUY  -> nearest EQH above entry (buyside liquidity pool)
        SELL -> nearest EQL below entry (sellside liquidity pool)
        """
        eqh_list, eql_list = SMCStrategy.detect_equal_levels(
            df, swing_strength, atr_threshold, use_closed_candles)

        if direction == "BUY":
            above = [eq.level for eq in eqh_list if eq.level > entry_price]
            return min(above) if above else None

        below = [eq.level for eq in eql_list if eq.level < entry_price]
        return max(below) if below else None
