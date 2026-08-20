"""Unit tests for the SMC strategy logic.

These cover the pure functions only — no MetaTrader 5 terminal required.

    python -m unittest discover -s tests -t .
"""

import unittest

import pandas as pd

from strategy import SMCStrategy, ZonePOI


def make_df(candles) -> pd.DataFrame:
    """Builds an OHLC frame from a list of (open, high, low, close) tuples."""
    base = pd.Timestamp("2026-01-01 00:00:00")
    return pd.DataFrame(
        [
            {
                "time": base + pd.Timedelta(minutes=i),
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
            }
            for i, (o, h, lo, c) in enumerate(candles)
        ]
    )


# A 15m sequence whose last candle completes a bullish FVG overlapping a
# bearish order block: OB at index 1, FVG across indices 2..4.
BULLISH_POI_CANDLES = [
    (100, 101, 99, 100),    # 0 filler
    (105, 106, 100, 101),   # 1 order block (down candle)
    (101, 103, 100, 102),   # 2 FVG c1 -> high 103
    (102, 105, 102, 104),   # 3 displacement
    (105, 108, 104, 107),   # 4 FVG c3 -> low 104 > 103
]

# A 1m sequence: seven quiet candles, then a displacement leaving a gap and a
# closing candle that breaks the recent high.
BULLISH_LTF_CANDLES = [(98, 100, 95, 99)] * 7 + [
    (99, 101, 97, 100),     # 7 FVG c1 -> high 101
    (100, 106, 100, 105),   # 8 displacement
    (105, 108, 102, 107),   # 9 signal: close 107 > 100, low 102 > 101
]


class ClosedCandlesTest(unittest.TestCase):
    def test_drops_the_forming_candle(self):
        df = make_df(BULLISH_POI_CANDLES)
        trimmed = SMCStrategy.closed_candles(df)
        self.assertEqual(len(trimmed), len(df) - 1)
        self.assertEqual(trimmed.iloc[-1]["close"], 104)

    def test_can_be_disabled(self):
        df = make_df(BULLISH_POI_CANDLES)
        self.assertEqual(len(SMCStrategy.closed_candles(df, False)), len(df))

    def test_never_empties_a_single_candle_frame(self):
        df = make_df([(1, 2, 0, 1)])
        self.assertEqual(len(SMCStrategy.closed_candles(df)), 1)


class TrendTest(unittest.TestCase):
    def test_bullish_when_close_is_above_the_ema(self):
        df = make_df([(i, i + 1, i - 1, i) for i in range(100, 130)])
        self.assertEqual(SMCStrategy.get_htf_trend(df, period=10), "BULLISH")

    def test_bearish_when_close_is_below_the_ema(self):
        df = make_df([(i, i + 1, i - 1, i) for i in range(130, 100, -1)])
        self.assertEqual(SMCStrategy.get_htf_trend(df, period=10), "BEARISH")

    def test_neutral_without_enough_history(self):
        df = make_df(BULLISH_POI_CANDLES)
        self.assertEqual(SMCStrategy.get_htf_trend(df, period=100), "NEUTRAL")


class ZoneInPlayTest(unittest.TestCase):
    poi = ZonePOI(type="BULLISH", top=106, bottom=100)

    def test_live_price_inside_the_zone(self):
        self.assertTrue(SMCStrategy.is_zone_in_play(self.poi, current_price=103))

    def test_candle_wick_into_the_zone_counts(self):
        df = make_df([(110, 112, 105, 111)])
        self.assertTrue(
            SMCStrategy.is_zone_in_play(self.poi, df, current_price=111)
        )

    def test_price_and_candle_both_clear_of_the_zone(self):
        df = make_df([(120, 122, 118, 121)])
        self.assertFalse(
            SMCStrategy.is_zone_in_play(self.poi, df, current_price=121)
        )


class DetectHtfPoiTest(unittest.TestCase):
    def test_finds_the_overlapping_fvg_and_order_block(self):
        df = make_df(BULLISH_POI_CANDLES)
        poi = SMCStrategy.detect_htf_poi(
            df, use_trend_filter=False, use_closed_candles=False
        )

        self.assertIsNotNone(poi)
        self.assertEqual(poi.type, "BULLISH")
        self.assertEqual(poi.top, 106)     # max(fvg_top 104, ob_top 106)
        self.assertEqual(poi.bottom, 100)  # min(fvg_bottom 103, ob_bottom 100)
        self.assertEqual(poi.index, 4)

    def test_detects_a_zone_completed_on_the_last_closed_candle(self):
        # Regression: the scan used to stop one candle short of the end, so a
        # freshly completed zone was invisible for a whole HTF bar.
        df = make_df(BULLISH_POI_CANDLES + [(107, 109, 106, 108)])
        poi = SMCStrategy.detect_htf_poi(
            df, use_trend_filter=False, use_closed_candles=True
        )

        self.assertIsNotNone(poi)
        self.assertEqual(poi.index, 4)

    def test_ignores_a_fully_mitigated_gap(self):
        # Price trades back down to 102, closing the 103-104 gap entirely.
        df = make_df(BULLISH_POI_CANDLES + [(107, 108, 102, 103)])
        poi = SMCStrategy.detect_htf_poi(
            df, use_trend_filter=False, use_closed_candles=False
        )
        self.assertIsNone(poi)

    def test_keeps_a_partially_filled_gap(self):
        # A tap into the gap without closing it is the setup, not a rejection.
        df = make_df(BULLISH_POI_CANDLES + [(107, 108, 103.5, 105)])
        poi = SMCStrategy.detect_htf_poi(
            df, use_trend_filter=False, use_closed_candles=False
        )
        self.assertIsNotNone(poi)
        self.assertEqual(poi.index, 4)

    def test_returns_none_without_enough_history(self):
        df = make_df(BULLISH_POI_CANDLES[:3])
        self.assertIsNone(
            SMCStrategy.detect_htf_poi(df, use_trend_filter=False)
        )


class TrendFilterTest(unittest.TestCase):
    """A bearish zone inside a bullish market must be skipped."""

    @staticmethod
    def _uptrend_with_a_bearish_zone() -> pd.DataFrame:
        candles = [(100 + 2 * n, 103 + 2 * n, 99 + 2 * n, 102 + 2 * n) for n in range(39)]
        candles += [
            (175, 181, 173, 180),   # 39 order block (up candle)
            (178, 180, 176, 177),   # 40 FVG c1 -> low 176
            (177, 177, 170, 171),   # 41 displacement
            (171, 173, 168, 169),   # 42 FVG c3 -> high 173 < 176
        ]
        return make_df(candles)

    def test_zone_exists_when_the_filter_is_off(self):
        df = self._uptrend_with_a_bearish_zone()
        poi = SMCStrategy.detect_htf_poi(
            df, use_trend_filter=False, use_closed_candles=False
        )
        self.assertIsNotNone(poi)
        self.assertEqual(poi.type, "BEARISH")

    def test_zone_is_skipped_against_the_trend(self):
        df = self._uptrend_with_a_bearish_zone()
        self.assertEqual(
            SMCStrategy.get_htf_trend(df, period=20, use_closed_candles=False),
            "BULLISH",
        )
        poi = SMCStrategy.detect_htf_poi(
            df, use_trend_filter=True, ema_period=20, use_closed_candles=False
        )
        self.assertIsNone(poi)


class LtfConfirmationTest(unittest.TestCase):
    poi = ZonePOI(type="BULLISH", top=105, bottom=95)

    def test_confirmed_buy_uses_the_configured_rrr(self):
        df = make_df(BULLISH_LTF_CANDLES)
        setup = SMCStrategy.check_ltf_confirmation(
            df, self.poi, rrr=3.0, use_closed_candles=False
        )

        self.assertIsNotNone(setup)
        self.assertEqual(setup["direction"], "BUY")
        self.assertEqual(setup["entry"], 107)
        self.assertEqual(setup["sl"], 95)
        self.assertEqual(setup["tp"], 107 + (107 - 95) * 3)

    def test_no_setup_without_displacement_fvg(self):
        candles = list(BULLISH_LTF_CANDLES)
        # Signal candle low touches fvg_first high — no gap, no displacement
        candles[-1] = (100, 107, 97, 107)
        setup = SMCStrategy.check_ltf_confirmation(
            make_df(candles), self.poi, rrr=3.0, use_closed_candles=False
        )
        self.assertIsNone(setup)

    def test_no_setup_when_price_is_outside_the_zone(self):
        far_poi = ZonePOI(type="BULLISH", top=60, bottom=50)
        setup = SMCStrategy.check_ltf_confirmation(
            make_df(BULLISH_LTF_CANDLES), far_poi, rrr=3.0, use_closed_candles=False
        )
        self.assertIsNone(setup)

    def test_the_forming_candle_is_ignored(self):
        # Regression: reading the live candle made signals appear and vanish
        # within a single bar.  The same frame must resolve to the setup that
        # the last *closed* candle produced.
        df = make_df(BULLISH_LTF_CANDLES + [(107, 107.5, 106, 106.5)])

        closed = SMCStrategy.check_ltf_confirmation(
            df, self.poi, rrr=3.0, use_closed_candles=True
        )
        forming = SMCStrategy.check_ltf_confirmation(
            df, self.poi, rrr=3.0, use_closed_candles=False
        )

        self.assertIsNotNone(closed)
        self.assertEqual(closed["entry"], 107)
        self.assertIsNone(forming)

    def test_returns_none_without_enough_history(self):
        setup = SMCStrategy.check_ltf_confirmation(
            make_df(BULLISH_LTF_CANDLES[:5]), self.poi, rrr=3.0
        )
        self.assertIsNone(setup)


class StopPlacementTest(unittest.TestCase):
    """The three stop rules must produce genuinely different levels."""

    poi = ZonePOI(type="BULLISH", top=105, bottom=95)

    def _sl(self, mode):
        setup = SMCStrategy.check_ltf_confirmation(
            make_df(BULLISH_LTF_CANDLES), self.poi, rrr=3.0,
            use_closed_candles=False, stop_mode=mode, buffer_atr=0.5)
        self.assertIsNotNone(setup)
        return setup["sl"]

    def test_window_is_the_lookback_extreme(self):
        self.assertEqual(self._sl("window"), 95)  # min low of all 10 candles

    def test_swing_starts_at_the_broken_pivot(self):
        # The MSS broke the high of the 7 structure candles; the stop is the
        # lowest low from that pivot onward, minus a buffer.
        self.assertLess(self._sl("swing"), 97)
        self.assertGreater(self._sl("swing"), 90)

    def test_zone_is_never_tighter_than_the_poi_floor(self):
        self.assertLessEqual(self._sl("zone"), self.poi.bottom)

    def test_buffer_widens_the_stop(self):
        setup_a = SMCStrategy.check_ltf_confirmation(
            make_df(BULLISH_LTF_CANDLES), self.poi, 3.0,
            use_closed_candles=False, stop_mode="swing", buffer_atr=0.0)
        setup_b = SMCStrategy.check_ltf_confirmation(
            make_df(BULLISH_LTF_CANDLES), self.poi, 3.0,
            use_closed_candles=False, stop_mode="swing", buffer_atr=2.0)
        self.assertLess(setup_b["sl"], setup_a["sl"])

    def test_target_still_honours_the_ratio(self):
        for mode in ("window", "swing", "zone"):
            setup = SMCStrategy.check_ltf_confirmation(
                make_df(BULLISH_LTF_CANDLES), self.poi, 2.5,
                use_closed_candles=False, stop_mode=mode)
            risk = setup["entry"] - setup["sl"]
            self.assertAlmostEqual(setup["tp"], setup["entry"] + risk * 2.5)


class InvertTest(unittest.TestCase):
    def test_buy_becomes_a_mirrored_sell(self):
        buy = {"direction": "BUY", "entry": 100.0, "sl": 90.0, "tp": 130.0}
        out = SMCStrategy.invert(buy, rrr=3.0)

        self.assertEqual(out["direction"], "SELL")
        self.assertEqual(out["entry"], 100.0)
        self.assertEqual(out["sl"], 110.0)   # 1R above instead of below
        self.assertEqual(out["tp"], 70.0)    # 3R below instead of above
        self.assertEqual(out["inverted_from"], "BUY")

    def test_sell_becomes_a_mirrored_buy(self):
        sell = {"direction": "SELL", "entry": 100.0, "sl": 110.0, "tp": 70.0}
        out = SMCStrategy.invert(sell, rrr=3.0)

        self.assertEqual(out["direction"], "BUY")
        self.assertEqual(out["sl"], 90.0)
        self.assertEqual(out["tp"], 130.0)
        self.assertEqual(out["inverted_from"], "SELL")

    def test_risk_is_preserved_so_the_modes_compare(self):
        buy = {"direction": "BUY", "entry": 64000.0, "sl": 63900.0, "tp": 64300.0}
        out = SMCStrategy.invert(buy, rrr=3.0)
        self.assertAlmostEqual(
            abs(buy["entry"] - buy["sl"]), abs(out["entry"] - out["sl"])
        )

    def test_inverting_twice_is_the_original(self):
        buy = {"direction": "BUY", "entry": 100.0, "sl": 90.0, "tp": 130.0}
        back = SMCStrategy.invert(SMCStrategy.invert(buy, 3.0), 3.0)
        self.assertEqual(back["direction"], buy["direction"])
        self.assertEqual(back["sl"], buy["sl"])
        self.assertEqual(back["tp"], buy["tp"])


if __name__ == "__main__":
    unittest.main()
