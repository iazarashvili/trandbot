"""Test sweep filter on BTCUSD and GBPUSD — compare with/without."""
import sys
import importlib
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from strategy import SMCStrategy
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

END = datetime.now()
START = END - timedelta(days=180)

symbols = [
    ("BTCUSD", "config_btcusd"),
    ("GBPUSD", "config_gbpusd"),
]

all_frames = {}
for sym, cfg_name in symbols:
    info = mt5.symbol_info(sym)
    frames = {}
    for key, tf in [("m5", mt5.TIMEFRAME_M5), ("h1", mt5.TIMEFRAME_H1),
                     ("m15", mt5.TIMEFRAME_M15)]:
        r = mt5.copy_rates_range(sym, tf, START, END)
        if r is None or len(r) == 0:
            r = mt5.copy_rates_from_pos(sym, tf, 0, 100000)
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[(df["time"] >= pd.Timestamp(START)) & (df["time"] <= pd.Timestamp(END))]
        frames[key] = df
    all_frames[sym] = {
        "frames": frames,
        "contract": info.trade_contract_size,
        "point": info.point,
        "cfg_name": cfg_name,
    }

mt5.shutdown()

BALANCE = 3000.0

print(f"{'Symbol':<10} {'Config':<30} {'Lot':>5} {'Trades':>7} {'Wins':>5} "
      f"{'WR%':>6} {'PF':>6} {'Net':>12} {'MaxDD':>7}")
print("-" * 95)

for sym, cfg_name in symbols:
    d = all_frames[sym]
    cfg = importlib.import_module(cfg_name)
    sys.modules["config"] = cfg

    import reports.engine as eng

    rrr = getattr(cfg, "RRR", 3.0)
    stop_mode = getattr(cfg, "STOP_MODE", "window")
    lot = 0.03 if sym == "BTCUSD" else 0.05

    # Run with and without sweep, with different RRR
    configs = []
    if sym == "BTCUSD":
        configs = [
            (f"No sweep, RRR {rrr}", False, rrr, 0.0),
            ("SWEEP, RRR 2.0", True, 2.0, 0.0),
            ("SWEEP, RRR 2.5", True, 2.5, 0.0),
            ("SWEEP, RRR 3.0", True, 3.0, 0.0),
            ("SWEEP, RRR 3.5", True, 3.5, 0.0),
            ("SWEEP, RRR 3.0, BE 2R", True, 3.0, 2.0),
            ("SWEEP, RRR 3.5, BE 2R", True, 3.5, 2.0),
        ]
    else:
        configs = [
            (f"No sweep, RRR {rrr}", False, rrr, 0.0),
            ("SWEEP, RRR 2.0", True, 2.0, 0.0),
            ("SWEEP, RRR 2.5", True, 2.5, 0.0),
            ("SWEEP, RRR 3.0", True, 3.0, 0.0),
            ("SWEEP, RRR 3.5", True, 3.5, 0.0),
            ("SWEEP, RRR 3.0, BE 2R", True, 3.0, 2.0),
            ("SWEEP, RRR 3.5, BE 2R", True, 3.5, 2.0),
        ]

    for label, use_sweep, test_rrr, be_r in configs:
        importlib.reload(eng)
        eng.START_BALANCE = BALANCE
        eng.VOLUME = lot
        eng._HTF_CACHE.clear()

        # Custom run with sweep filter
        m5 = d["frames"]["m5"]
        h1 = d["frames"]["h1"]
        m15 = d["frames"]["m15"]
        contract = d["contract"]
        point = d["point"]

        from config import (
            HTF_CANDLES_LOOKBACK, LTF_CANDLES_LOOKBACK,
            TREND_EMA_PERIOD, MAX_LTF_WAIT_CANDLES,
            SKIP_WEEKENDS, BLOCKED_DAYS, NIGHT_START_HOUR, NIGHT_END_HOUR,
        )

        h1_t = h1["time"].values
        m5_t = m5["time"].values
        h_idx = np.searchsorted(h1_t, m5_t - np.timedelta64(60, "m"), side="right") - 1

        o = m5["open"].to_numpy(float)
        hi = m5["high"].to_numpy(float)
        lo = m5["low"].to_numpy(float)
        cl = m5["close"].to_numpy(float)
        spread = m5["spread"].to_numpy(float) * point
        times = m5["time"]
        _weekdays = pd.to_datetime(m5["time"]).dt.weekday.to_numpy()
        _hours = pd.to_datetime(m5["time"]).dt.hour.to_numpy()

        start_bar = LTF_CANDLES_LOOKBACK
        while start_bar < len(m5) and h_idx[start_bar] < HTF_CANDLES_LOOKBACK:
            start_bar += 1

        htf_cache = {}
        poi = None
        poi_h = -1
        watch_key = None
        watch_start = None
        abandoned = False
        position = None
        trades = []
        balance = BALANCE

        for t in range(start_bar, len(m5) - 1):
            if position is not None:
                p = position
                vol = lot

                # Breakeven check
                if be_r > 0 and not p.get("be_moved"):
                    trigger_dist = p["risk_px"] * be_r
                    if p["dir"] == "BUY":
                        if hi[t] >= p["entry"] + trigger_dist:
                            p["sl"] = p["entry"]
                            p["be_moved"] = True
                    else:
                        if lo[t] + spread[t] <= p["entry"] - trigger_dist:
                            p["sl"] = p["entry"]
                            p["be_moved"] = True

                if p["dir"] == "BUY":
                    hit_sl = lo[t] <= p["sl"]
                    hit_tp = hi[t] >= p["tp"]
                else:
                    hit_sl = hi[t] + spread[t] >= p["sl"]
                    hit_tp = lo[t] + spread[t] <= p["tp"]

                if hit_sl or hit_tp:
                    exit_px = p["sl"] if hit_sl else p["tp"]
                    pnl = ((exit_px - p["entry"]) if p["dir"] == "BUY"
                           else (p["entry"] - exit_px)) * vol * contract
                    balance += pnl
                    risk_usd = p["risk_px"] * vol * contract
                    p.update(
                        pnl=round(pnl, 2),
                        balance=round(balance, 2),
                    )
                    trades.append(p)
                    position = None
                    watch_key = None
                    abandoned = False
                    if balance <= 0:
                        break
                continue

            if SKIP_WEEKENDS and _weekdays[t] >= 5:
                continue

            if BLOCKED_DAYS and _weekdays[t] in BLOCKED_DAYS:
                continue

            h = _hours[t]
            if NIGHT_START_HOUR > NIGHT_END_HOUR:
                if h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR:
                    continue
            elif NIGHT_START_HOUR != NIGHT_END_HOUR and NIGHT_START_HOUR <= h < NIGHT_END_HOUR:
                continue

            if h_idx[t] != poi_h:
                poi_h = h_idx[t]
                if poi_h in htf_cache:
                    trend, poi = htf_cache[poi_h]
                else:
                    wh = h1.iloc[max(0, poi_h - HTF_CANDLES_LOOKBACK + 1): poi_h + 1]
                    trend = SMCStrategy.get_htf_trend(wh, TREND_EMA_PERIOD, use_closed_candles=False)
                    poi = SMCStrategy.detect_htf_poi(
                        wh, use_trend_filter=True,
                        ema_period=TREND_EMA_PERIOD, use_closed_candles=False)
                    htf_cache[poi_h] = (trend, poi)
                if poi is None:
                    watch_key = None
                    abandoned = False

            if poi is None:
                continue

            mid = cl[t] + spread[t] / 2.0
            if not (poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom)):
                watch_key = None
                abandoned = False
                continue

            ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
            if not SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
                watch_key = None
                abandoned = False
                continue

            zone_key = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
            if zone_key != watch_key:
                watch_key = zone_key
                watch_start = t
                abandoned = False
            if abandoned:
                continue
            if (t - watch_start + 1) > MAX_LTF_WAIT_CANDLES:
                abandoned = True
                continue

            if SMCStrategy.is_consolidating(ltf, use_closed_candles=False):
                continue

            if use_sweep:
                sweep = SMCStrategy.detect_liquidity_sweep(
                    ltf, swing_strength=3, use_closed_candles=False)
                if sweep is None:
                    continue
                if poi.type == "BULLISH" and sweep.direction != "BULLISH":
                    continue
                if poi.type == "BEARISH" and sweep.direction != "BEARISH":
                    continue

            setup = SMCStrategy.check_ltf_confirmation(
                ltf, poi, test_rrr, use_closed_candles=False,
                stop_mode=stop_mode, buffer_atr=0.5)
            if setup is None:
                continue

            entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
            sl = setup["sl"]
            risk = abs(entry - sl)
            if risk <= 0:
                continue
            tp = entry + risk * test_rrr if setup["direction"] == "BUY" else entry - risk * test_rrr

            position = {
                "dir": setup["direction"],
                "entry_time": str(times.iloc[t + 1]),
                "entry": round(entry, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "risk_px": round(risk, 2),
                "entry_bar": t + 1,
            }

        wins = len([x for x in trades if x["pnl"] > 0])
        losses = len(trades) - wins
        net = balance - BALANCE
        gw = sum(x["pnl"] for x in trades if x["pnl"] > 0)
        gl = -sum(x["pnl"] for x in trades if x["pnl"] <= 0)
        pf = round(gw / gl, 2) if gl > 0 else "-"
        wr = round(wins / len(trades) * 100, 1) if trades else 0
        peak = BALANCE
        max_dd = 0
        for tr in trades:
            peak = max(peak, tr["balance"])
            dd = peak - tr["balance"]
            max_dd = max(max_dd, dd)
        max_dd_pct = round(max_dd / peak * 100, 1) if peak > 0 else 0
        sign = "+" if net >= 0 else ""

        print(f"{sym:<10} {label:<30} {lot:>5} {len(trades):>7} {wins:>5} "
              f"{wr:>5.1f}% {str(pf):>6} {sign}${net:>10,.2f} {max_dd_pct:>6.1f}%")

    print()
