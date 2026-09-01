"""GBPUSD sweep test with Mon/Tue unblocked."""
import sys
import importlib
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "config_gbpusd" in sys.modules:
    del sys.modules["config_gbpusd"]
cfg = importlib.import_module("config_gbpusd")
sys.modules["config"] = cfg

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from strategy import SMCStrategy
from config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    HTF_CANDLES_LOOKBACK, LTF_CANDLES_LOOKBACK,
    TREND_EMA_PERIOD, MAX_LTF_WAIT_CANDLES,
    SKIP_WEEKENDS, NIGHT_START_HOUR, NIGHT_END_HOUR, STOP_MODE,
)

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
info = mt5.symbol_info("GBPUSD")
END = datetime.now()
START = END - timedelta(days=180)
frames = {}
for key, tf in [("m5", mt5.TIMEFRAME_M5), ("h1", mt5.TIMEFRAME_H1)]:
    r = mt5.copy_rates_range("GBPUSD", tf, START, END)
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[(df["time"] >= pd.Timestamp(START)) & (df["time"] <= pd.Timestamp(END))]
    frames[key] = df
mt5.shutdown()

m5 = frames["m5"]
h1 = frames["h1"]
point = info.point
contract = info.trade_contract_size
BALANCE = 3000.0
LOT = 0.05


def run_test(use_sweep, rrr, be_r=0.0, blocked=None):
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
    bl = blocked or []

    sb = LTF_CANDLES_LOOKBACK
    while sb < len(m5) and h_idx[sb] < HTF_CANDLES_LOOKBACK:
        sb += 1

    htf_cache = {}
    poi = None
    poi_h = -1
    wk = ws = None
    ab = False
    pos = None
    trades = []
    balance = BALANCE

    for t in range(sb, len(m5) - 1):
        if pos is not None:
            p = pos
            if be_r > 0 and not p.get("be"):
                td = p["risk_px"] * be_r
                if p["dir"] == "BUY" and hi[t] >= p["entry"] + td:
                    p["sl"] = p["entry"]
                    p["be"] = True
                elif p["dir"] == "SELL" and lo[t] + spread[t] <= p["entry"] - td:
                    p["sl"] = p["entry"]
                    p["be"] = True
            if p["dir"] == "BUY":
                hs = lo[t] <= p["sl"]
                ht = hi[t] >= p["tp"]
            else:
                hs = hi[t] + spread[t] >= p["sl"]
                ht = lo[t] + spread[t] <= p["tp"]
            if hs or ht:
                ep = p["sl"] if hs else p["tp"]
                pnl = ((ep - p["entry"]) if p["dir"] == "BUY"
                       else (p["entry"] - ep)) * LOT * contract
                balance += pnl
                trades.append({
                    "pnl": round(pnl, 2), "balance": round(balance, 2),
                    "entry_time": p["entry_time"], "dir": p["dir"],
                    "exit_reason": "SL" if hs else "TP",
                })
                pos = None
                wk = None
                ab = False
                if balance <= 0:
                    break
            continue

        if SKIP_WEEKENDS and _weekdays[t] >= 5:
            continue
        if bl and _weekdays[t] in bl:
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
                wk = None
                ab = False

        if poi is None:
            continue
        mid = cl[t] + spread[t] / 2.0
        if not (poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom)):
            wk = None
            ab = False
            continue
        ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
        if not SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
            wk = None
            ab = False
            continue
        zk = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
        if zk != wk:
            wk = zk
            ws = t
            ab = False
        if ab:
            continue
        if (t - ws + 1) > MAX_LTF_WAIT_CANDLES:
            ab = True
            continue
        if SMCStrategy.is_consolidating(ltf, use_closed_candles=False):
            continue

        if use_sweep:
            sw = SMCStrategy.detect_liquidity_sweep(
                ltf, swing_strength=3, use_closed_candles=False)
            if sw is None:
                continue
            if poi.type == "BULLISH" and sw.direction != "BULLISH":
                continue
            if poi.type == "BEARISH" and sw.direction != "BEARISH":
                continue

        setup = SMCStrategy.check_ltf_confirmation(
            ltf, poi, rrr, use_closed_candles=False,
            stop_mode=STOP_MODE, buffer_atr=0.5)
        if setup is None:
            continue

        entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
        sl = setup["sl"]
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        tp = entry + risk * rrr if setup["direction"] == "BUY" else entry - risk * rrr
        pos = {
            "dir": setup["direction"], "entry_time": str(times.iloc[t + 1]),
            "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "risk_px": round(risk, 2), "entry_bar": t + 1,
        }

    wins = len([x for x in trades if x["pnl"] > 0])
    net = balance - BALANCE
    gw = sum(x["pnl"] for x in trades if x["pnl"] > 0)
    gl = -sum(x["pnl"] for x in trades if x["pnl"] <= 0)
    pf = round(gw / gl, 2) if gl > 0 else "-"
    wr = round(wins / len(trades) * 100, 1) if trades else 0
    peak = BALANCE
    mdd = 0
    for tr in trades:
        peak = max(peak, tr["balance"])
        dd = peak - tr["balance"]
        mdd = max(mdd, dd)
    return {
        "t": len(trades), "w": wins, "l": len(trades) - wins, "wr": wr,
        "pf": pf, "net": round(net, 2),
        "mdd": round(mdd / peak * 100, 1) if peak else 0,
        "trades": trades,
    }


print(f"{'Config':<40} {'Trades':>7} {'Wins':>5} {'WR%':>6} "
      f"{'PF':>6} {'Net':>12} {'MaxDD':>7}")
print("-" * 90)

configs = [
    ("Old: no sweep, Mon/Tue blocked",    False, 3.0, 0.0, [0, 1]),
    ("No sweep, no blocked days",          False, 3.0, 0.0, []),
    ("Sweep, Mon/Tue blocked",             True,  3.0, 0.0, [0, 1]),
    ("Sweep, no blocked days",             True,  3.0, 0.0, []),
    ("Sweep, no blocked, RRR 2.0",         True,  2.0, 0.0, []),
    ("Sweep, no blocked, RRR 2.5",         True,  2.5, 0.0, []),
    ("Sweep, no blocked, RRR 3.5",         True,  3.5, 0.0, []),
    ("Sweep, no blocked, RRR 3.0, BE 2R",  True,  3.0, 2.0, []),
]

for label, sw, rrr, be, bl in configs:
    r = run_test(sw, rrr, be, bl)
    sign = "+" if r["net"] >= 0 else ""
    print(f"{label:<40} {r['t']:>7} {r['w']:>5} {r['wr']:>5.1f}% "
          f"{str(r['pf']):>6} {sign}${r['net']:>10,.2f} {r['mdd']:>6.1f}%")

# Show trades for best config
print()
print("=== Sweep, no blocked, RRR 3.0 --- TRADES ===")
r = run_test(True, 3.0, 0.0, [])
for t in r["trades"]:
    won = "WIN" if t["pnl"] > 0 else "LOSS"
    wd = pd.Timestamp(t["entry_time"]).day_name()[:3]
    print(f"{t['entry_time'][:16]} {wd} {t['dir']:>4} {won:>4} "
          f"${t['pnl']:>+8.2f}  ${t['balance']:>9.2f}")
