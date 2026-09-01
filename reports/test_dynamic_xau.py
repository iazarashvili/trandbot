"""Test dynamic RRR on XAUUSD (with sweep)."""
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cfg = importlib.import_module("config_xauusd")
sys.modules["config"] = cfg

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from strategy import SMCStrategy
from config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    HTF_CANDLES_LOOKBACK, LTF_CANDLES_LOOKBACK,
    TREND_EMA_PERIOD, MAX_LTF_WAIT_CANDLES,
    SKIP_WEEKENDS, BLOCKED_DAYS, NIGHT_START_HOUR, NIGHT_END_HOUR, STOP_MODE,
)

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
info = mt5.symbol_info("XAUUSD")
frames = {}
for key, tf, n in [("m5", mt5.TIMEFRAME_M5, 200000),
                    ("h1", mt5.TIMEFRAME_H1, 10000)]:
    rates = mt5.copy_rates_from_pos("XAUUSD", tf, 0, n)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    frames[key] = df
mt5.shutdown()

m5 = frames["m5"]
h1 = frames["h1"]
point = info.point
contract = info.trade_contract_size
BALANCE = 1000.0

print(f"Data: {m5['time'].iloc[0]} to {m5['time'].iloc[-1]}")
print()

h1_ranges = (h1["high"] - h1["low"]).to_numpy(float)
h1_atr14 = pd.Series(h1_ranges).rolling(14).mean().to_numpy()
h1_atr100 = pd.Series(h1_ranges).rolling(100).mean().to_numpy()
h1_atr_ratio = h1_atr14 / np.where(h1_atr100 > 0, h1_atr100, 1.0)


def run_test(rrr_mode="fixed", base_rrr=3.5, high_rrr=2.0, atr_threshold=1.5, be_r=2.0):
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
                    p["sl"] = p["entry"]; p["be"] = True
                elif p["dir"] == "SELL" and lo[t] + spread[t] <= p["entry"] - td:
                    p["sl"] = p["entry"]; p["be"] = True
            if p["dir"] == "BUY":
                hs = lo[t] <= p["sl"]; ht = hi[t] >= p["tp"]
            else:
                hs = hi[t] + spread[t] >= p["sl"]; ht = lo[t] + spread[t] <= p["tp"]
            if hs or ht:
                ep = p["sl"] if hs else p["tp"]
                pnl = ((ep - p["entry"]) if p["dir"] == "BUY"
                       else (p["entry"] - ep)) * 0.01 * contract
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "balance": round(balance, 2),
                               "entry_time": p["entry_time"], "rrr_used": p["rrr_used"],
                               "atr_ratio": p.get("atr_ratio", 0)})
                pos = None; wk = None; ab = False
                if balance <= 0: break
            continue

        if SKIP_WEEKENDS and _weekdays[t] >= 5: continue
        if BLOCKED_DAYS and _weekdays[t] in BLOCKED_DAYS: continue

        if h_idx[t] != poi_h:
            poi_h = h_idx[t]
            if poi_h in htf_cache:
                trend, poi = htf_cache[poi_h]
            else:
                wh = h1.iloc[max(0, poi_h - HTF_CANDLES_LOOKBACK + 1): poi_h + 1]
                trend = SMCStrategy.get_htf_trend(wh, TREND_EMA_PERIOD, use_closed_candles=False)
                poi = SMCStrategy.detect_htf_poi(wh, use_trend_filter=True,
                        ema_period=TREND_EMA_PERIOD, use_closed_candles=False)
                htf_cache[poi_h] = (trend, poi)
            if poi is None: wk = None; ab = False

        if poi is None: continue
        mid = cl[t] + spread[t] / 2.0
        if not (poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom)):
            wk = None; ab = False; continue
        ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
        if not SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
            wk = None; ab = False; continue
        zk = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
        if zk != wk: wk = zk; ws = t; ab = False
        if ab: continue
        if (t - ws + 1) > MAX_LTF_WAIT_CANDLES: ab = True; continue
        if SMCStrategy.is_consolidating(ltf, use_closed_candles=False): continue

        # Sweep required for XAUUSD
        sw = SMCStrategy.detect_liquidity_sweep(ltf, swing_strength=3, use_closed_candles=False)
        if sw is None: continue
        if poi.type == "BULLISH" and sw.direction != "BULLISH": continue
        if poi.type == "BEARISH" and sw.direction != "BEARISH": continue

        hi_idx = h_idx[t]
        atr_r = h1_atr_ratio[hi_idx] if 0 <= hi_idx < len(h1_atr_ratio) else 1.0
        if np.isnan(atr_r): atr_r = 1.0

        if rrr_mode == "dynamic":
            rrr = high_rrr if atr_r > atr_threshold else base_rrr
        elif rrr_mode == "skip":
            if atr_r > atr_threshold: continue
            rrr = base_rrr
        else:
            rrr = base_rrr

        setup = SMCStrategy.check_ltf_confirmation(ltf, poi, rrr, use_closed_candles=False,
                    stop_mode=STOP_MODE, buffer_atr=0.5)
        if setup is None: continue

        entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
        sl = setup["sl"]; risk = abs(entry - sl)
        if risk <= 0: continue
        tp = entry + risk * rrr if setup["direction"] == "BUY" else entry - risk * rrr
        pos = {"dir": setup["direction"], "entry_time": str(times.iloc[t + 1]),
               "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
               "risk_px": round(risk, 2), "entry_bar": t + 1,
               "rrr_used": rrr, "atr_ratio": round(atr_r, 2)}

    wins = len([x for x in trades if x["pnl"] > 0])
    net = balance - BALANCE
    gw = sum(x["pnl"] for x in trades if x["pnl"] > 0)
    gl = -sum(x["pnl"] for x in trades if x["pnl"] <= 0)
    pf = round(gw / gl, 2) if gl > 0 else "-"
    wr = round(wins / len(trades) * 100, 1) if trades else 0
    peak = BALANCE; mdd = 0
    for tr in trades:
        peak = max(peak, tr["balance"]); dd = peak - tr["balance"]; mdd = max(mdd, dd)

    # Analyze high vol vs low vol trades
    hv = [x for x in trades if x["atr_ratio"] > 1.5]
    lv = [x for x in trades if x["atr_ratio"] <= 1.5]
    hv_net = sum(x["pnl"] for x in hv)
    lv_net = sum(x["pnl"] for x in lv)

    return {"t": len(trades), "w": wins, "wr": wr, "pf": pf,
            "net": round(net, 2), "mdd": round(mdd / peak * 100, 1) if peak else 0,
            "hv": len(hv), "hv_net": round(hv_net, 2),
            "lv": len(lv), "lv_net": round(lv_net, 2)}


print(f"{'Config':<40} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net':>10} {'MaxDD':>6} "
      f"| {'HiVol':>5} {'HV$':>8} {'LoVol':>5} {'LV$':>8}")
print("-" * 110)

configs = [
    ("Fixed RRR 3.5, BE 2R (current)",    "fixed",   3.5, 0,   0,   2.0),
    ("Dynamic: 3.5/2.0 @ ATR>1.3",        "dynamic", 3.5, 2.0, 1.3, 2.0),
    ("Dynamic: 3.5/2.0 @ ATR>1.5",        "dynamic", 3.5, 2.0, 1.5, 2.0),
    ("Dynamic: 3.5/2.0 @ ATR>1.8",        "dynamic", 3.5, 2.0, 1.8, 2.0),
    ("Dynamic: 3.5/1.5 @ ATR>1.3",        "dynamic", 3.5, 1.5, 1.3, 2.0),
    ("Dynamic: 3.5/1.5 @ ATR>1.5",        "dynamic", 3.5, 1.5, 1.5, 2.0),
    ("Skip @ ATR>1.5",                     "skip",    3.5, 0,   1.5, 2.0),
    ("Skip @ ATR>1.8",                     "skip",    3.5, 0,   1.8, 2.0),
]

for label, mode, base, high, thresh, be in configs:
    r = run_test(rrr_mode=mode, base_rrr=base, high_rrr=high, atr_threshold=thresh, be_r=be)
    sign = "+" if r["net"] >= 0 else ""
    print(f"{label:<40} {r['t']:>6} {r['wr']:>5.1f}% {str(r['pf']):>6} "
          f"{sign}${r['net']:>8.2f} {r['mdd']:>5.1f}% "
          f"| {r['hv']:>5} ${r['hv_net']:>+7.2f} {r['lv']:>5} ${r['lv_net']:>+7.2f}")
