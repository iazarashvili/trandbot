"""Check opposite signals while XAUUSD position is open."""
import sys
import importlib
from pathlib import Path
from datetime import datetime, timedelta

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
)

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
info = mt5.symbol_info("XAUUSD")
END = datetime.now()
START = END - timedelta(days=90)
frames = {}
for key, tf in [("m5", mt5.TIMEFRAME_M5), ("h1", mt5.TIMEFRAME_H1)]:
    r = mt5.copy_rates_range("XAUUSD", tf, START, END)
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[(df["time"] >= pd.Timestamp(START)) & (df["time"] <= pd.Timestamp(END))]
    frames[key] = df
mt5.shutdown()

m5 = frames["m5"]
h1 = frames["h1"]
point = info.point
contract = info.trade_contract_size
VOLUME = 0.01

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

start = LTF_CANDLES_LOOKBACK
while start < len(m5) and h_idx[start] < HTF_CANDLES_LOOKBACK:
    start += 1


def get_signal(t, htf_cache):
    """Check if bar t has a valid sweep+FVG signal. Returns (setup, poi, htf_cache) or None."""
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
        return None

    mid = cl[t] + spread[t] / 2.0
    if not (poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom)):
        return None

    ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
    if not SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
        return None

    if SMCStrategy.is_consolidating(ltf, use_closed_candles=False):
        return None

    sweep = SMCStrategy.detect_liquidity_sweep(ltf, swing_strength=3, use_closed_candles=False)
    if sweep is None:
        return None
    if poi.type == "BULLISH" and sweep.direction != "BULLISH":
        return None
    if poi.type == "BEARISH" and sweep.direction != "BEARISH":
        return None

    setup = SMCStrategy.check_ltf_confirmation(
        ltf, poi, 3.5, use_closed_candles=False,
        stop_mode="ob", buffer_atr=0.5)
    return setup


htf_cache = {}
position = None
trades = []
opposite_signals = []
same_signals_blocked = 0

for t in range(start, len(m5) - 1):
    # Manage position
    if position is not None:
        p = position
        if p["dir"] == "BUY":
            hit_sl = lo[t] <= p["sl"]
            hit_tp = hi[t] >= p["tp"]
        else:
            hit_sl = hi[t] + spread[t] >= p["sl"]
            hit_tp = lo[t] + spread[t] <= p["tp"]

        if hit_sl or hit_tp:
            exit_px = p["sl"] if hit_sl else p["tp"]
            pnl = ((exit_px - p["entry"]) if p["dir"] == "BUY"
                   else (p["entry"] - exit_px)) * VOLUME * contract
            p["pnl"] = round(pnl, 2)
            p["exit_time"] = str(times.iloc[t])[:16]
            p["exit_reason"] = "SL" if hit_sl else "TP"
            trades.append(p)
            position = None
            continue

        # Check for signals while in position
        if _weekdays[t] >= 5:
            continue
        setup = get_signal(t, htf_cache)
        if setup is not None:
            if setup["direction"] != p["dir"]:
                cur_price = cl[t] if p["dir"] == "BUY" else cl[t] + spread[t]
                cur_pnl = ((cur_price - p["entry"]) if p["dir"] == "BUY"
                           else (p["entry"] - cur_price)) * VOLUME * contract
                opposite_signals.append({
                    "time": str(times.iloc[t])[:16],
                    "pos_dir": p["dir"],
                    "pos_entry_time": p["entry_time"][:16],
                    "signal_dir": setup["direction"],
                    "cur_pnl": round(cur_pnl, 2),
                    "trade_idx": len(trades),
                })
            else:
                same_signals_blocked += 1
        continue

    # No position
    if _weekdays[t] >= 5:
        continue

    setup = get_signal(t, htf_cache)
    if setup is None:
        continue

    entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
    sl = setup["sl"]
    risk = abs(entry - sl)
    if risk <= 0:
        continue
    tp = entry + risk * 3.5 if setup["direction"] == "BUY" else entry - risk * 3.5

    position = {
        "dir": setup["direction"],
        "entry_time": str(times.iloc[t + 1]),
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "risk_px": round(risk, 2),
        "entry_bar": t + 1,
    }

# Fill final P&L
for opp in opposite_signals:
    idx = opp["trade_idx"]
    if idx < len(trades):
        opp["final_pnl"] = trades[idx]["pnl"]
        opp["final_reason"] = trades[idx]["exit_reason"]

print("=== OPPOSITE SIGNALS WHILE POSITION OPEN (XAUUSD, 3 months) ===")
print()
print(f"Total trades taken: {len(trades)}")
print(f"Same-direction signals blocked (already in position): {same_signals_blocked}")
print(f"Opposite signals while in position: {len(opposite_signals)}")
print()

if opposite_signals:
    print(f"{'Signal time':<18} {'Pos':>4} {'Signal':>6} {'Cur P&L':>10} {'Final P&L':>10} {'Result':>8}")
    print("-" * 60)
    for opp in opposite_signals:
        final = opp.get("final_pnl", "?")
        reason = opp.get("final_reason", "?")
        print(f"{opp['time']:<18} {opp['pos_dir']:>4} {opp['signal_dir']:>6} "
              f"${opp['cur_pnl']:>+8.2f} ${final:>+8.2f} {reason:>8}")

    print()
    better = sum(1 for opp in opposite_signals
                 if opp.get("final_pnl") is not None and opp["cur_pnl"] > opp["final_pnl"])
    print(f"Closing on opposite signal would have been better: {better}/{len(opposite_signals)}")
else:
    print("No opposite signals found.")
