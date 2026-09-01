"""XAUUSD trade analysis: day-of-week stats and MFE (max favorable excursion)."""
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
START = END - timedelta(days=500)

frames = {}
for key, tf in [("m5", mt5.TIMEFRAME_M5), ("h1", mt5.TIMEFRAME_H1)]:
    r = mt5.copy_rates_range("XAUUSD", tf, START, END)
    if r is None or len(r) == 0:
        r = mt5.copy_rates_from_pos("XAUUSD", tf, 0, 100000)
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    frames[key] = df
mt5.shutdown()

m5 = frames["m5"]
h1 = frames["h1"]
point = info.point
contract = info.trade_contract_size
VOLUME = 0.01
BALANCE = 1000.0
RRR = 3.5

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
        # Track MFE/MAE
        if p["dir"] == "BUY":
            fav = hi[t] - p["entry"]
            adv = p["entry"] - lo[t]
            hit_sl = lo[t] <= p["sl"]
            hit_tp = hi[t] >= p["tp"]
        else:
            fav = p["entry"] - (lo[t] + spread[t])
            adv = (hi[t] + spread[t]) - p["entry"]
            hit_sl = hi[t] + spread[t] >= p["sl"]
            hit_tp = lo[t] + spread[t] <= p["tp"]

        p["mfe"] = max(p.get("mfe", 0), fav)
        p["mae"] = max(p.get("mae", 0), adv)

        if hit_sl or hit_tp:
            exit_px = p["sl"] if hit_sl else p["tp"]
            pnl = ((exit_px - p["entry"]) if p["dir"] == "BUY"
                   else (p["entry"] - exit_px)) * VOLUME * contract
            balance += pnl
            risk_usd = p["risk_px"] * VOLUME * contract
            p.update(
                exit_time=str(times.iloc[t]),
                exit_reason="SL" if hit_sl else "TP",
                pnl=round(pnl, 2),
                balance=round(balance, 2),
                r_multiple=round(pnl / risk_usd, 2) if risk_usd > 0 else 0,
                mfe_r=round(p["mfe"] / p["risk_px"], 2) if p["risk_px"] > 0 else 0,
                mae_r=round(p["mae"] / p["risk_px"], 2) if p["risk_px"] > 0 else 0,
                entry_weekday=pd.Timestamp(p["entry_time"]).weekday(),
            )
            trades.append(p)
            position = None
            watch_key = None
            abandoned = False
            if balance <= 0:
                break
        continue

    if _weekdays[t] >= 5:
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

    sweep = SMCStrategy.detect_liquidity_sweep(ltf, swing_strength=3, use_closed_candles=False)
    if sweep is None:
        continue
    if poi.type == "BULLISH" and sweep.direction != "BULLISH":
        continue
    if poi.type == "BEARISH" and sweep.direction != "BEARISH":
        continue

    setup = SMCStrategy.check_ltf_confirmation(
        ltf, poi, RRR, use_closed_candles=False,
        stop_mode="ob", buffer_atr=0.5)
    if setup is None:
        continue

    entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
    sl = setup["sl"]
    risk = abs(entry - sl)
    if risk <= 0:
        continue
    tp = entry + risk * RRR if setup["direction"] == "BUY" else entry - risk * RRR

    position = {
        "dir": setup["direction"],
        "entry_time": str(times.iloc[t + 1]),
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "risk_px": round(risk, 2),
        "entry_bar": t + 1,
        "mfe": 0.0,
        "mae": 0.0,
    }

# === DAY OF WEEK ANALYSIS ===
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
print("=" * 60)
print("  DAY OF WEEK ANALYSIS")
print("=" * 60)
print(f"{'Day':<6} {'Trades':>7} {'Wins':>5} {'Loss':>6} {'WR%':>6} {'Net P&L':>10}")
print("-" * 45)

for d in range(5):
    day_trades = [t for t in trades if t["entry_weekday"] == d]
    if not day_trades:
        print(f"{day_names[d]:<6} {0:>7}")
        continue
    w = len([t for t in day_trades if t["pnl"] > 0])
    l = len(day_trades) - w
    net = sum(t["pnl"] for t in day_trades)
    wr = round(w / len(day_trades) * 100, 1)
    sign = "+" if net >= 0 else ""
    print(f"{day_names[d]:<6} {len(day_trades):>7} {w:>5} {l:>6} {wr:>5.1f}% {sign}${net:>8.2f}")

# === MFE ANALYSIS (losses that were in profit) ===
print()
print("=" * 60)
print("  MFE ANALYSIS - LOSSES THAT WERE IN PROFIT")
print("=" * 60)
print()

losses_with_profit = [t for t in trades if t["pnl"] <= 0 and t["mfe_r"] >= 0.5]
losses_with_profit.sort(key=lambda t: t["mfe_r"], reverse=True)

print(f"{'Date':<18} {'Dir':>5} {'MFE':>6} {'MAE':>6} {'P&L':>10} {'TP%':>6}")
print("-" * 55)

for t in losses_with_profit:
    tp_dist = abs(t["tp"] - t["entry"])
    mfe_pct = round(t["mfe"] / tp_dist * 100, 0) if tp_dist > 0 else 0
    print(f"{t['entry_time'][:16]:<18} {t['dir']:>5} {t['mfe_r']:>5.1f}R {t['mae_r']:>5.1f}R "
          f"${t['pnl']:>+8.2f} {mfe_pct:>5.0f}%")

print()
print(f"Total losses: {len([t for t in trades if t['pnl'] <= 0])}")
print(f"Losses that reached 50%+ of TP: {len(losses_with_profit)}")
print(f"Losses that reached 80%+ of TP: {len([t for t in losses_with_profit if t['mfe_r'] >= 0.8 * RRR])}")

# MFE distribution for all losses
all_losses = [t for t in trades if t["pnl"] <= 0]
print()
print("MFE distribution (losses only):")
for threshold in [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    count = len([t for t in all_losses if t["mfe_r"] >= threshold])
    print(f"  Reached {threshold:.1f}R+ before SL: {count}/{len(all_losses)}")

# === TRADES THAT REVERSED AFTER 50%+ PROFIT ===
print()
print("=" * 60)
print("  REVERSAL TRADES (reached 50%+ of TP, then hit SL)")
print("=" * 60)
print()

tp_dist_trades = []
for t in trades:
    tp_dist = abs(t["tp"] - t["entry"])
    mfe_pct = t["mfe"] / tp_dist * 100 if tp_dist > 0 else 0
    t["tp_pct_reached"] = round(mfe_pct, 1)
    if t["pnl"] <= 0 and mfe_pct >= 50:
        tp_dist_trades.append(t)

if tp_dist_trades:
    print(f"{'Date':<18} {'Dir':>5} {'TP% reached':>12} {'MFE':>6} {'P&L':>10}")
    print("-" * 55)
    for t in tp_dist_trades:
        print(f"{t['entry_time'][:16]:<18} {t['dir']:>5} {t['tp_pct_reached']:>10.1f}% "
              f"{t['mfe_r']:>5.1f}R ${t['pnl']:>+8.2f}")
    lost_amount = sum(t["pnl"] for t in tp_dist_trades)
    print(f"\nThese reversals cost: ${lost_amount:+.2f}")
else:
    print("No trades reached 50%+ of TP and then hit SL.")
