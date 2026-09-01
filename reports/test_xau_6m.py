"""XAUUSD sweep+FVG backtest, last 6 months, $2000, 0.03 lot."""
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
START = END - timedelta(days=180)

frames = {}
for key, tf in [("m5", mt5.TIMEFRAME_M5), ("h1", mt5.TIMEFRAME_H1)]:
    r = mt5.copy_rates_range("XAUUSD", tf, START, END)
    if r is None or len(r) == 0:
        r = mt5.copy_rates_from_pos("XAUUSD", tf, 0, 100000)
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[(df["time"] >= pd.Timestamp(START)) & (df["time"] <= pd.Timestamp(END))]
    frames[key] = df
mt5.shutdown()

m5 = frames["m5"]
h1 = frames["h1"]
point = info.point
contract = info.trade_contract_size

BALANCE = 2000.0
VOLUME = 0.03
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
        vol = VOLUME
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
                exit_time=str(times.iloc[t])[:16],
                exit_price=round(exit_px, 2),
                exit_reason="SL" if hit_sl else "TP",
                pnl=round(pnl, 2),
                bars_held=t - p["entry_bar"],
                balance=round(balance, 2),
                r_multiple=round(pnl / risk_usd, 2) if risk_usd > 0 else 0,
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
    }

# Results
wins = len([t for t in trades if t["pnl"] > 0])
losses = len(trades) - wins
net = balance - BALANCE
gw = sum(t["pnl"] for t in trades if t["pnl"] > 0)
gl = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
pf = round(gw / gl, 2) if gl > 0 else "-"

peak = BALANCE
max_dd = 0
for tr in trades:
    peak = max(peak, tr["balance"])
    dd = peak - tr["balance"]
    max_dd = max(max_dd, dd)
max_dd_pct = round(max_dd / peak * 100, 1) if peak > 0 else 0

print("=" * 65)
print(f"  XAUUSD SWEEP+FVG | {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d')}")
print(f"  Balance: ${BALANCE:,.0f} | Lot: {VOLUME} | RRR: {RRR}")
print("=" * 65)
print()
print(f"{'#':<4} {'Date':<18} {'Dir':>5} {'W/L':>5} "
      f"{'Entry':>10} {'SL':>10} {'TP':>10} {'P&L':>10} {'R':>6} {'Balance':>10}")
print("-" * 95)

for t in trades:
    won = "WIN" if t["pnl"] > 0 else "LOSS"
    print(f"{trades.index(t)+1:<4} {t['entry_time'][:16]:<18} {t['dir']:>5} {won:>5} "
          f"{t['entry']:>10.2f} {t['sl']:>10.2f} {t['tp']:>10.2f} "
          f"${t['pnl']:>+8.2f} {t['r_multiple']:>+5.2f} ${t['balance']:>9.2f}")

print("-" * 95)
print(f"TOTAL: {len(trades)} trades | {wins}W / {losses}L | WR: {round(wins/len(trades)*100,1) if trades else 0}%")
print()
print(f"${BALANCE:,.0f} -> ${balance:,.2f} ({(balance/BALANCE-1)*100:+.1f}%)")
print(f"Net: ${net:+,.2f} | PF: {pf} | MaxDD: ${max_dd:,.2f} ({max_dd_pct}%)")
print(f"Gross Win: ${gw:,.2f} | Gross Loss: ${gl:,.2f}")
