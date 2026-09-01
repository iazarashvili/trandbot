"""Test sweep+FVG strategy for XAUUSD."""
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cfg = importlib.import_module("config_xauusd")
sys.modules["config"] = cfg

import MetaTrader5 as mt5
from strategy import SMCStrategy
from config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    HTF_CANDLES_LOOKBACK, LTF_CANDLES_LOOKBACK,
    TREND_EMA_PERIOD, MAX_LTF_WAIT_CANDLES,
)

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
info = mt5.symbol_info("XAUUSD")
frames = {}
for key, tf, n in [("m5", mt5.TIMEFRAME_M5, 100000),
                    ("h1", mt5.TIMEFRAME_H1, 10000),
                    ("m15", mt5.TIMEFRAME_M15, 50000)]:
    rates = mt5.copy_rates_from_pos("XAUUSD", tf, 0, n)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    frames[key] = df
mt5.shutdown()

m5 = frames["m5"]
h1 = frames["h1"]
point = info.point
contract = info.trade_contract_size


def run_sweep_fvg(m5, h1, contract, point, rrr=2.0, require_sweep=True):
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

    BALANCE = 1000.0
    VOLUME = 0.01
    start = LTF_CANDLES_LOOKBACK
    while start < len(m5) and h_idx[start] < HTF_CANDLES_LOOKBACK:
        start += 1

    trades = []
    balance = BALANCE
    poi = None
    poi_h = -1
    position = None
    watch_key = None
    watch_start = None
    abandoned = False
    htf_cache = {}

    for t in range(start, len(m5) - 1):
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
                balance += pnl
                risk_usd = p["risk_px"] * VOLUME * contract
                p.update(
                    exit_time=str(times.iloc[t]),
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

        # SWEEP CHECK
        if require_sweep:
            sweep = SMCStrategy.detect_liquidity_sweep(
                ltf, swing_strength=3, use_closed_candles=False)
            if sweep is None:
                continue
            if poi.type == "BULLISH" and sweep.direction != "BULLISH":
                continue
            if poi.type == "BEARISH" and sweep.direction != "BEARISH":
                continue

        # FVG confirmation
        setup = SMCStrategy.check_ltf_confirmation(
            ltf, poi, rrr, use_closed_candles=False,
            stop_mode="ob", buffer_atr=0.5)
        if setup is None:
            continue

        entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
        sl = setup["sl"]
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        tp = entry + risk * rrr if setup["direction"] == "BUY" else entry - risk * rrr

        position = {
            "dir": setup["direction"],
            "entry_time": str(times.iloc[t + 1]),
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "risk_px": round(risk, 2),
            "entry_bar": t + 1,
        }

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

    return {
        "trades": len(trades), "wins": wins, "losses": losses,
        "wr": round(wins / len(trades) * 100, 1) if trades else 0,
        "pf": pf, "net": round(net, 2), "max_dd_pct": max_dd_pct,
        "trade_list": trades, "balance": round(balance, 2),
    }


# Run comparisons
print(f"{'Config':<35} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PF':>6} {'Net':>10} {'MaxDD':>7}")
print("-" * 80)

configs = [
    ("No sweep, RRR 3.5 (current)",    False, 3.5),
    ("No sweep, RRR 2.0",              False, 2.0),
    ("No sweep, RRR 2.5",              False, 2.5),
    ("SWEEP + FVG, RRR 1.5",           True,  1.5),
    ("SWEEP + FVG, RRR 2.0",           True,  2.0),
    ("SWEEP + FVG, RRR 2.5",           True,  2.5),
    ("SWEEP + FVG, RRR 3.0",           True,  3.0),
    ("SWEEP + FVG, RRR 3.5",           True,  3.5),
]

for label, sweep, rrr in configs:
    r = run_sweep_fvg(m5, h1, contract, point, rrr=rrr, require_sweep=sweep)
    sign = "+" if r["net"] >= 0 else ""
    print(f"{label:<35} {r['trades']:>7} {r['wins']:>5} {r['wr']:>5.1f}% {str(r['pf']):>6} "
          f"{sign}${r['net']:>8.2f} {r['max_dd_pct']:>6.1f}%")

# Show trades for best sweep variant
print()
print("=== SWEEP+FVG RRR 2.0 TRADES ===")
r = run_sweep_fvg(m5, h1, contract, point, rrr=2.0, require_sweep=True)
for t in r["trade_list"]:
    won = "WIN" if t["pnl"] > 0 else "LOSS"
    print(f"{t['entry_time'][:16]} {t['dir']:>4} {won:>4} "
          f"entry={t['entry']:>8.2f} sl={t['sl']:>8.2f} tp={t['tp']:>8.2f} "
          f"pnl={t['pnl']:>+8.2f} bars={t['bars_held']:>4}")
