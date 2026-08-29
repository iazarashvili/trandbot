"""Backtest all 4 symbols, last 30 days, $1000 balance, daily P&L per symbol."""
import sys
import importlib
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

END = datetime.now()
START = END - timedelta(days=30)
BALANCE = 1000.0

symbols_cfg = [
    ("BTCUSD", 0.01, "config_btcusd"),
    ("XAUUSD", 0.01, "config_xauusd"),
    ("GBPUSD", 0.01, "config_gbpusd"),
    ("EURUSD", 0.01, "config_eurusd"),
]

# Pull data
all_data = {}
for sym, lot, cfg_name in symbols_cfg:
    info = mt5.symbol_info(sym)
    if info is None:
        print(f"WARNING: {sym} not found, skipping")
        continue
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
    all_data[sym] = {
        "frames": frames,
        "contract": info.trade_contract_size,
        "point": info.point,
        "lot": lot,
        "cfg_name": cfg_name,
    }

mt5.shutdown()

# Run each symbol, collect trades
all_trades = []
symbol_trades = defaultdict(list)

for sym, lot, cfg_name in symbols_cfg:
    if sym not in all_data:
        continue
    d = all_data[sym]
    cfg = importlib.import_module(cfg_name)
    sys.modules["config"] = cfg

    import reports.engine as eng
    importlib.reload(eng)
    eng.START_BALANCE = BALANCE
    eng.VOLUME = lot
    eng._HTF_CACHE.clear()

    res = eng.run(d["frames"]["m5"], d["frames"]["h1"], d["frames"]["m15"],
                  d["contract"], d["point"], rrr=3.0, invert=False,
                  use_liq_tp=True, use_partial=True)

    for t in res["trades"]:
        t["symbol"] = sym
        all_trades.append(t)
        symbol_trades[sym].append(t)

# Sort by exit time
all_trades.sort(key=lambda t: t.get("exit_time", ""))

# Replay on shared balance
balance = BALANCE
for t in all_trades:
    balance += t["pnl"]
    t["shared_balance"] = round(balance, 2)

# === DAILY P&L PER SYMBOL ===
print(f"{'='*80}")
print(f"  DAILY P&L REPORT | {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d')} | Balance: ${BALANCE:,.0f}")
print(f"{'='*80}")

# Collect all dates
daily = defaultdict(lambda: defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}))
for t in all_trades:
    day = t["exit_time"][:10]
    sym = t["symbol"]
    daily[day][sym]["trades"] += 1
    daily[day][sym]["pnl"] += t["pnl"]
    if t["pnl"] > 0:
        daily[day][sym]["wins"] += 1
    else:
        daily[day][sym]["losses"] += 1

syms = ["BTCUSD", "XAUUSD", "GBPUSD", "EURUSD"]
header = f"{'Date':<12}"
for s in syms:
    header += f"{'|':>2} {s:>18}"
header += f"{'|':>2} {'TOTAL':>10}"
print(header)
print("-" * len(header))

running_balance = BALANCE
for day in sorted(daily):
    row = f"{day:<12}"
    day_total = 0.0
    for s in syms:
        d = daily[day][s]
        if d["trades"] > 0:
            sign = "+" if d["pnl"] >= 0 else ""
            row += f"{'|':>2} {sign}${d['pnl']:>8.2f} ({d['wins']}W/{d['losses']}L)"
        else:
            row += f"{'|':>2} {'---':>18}"
        day_total += d["pnl"]
    running_balance += day_total
    sign = "+" if day_total >= 0 else ""
    row += f"{'|':>2} {sign}${day_total:>7.2f}"
    print(row)

print("-" * len(header))

# === PER SYMBOL SUMMARY ===
print(f"\n{'='*60}")
print(f"  PER SYMBOL SUMMARY")
print(f"{'='*60}")
print(f"{'Symbol':<10} {'Trades':>7} {'Wins':>5} {'Loss':>6} {'WinR%':>7} {'Net P&L':>12} {'PF':>6}")
print("-" * 60)

total_trades = 0
total_wins = 0
total_losses = 0
total_pnl = 0.0

for sym in syms:
    tt = symbol_trades.get(sym, [])
    wins = len([t for t in tt if t["pnl"] > 0])
    losses = len([t for t in tt if t["pnl"] <= 0])
    net = sum(t["pnl"] for t in tt)
    gross_w = sum(t["pnl"] for t in tt if t["pnl"] > 0)
    gross_l = -sum(t["pnl"] for t in tt if t["pnl"] <= 0)
    pf = round(gross_w / gross_l, 2) if gross_l > 0 else "-"
    wr = round(wins / len(tt) * 100, 1) if tt else 0
    sign = "+" if net >= 0 else ""
    print(f"{sym:<10} {len(tt):>7} {wins:>5} {losses:>6} {wr:>6.1f}% {sign}${net:>10,.2f} {str(pf):>6}")
    total_trades += len(tt)
    total_wins += wins
    total_losses += losses
    total_pnl += net

print("-" * 60)
wr = round(total_wins / total_trades * 100, 1) if total_trades else 0
gross_w = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
gross_l = -sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)
pf = round(gross_w / gross_l, 2) if gross_l > 0 else "-"
sign = "+" if total_pnl >= 0 else ""
print(f"{'TOTAL':<10} {total_trades:>7} {total_wins:>5} {total_losses:>6} {wr:>6.1f}% {sign}${total_pnl:>10,.2f} {str(pf):>6}")

# Drawdown
peak = BALANCE
max_dd = 0
for t in all_trades:
    peak = max(peak, t["shared_balance"])
    dd = peak - t["shared_balance"]
    max_dd = max(max_dd, dd)
max_dd_pct = max_dd / peak * 100 if peak > 0 else 0

print(f"\n${BALANCE:,.0f} -> ${balance:,.2f} ({(balance/BALANCE - 1)*100:.1f}%)")
print(f"MaxDD: ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
