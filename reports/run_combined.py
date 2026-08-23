"""Backtest all symbols on SHARED balance — trades happen in parallel."""
import sys
import importlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

START = datetime(2026, 1, 1)
END = datetime(2026, 8, 23, 23, 59)
BALANCE = 2500.0

symbols_cfg = [
    ("BTCUSD", 0.01, "config_btcusd"),
    ("XAUUSD", 0.01, "config_xauusd"),
    ("GBPUSD", 0.01, "config_gbpusd"),
    ("EURUSD", 0.01, "config_eurusd"),
]

# Pull data for all symbols
all_data = {}
for sym, lot, cfg_name in symbols_cfg:
    info = mt5.symbol_info(sym)
    frames = {}
    for key, tf in [("m5", mt5.TIMEFRAME_M5), ("h1", mt5.TIMEFRAME_H1),
                     ("m15", mt5.TIMEFRAME_M15)]:
        r = mt5.copy_rates_range(sym, tf, START, END)
        if r is None or len(r) == 0:
            r = mt5.copy_rates_from_pos(sym, tf, 0, 100000)
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        # Filter to date range
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

# Run each symbol independently, collect all trades with timestamps
all_trades = []

for sym, lot, cfg_name in symbols_cfg:
    d = all_data[sym]
    cfg = importlib.import_module(cfg_name)
    sys.modules["config"] = cfg

    import reports.engine as eng
    importlib.reload(eng)
    eng.START_BALANCE = BALANCE  # doesn't matter for trade P&L, only for risk cap
    eng.VOLUME = lot
    eng._HTF_CACHE.clear()

    res = eng.run(d["frames"]["m5"], d["frames"]["h1"], d["frames"]["m15"],
                  d["contract"], d["point"], rrr=3.0, invert=False,
                  use_liq_tp=True, use_partial=True)

    for t in res["trades"]:
        t["symbol"] = sym
        all_trades.append(t)

# Sort all trades by exit time (chronological order on shared balance)
all_trades.sort(key=lambda t: t.get("exit_time", ""))

# Replay on shared balance
balance = BALANCE
monthly = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
symbol_totals = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})

for t in all_trades:
    balance += t["pnl"]
    t["shared_balance"] = round(balance, 2)
    month = t["exit_time"][:7]
    monthly[month]["trades"] += 1
    monthly[month]["pnl"] += t["pnl"]
    if t["pnl"] > 0:
        monthly[month]["wins"] += 1
    else:
        monthly[month]["losses"] += 1

    sym = t["symbol"]
    symbol_totals[sym]["trades"] += 1
    symbol_totals[sym]["pnl"] += t["pnl"]
    if t["pnl"] > 0:
        symbol_totals[sym]["wins"] += 1

# Track drawdown
peak = BALANCE
max_dd = 0
for t in all_trades:
    peak = max(peak, t["shared_balance"])
    dd = peak - t["shared_balance"]
    max_dd = max(max_dd, dd)
max_dd_pct = max_dd / peak * 100 if peak > 0 else 0

# Print results
print(f"=== COMBINED PORTFOLIO ($2,500,000 shared balance) ===")
print(f"=== 2025-01 to 2026-08 | BTCUSD + XAUUSD + GBPUSD + EURUSD ===")
print()
print(f"{'Month':<10} {'Trades':>7} {'Wins':>5} {'Loss':>6} {'Net':>12} {'Balance':>12}")
print("-" * 56)
for m in sorted(monthly):
    d = monthly[m]
    sign = "+" if d["pnl"] >= 0 else ""
    # Find balance at end of month
    month_trades = [t for t in all_trades if t["exit_time"][:7] == m]
    bal = month_trades[-1]["shared_balance"] if month_trades else BALANCE
    print(f"{m:<10} {d['trades']:>7} {d['wins']:>5} {d['losses']:>6} "
          f"{sign}${d['pnl']:>10,.2f} ${bal:>10,.2f}")

print("-" * 56)
total_pnl = balance - BALANCE
sign = "+" if total_pnl >= 0 else ""
total_trades = len(all_trades)
total_wins = len([t for t in all_trades if t["pnl"] > 0])
total_losses = total_trades - total_wins
print(f"{'TOTAL':<10} {total_trades:>7} {total_wins:>5} {total_losses:>6} "
      f"{sign}${total_pnl:>10,.2f} ${balance:>10,.2f}")

print(f"\n${BALANCE:,.0f} -> ${balance:,.2f} ({total_pnl/BALANCE*100:.1f}%)")
print(f"MaxDD: ${max_dd:,.2f} ({max_dd_pct:.1f}%)")

gross_w = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
gross_l = -sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)
pf = round(gross_w / gross_l, 2) if gross_l > 0 else None
print(f"WinR: {total_wins/total_trades*100:.1f}% | PF: {pf}")

# Per-symbol breakdown
print(f"\n{'Symbol':<10} {'Trades':>7} {'Wins':>5} {'Net':>12}")
print("-" * 36)
for sym in ["BTCUSD", "XAUUSD", "GBPUSD", "EURUSD"]:
    s = symbol_totals[sym]
    sign = "+" if s["pnl"] >= 0 else ""
    print(f"{sym:<10} {s['trades']:>7} {s['wins']:>5} {sign}${s['pnl']:>10,.2f}")
