"""XAUUSD sweep+FVG backtest, last 3 months, $1000."""
import sys
import importlib
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cfg = importlib.import_module("config_xauusd")
sys.modules["config"] = cfg

import MetaTrader5 as mt5
import pandas as pd
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
from reports.test_sweep import run_sweep_fvg

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
info = mt5.symbol_info("XAUUSD")

END = datetime.now()
START = END - timedelta(days=90)

frames = {}
for key, tf in [("m5", mt5.TIMEFRAME_M5), ("h1", mt5.TIMEFRAME_H1), ("m15", mt5.TIMEFRAME_M15)]:
    r = mt5.copy_rates_range("XAUUSD", tf, START, END)
    if r is None or len(r) == 0:
        r = mt5.copy_rates_from_pos("XAUUSD", tf, 0, 100000)
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[(df["time"] >= pd.Timestamp(START)) & (df["time"] <= pd.Timestamp(END))]
    frames[key] = df
mt5.shutdown()

r = run_sweep_fvg(frames["m5"], frames["h1"], info.trade_contract_size, info.point,
                  rrr=3.5, require_sweep=True)

print("=" * 55)
print(f"  XAUUSD SWEEP+FVG | {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d')} | $1,000")
print("=" * 55)
print()
print(f"{'Date':<12} {'Dir':>5} {'W/L':>5} {'P&L':>12} {'Balance':>10}")
print("-" * 48)

for t in r["trade_list"]:
    won = "WIN" if t["pnl"] > 0 else "LOSS"
    print(f"{t['exit_time'][:10]:<12} {t['dir']:>5} {won:>5} "
          f"{t['pnl']:>+10.2f}   ${t['balance']:>8.2f}")

print("-" * 48)
print(f"{'TOTAL':<12} {r['trades']:>3}t  {r['wins']}W/{r['losses']}L "
      f"{r['net']:>+10.2f}   ${r['balance']:>8.2f}")
print()
print(f"$1,000 -> ${r['balance']:,.2f} ({(r['balance'] / 1000 - 1) * 100:.1f}%)")
print(f"WinRate: {r['wr']}% | PF: {r['pf']} | MaxDD: {r['max_dd_pct']}%")
