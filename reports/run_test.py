"""Quick backtest for a single symbol with its own config.

Usage: python run_test.py SYMBOL LOT BALANCE
"""
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYM = sys.argv[1]
LOT = float(sys.argv[2])
BAL = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0

# Load symbol-specific config, swap it in as 'config' so engine picks it up
try:
    sym_cfg = importlib.import_module(f"config_{SYM.lower()}")
    sys.modules["config"] = sym_cfg
except ModuleNotFoundError:
    pass  # fall back to default config

from collections import defaultdict
import MetaTrader5 as mt5
import pandas as pd
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
import reports.engine as eng
from reports.engine import run, _HTF_CACHE

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
info = mt5.symbol_info(SYM)
frames = {}
for key, tf, n in [("m5", mt5.TIMEFRAME_M5, 100000),
                    ("h1", mt5.TIMEFRAME_H1, 10000),
                    ("m15", mt5.TIMEFRAME_M15, 50000)]:
    rates = mt5.copy_rates_from_pos(SYM, tf, 0, n)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    frames[key] = df
mt5.shutdown()

eng.START_BALANCE = BAL
eng.VOLUME = LOT
_HTF_CACHE.clear()

res = run(frames["m5"], frames["h1"], frames["m15"],
          info.trade_contract_size, info.point, rrr=3.0, invert=False,
          use_liq_tp=True, use_partial=True)
s = res["summary"]

months = defaultdict(list)
for t in res["trades"]:
    months[t["entry_time"][:7]].append(t)

# Show which config was used
from config import STOP_MODE, MAX_RISK_USD, NIGHT_START_HOUR, NIGHT_END_HOUR
print(f"=== {SYM} ({LOT} lot, ${BAL:.0f}) | stop={STOP_MODE} | "
      f"maxRisk=${MAX_RISK_USD} | night={NIGHT_START_HOUR}-{NIGHT_END_HOUR} ===")
print(f"{'Month':<10} {'Trades':>7} {'Wins':>5} {'Loss':>6} {'Net':>10} {'Balance':>10}")
print("-" * 52)
for m in sorted(months):
    tt = months[m]
    wins = len([t for t in tt if t["pnl"] > 0])
    losses = len([t for t in tt if t["pnl"] <= 0])
    net = sum(t["pnl"] for t in tt)
    bal = tt[-1]["balance"]
    sign = "+" if net >= 0 else ""
    print(f"{m:<10} {len(tt):>7} {wins:>5} {losses:>6} {sign}${net:>8.2f} ${bal:>8.2f}")
print("-" * 52)
sign = "+" if s["net_pnl"] >= 0 else ""
print(f"{'TOTAL':<10} {s['trades']:>7} {s['wins']:>5} {s['losses']:>6} "
      f"{sign}${s['net_pnl']:>8.2f} ${s['end_balance']:>8.2f}")
print(f"${BAL:.0f} -> ${s['end_balance']:.2f} ({s['return_pct']}%) | "
      f"PF {s['profit_factor']} | WinR {s['win_rate']}% | MaxDD {s['max_drawdown_pct']}%")
