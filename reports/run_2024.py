"""Backtest all three symbols on 2024 data only."""
import sys
import importlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5
import pandas as pd
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

mt5.initialize()
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31, 23, 59)

symbols = [
    ("BTCUSD", 0.02, "config_btcusd"),
    ("XAUUSD", 0.01, "config_xauusd"),
    ("GBPUSD", 0.05, "config_gbpusd"),
]

for sym, lot, cfg_name in symbols:
    info = mt5.symbol_info(sym)
    frames = {}
    for key, tf in [("m5", mt5.TIMEFRAME_M5), ("h1", mt5.TIMEFRAME_H1),
                     ("m15", mt5.TIMEFRAME_M15)]:
        r = mt5.copy_rates_range(sym, tf, start, end)
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        frames[key] = df

    # Load symbol-specific config
    cfg = importlib.import_module(cfg_name)
    sys.modules["config"] = cfg

    import reports.engine as eng
    importlib.reload(eng)
    eng.START_BALANCE = 300.0
    eng.VOLUME = lot
    eng._HTF_CACHE.clear()

    res = eng.run(frames["m5"], frames["h1"], frames["m15"],
                  info.trade_contract_size, info.point, rrr=3.0, invert=False,
                  use_liq_tp=True, use_partial=True)
    s = res["summary"]

    months = defaultdict(list)
    for t in res["trades"]:
        months[t["entry_time"][:7]].append(t)

    from config import STOP_MODE, NIGHT_START_HOUR, NIGHT_END_HOUR
    bdays = getattr(cfg, "BLOCKED_DAYS", [])
    print(f"=== {sym} ({lot} lot, $300) 2024 | stop={STOP_MODE} | night={NIGHT_START_HOUR}-{NIGHT_END_HOUR} | blocked={bdays} ===")
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
    print(f"PF {s['profit_factor']} | WinR {s['win_rate']}% | MaxDD {s['max_drawdown_pct']}%")
    print()

mt5.shutdown()
