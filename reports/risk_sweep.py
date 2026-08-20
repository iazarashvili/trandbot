"""Sweep MAX_RISK_USD on XAUUSD to find optimal cap."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5
import pandas as pd
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

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

# We need to patch the engine's imported MAX_RISK_USD for each run
from reports import engine as eng
from strategy import SMCStrategy

print(f"{'MaxRisk$':>9} {'Trades':>7} {'WinR%':>6} {'Net':>10} {'PF':>6} {'MaxDD%':>7}")
print("-" * 50)

for cap in [0, 3, 5, 8, 10, 15, 20, 30, 50, 80]:
    # Monkey-patch the cap in the engine module
    eng.MAX_RISK_PCT = 0
    # We need to patch _open to use this cap; since _open reads from module globals,
    # let's just run with balance-based cap
    # Actually _open uses MAX_RISK_PCT from config import. Let's patch config directly.
    import config
    config.MAX_RISK_PCT = 0
    config.MAX_RISK_USD = cap
    config.NIGHT_START_HOUR = 0
    config.NIGHT_END_HOUR = 0

    import importlib
    importlib.reload(eng)
    eng.START_BALANCE = 100.0
    eng.VOLUME = 0.01
    eng._HTF_CACHE.clear()

    res = eng.run(frames["m5"], frames["h1"], frames["m15"],
                  info.trade_contract_size, info.point, rrr=3.0, invert=False,
                  use_liq_tp=True, use_partial=True)
    s = res["summary"]
    pf = str(s["profit_factor"]) if s["profit_factor"] else "-"
    sign = "+" if s["net_pnl"] >= 0 else ""
    label = "OFF" if cap == 0 else f"${cap}"
    print(f"{label:>9} {s['trades']:>7} {s['win_rate']:>6.1f} "
          f"{sign}${s['net_pnl']:>8.2f} {pf:>5} {s['max_drawdown_pct']:>7.2f}")
