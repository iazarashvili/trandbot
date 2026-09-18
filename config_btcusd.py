"""BTCUSD-specific overrides. Everything else comes from config.py."""
from config import *  # noqa: F401,F403

SYMBOL = "BTCUSD"
LOT_SIZE = 0.01           # fallback if USE_RISK_BASED_LOT=False
USE_RISK_BASED_LOT = True
RISK_PERCENT = 1.0        # 1% of balance per trade
MAGIC_NUMBER = 100200
STOP_MODE = "window"
MAX_RISK_USD = 0.0     # disabled — %-based sizing controls risk
MAX_RISK_PCT = 0.0
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 23
BLOCKED_DAYS = []

# Backtest 2026-09-16: SB at 15:xx = 39 trades, 23% WR, -$1044.
# SB at 08:xx = 38 trades, 45% WR, +$1360. Block the losing window.
BLOCKED_SB_HOURS = [15]

# Backtest 2026-09-16: AMD at 09-10:xx = 8 trades, 0% WR, -$763.
BLOCKED_AMD_HOURS = [9, 10]

# IFVG filter: PF 1.23->1.60, WR 27->34%, Net x2, MaxDD 3.5->2.3% (2026-09-01)
# NOT WIRED — measured in reports/test_ict_filters.py only.  SymbolConfig loads
# this flag (core/symbol_context.py:119) but no executing path reads it, so it
# currently has no effect on live trading or on reports/engine.py.
USE_IFVG = True
