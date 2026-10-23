"""EURUSD-specific overrides. Everything else comes from config.py."""
from config import *  # noqa: F401,F403

SYMBOL = "EURUSD"
LOT_SIZE = 0.01           # fallback if USE_RISK_BASED_LOT=False
USE_RISK_BASED_LOT = True
RISK_PERCENT = 0.4        # overridden by ACCOUNT_TYPE if set
MAGIC_NUMBER = 100203
STOP_MODE = "ob"
MAX_RISK_USD = 0.0
MAX_RISK_PCT = 0.0
# NY session only, like GBPUSD — forex pairs perform best during London/NY.
NIGHT_START_HOUR = 21
NIGHT_END_HOUR = 13

# Backtest 2026-09-16: Mon -$591, Tue -$111. Match GBPUSD config.
BLOCKED_DAYS = [0, 1]  # 0=Monday, 1=Tuesday

# SMT: compare with GBPUSD for divergence confirmation
SMT_CORRELATED_SYMBOL = "GBPUSD"

# Backtest 2026-09-16: BEARISH BUY = 33 trades, 15% WR, PF 0.29, -$1792.
# BEARISH Friday = 17 trades, 0 wins, -$1536 (p<0.00001).
# Blocking BEARISH trend entirely: 66 trades removed, -$2456 -> +$314.
BLOCKED_TRENDS = ["BEARISH"]


# Trailing stop is not implemented — SymbolConfig does not load these fields.

