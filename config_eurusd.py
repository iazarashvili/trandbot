"""EURUSD-specific overrides. Everything else comes from config.py."""
from config import *  # noqa: F401,F403

SYMBOL = "EURUSD"
LOT_SIZE = 0.01           # fallback if USE_RISK_BASED_LOT=False
USE_RISK_BASED_LOT = True
RISK_PERCENT = 1.0        # 1% of balance per trade
MAGIC_NUMBER = 100203
STOP_MODE = "ob"
MAX_RISK_USD = 0.0
MAX_RISK_PCT = 0.0
# NY session only, like GBPUSD — forex pairs perform best during London/NY.
NIGHT_START_HOUR = 21
NIGHT_END_HOUR = 13
# Start with no blocked days — will test and adjust.
BLOCKED_DAYS = []

