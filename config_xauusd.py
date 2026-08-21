"""XAUUSD-specific overrides. Everything else comes from config.py."""
from config import *  # noqa: F401,F403

SYMBOL = "XAUUSD"
LOT_SIZE = 0.01           # fallback if USE_RISK_BASED_LOT=False
USE_RISK_BASED_LOT = True
RISK_PERCENT = 1.0        # 1% of balance per trade
MAGIC_NUMBER = 100201
STOP_MODE = "ob"
MAX_RISK_USD = 0.0
MAX_RISK_PCT = 0.0
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 0
BLOCKED_DAYS = []
