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
