"""BTCUSD-specific overrides. Everything else comes from config.py."""
from config import *  # noqa: F401,F403

SYMBOL = "BTCUSD"
LOT_SIZE = 0.02
MAGIC_NUMBER = 100200
STOP_MODE = "window"
MAX_RISK_USD = 80.0
MAX_RISK_PCT = 0.0
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 23
