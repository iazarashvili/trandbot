"""GBPUSD-specific overrides. Everything else comes from config.py."""
from config import *  # noqa: F401,F403

SYMBOL = "GBPUSD"
LOT_SIZE = 0.01           # fallback if USE_RISK_BASED_LOT=False
USE_RISK_BASED_LOT = True
RISK_PERCENT = 1.0        # 1% of balance per trade
MAGIC_NUMBER = 100202
STOP_MODE = "ob"
MAX_RISK_USD = 0.0
MAX_RISK_PCT = 0.0
# Only trade during NY session (13:00-21:00 UTC) — best results.
NIGHT_START_HOUR = 21
NIGHT_END_HOUR = 13
# Block Monday and Tuesday — both losing days on GBPUSD.
BLOCKED_DAYS = [0, 1]  # 0=Monday, 1=Tuesday

TRAILING_STOP_TRIGGER_PCT = 0.50
TRAILING_STOP_DISTANCE_PCT = 0.30
