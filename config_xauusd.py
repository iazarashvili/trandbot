"""XAUUSD-specific overrides. Everything else comes from config.py."""
from config import *  # noqa: F401,F403

SYMBOL = "XAUUSD"
LOT_SIZE = 0.01           # fallback if USE_RISK_BASED_LOT=False
USE_RISK_BASED_LOT = True
RISK_PERCENT = 1.0        # 1% of balance per trade
MAGIC_NUMBER = 100201
STOP_MODE = "ob"
MAX_RISK_USD = 40.0       # hard cap $40 per trade
MAX_RISK_PCT = 0.0
# Backtest 2026-09-16: Asian+London = 13 trades, 31% WR, +$263.
# NY AM+PM = 23 trades, 13% WR, -$431. Block NY sessions.
NIGHT_START_HOUR = 12
NIGHT_END_HOUR = 2

# Friday blocked: 0W/6L, -$150 over 47 trades (2026-08-29)
# Thursday blocked: 7 trades, 14% WR, -$413 (3 biggest losses on Thu, 2026-09-16)
BLOCKED_DAYS = [3, 4]  # 3=Thursday, 4=Friday

# RRR 3.5: measured peak on 81 trades — PF 1.31, expR +0.167 (2026-08-29)
RRR = 3.5

# Sweep filter ON: PF 1.42->1.84, MaxDD 15.1%->6.9%, filters 38 bad trades (2026-08-29)
USE_SWEEP_FILTER = True

# Breakeven at 2R: 6 trades reached 50%+ TP then reversed, cost $147 (2026-08-29)
USE_BREAKEVEN = True
BREAKEVEN_R = 2.0

# Liquidity TP OFF: liq TP trades lose -$70 on XAUUSD, fixed TP wins +$330
USE_LIQUIDITY_TP = False

# Partial close OFF: fixed-only (no partial) gives better net and lower MaxDD
USE_PARTIAL_CLOSE = False

# Backtest 2026-09-16: trades #4 (-$353) and #14 (-$136) had enormous OB stops
# that bypassed the $40 cap at min lot. Cap OB width at 80 price points.
MAX_OB_RISK_PX = 80.0

# Time stop: disabled — XAUUSD needs time, early close kills winners.
TIME_STOP_BARS = 0
TIME_STOP_MIN_PCT = 30

# Trailing stop: disabled for XAUUSD — needs time to reach TP.
TRAILING_STOP_TRIGGER_PCT = 0.0
TRAILING_STOP_DISTANCE_PCT = 0.30
