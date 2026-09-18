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

# Premium/Discount + IFVG: PF 1.25->inf, Net $10->$40, MaxDD 0% (2026-09-01)
# NOT WIRED — _try_pd_fvg is disabled in engine (lost money in 2026-09-01 backtest).
# USE_IFVG is loaded into SymbolConfig but no executing path reads it.
# Trailing stop is not implemented — SymbolConfig does not load these fields.
# Keeping the measurements here for reference if these features are ever wired.

# Backtest 2026-09-16: SB at 15:xx = 31 trades, 29% WR, -$491.
# SB at 19:xx = 15 trades, 47% WR, +$787. Block the losing window.
BLOCKED_SB_HOURS = [15]
