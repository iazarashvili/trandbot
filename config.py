"""Configuration settings for the MT5 SMC Trading Bot.

Credentials are read from the environment (or a local, git-ignored `.env`
file) — never hard-code them in this file.  See `.env.example`.
"""

import os
from pathlib import Path

import MetaTrader5 as mt5


# ==========================================
# MT5 Account Credentials
# ==========================================

def _load_env_file(filename: str = ".env") -> None:
    """Loads KEY=VALUE pairs from a local .env file into os.environ.

    Deliberately dependency-free.  Real environment variables always win over
    the file, so production deployments can ignore `.env` entirely.
    """
    env_path = Path(__file__).resolve().parent / filename
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill in your MT5 credentials."
        )
    return value


_load_env_file()

try:
    MT5_LOGIN: int = int(_require_env("MT5_LOGIN"))
except ValueError as exc:
    raise RuntimeError("MT5_LOGIN must be a number (your MT5 account id).") from exc

MT5_PASSWORD: str = _require_env("MT5_PASSWORD")
MT5_SERVER: str = _require_env("MT5_SERVER")

# ==========================================
# Trading Pair & Execution Parameters
# ==========================================
SYMBOL: str = "BTCUSD"
LOT_SIZE: float = 0.01
MAGIC_NUMBER: int = 100200
DEVIATION: int = 20  # max slippage in points accepted on a market order

# Timeframes — 1H OB zone, 5m displacement+FVG confirmation.
# Changed from M15/M1 on 2026-08-20: the user's manual approach uses 1H zones
# with 5m entry, and the bot hadn't fired in two days on the old timeframes.
HTF: int = mt5.TIMEFRAME_H1
LTF: int = mt5.TIMEFRAME_M5

# Strategy Parameters
HTF_CANDLES_LOOKBACK: int = 200
LTF_CANDLES_LOOKBACK: int = 100
# Risk to Reward Ratio.  Swept 1.0-4.0 over 34 days on 2026-08-17: inverted
# expectancy was positive at *every* value (that is the robust finding) and
# peaked at 2.5.  The peak itself is not meaningful — with ~23 trades the
# standard error on expectancy is about ±0.35R, so every value in the sweep
# sits inside one error bar of every other.  2.5 is a reasonable pick, not a
# measured optimum.  See reports/sweep_rrr.json.
RRR: float = 3.0

# Stop placement.  Measured on the same data (reports/stop_rules.json):
#   "window"  extreme of the last 10 M1 candles — expR +0.370  <- best
#   "swing"   extreme since the swing the MSS broke   — expR -0.045
#   "zone"    never tighter than the far side of the POI — expR +0.050
# The structural rules are more defensible in theory and measurably worse
# here: the bot is fading its own signal, so a tighter stop on the fade side
# is simply hit more often.  Keeping "window" on the evidence.
STOP_MODE: str = "window"  # "window" | "ob" | "swing" | "zone"
STOP_BUFFER_ATR: float = 0.5  # wick buffer for "swing"/"zone", in 1m ATRs
MAX_LTF_WAIT_CANDLES: int = 15  # Max 1m candles to wait for MSS after HTF touch

# Only evaluate fully closed candles.  Keep this True: the still-forming candle
# repaints, so a signal read from it can appear and vanish within one minute.
USE_CLOSED_CANDLES_ONLY: bool = True

# Trade against the strategy: a BUY signal is executed as a SELL and vice versa.
# The stop and target are mirrored around the entry, so the risk per trade is
# unchanged and the two modes compare directly.
#
# Set to False to go back to trading the signals as generated — that single
# change is the whole revert.  See [[invert-signals-experiment]] in memory.
INVERT_SIGNALS: bool = False

# System Loop Interval (seconds)
POLL_INTERVAL: int = 10

# Trend Filter Parameters
USE_TREND_FILTER: bool = True
TREND_EMA_PERIOD: int = 100  # EMA period used to define the macro trend

# Weekend filter.  Crypto markets are open 24/7, but weekend price action is
# thin and choppy.  Backtested on 70 days (2026-06 to 2026-08): keeping this
# True improved expectancy.  Saturday = 5, Sunday = 6 in Python's weekday().
SKIP_WEEKENDS: bool = True

# Night filter.  No new entries between these UTC hours.
# Open positions are still managed (SL/TP will fire normally).
# Disabled: backtested on 11 months — per-trade expR improved but total profit
# halved because 17 trades were removed.  Not worth it.
NIGHT_START_HOUR: int = 20
NIGHT_END_HOUR: int = 23

# Liquidity-based take profit.  Instead of a fixed RRR target, the TP is
# placed at the nearest swing high/low (liquidity pool) on the 15m chart.
# MIN_RRR_LIQUIDITY: skip the trade if the liquidity level is closer than
# this many R from entry — too little reward for the risk.
LIQUIDITY_TF: int = mt5.TIMEFRAME_M15
LIQUIDITY_CANDLES: int = 200       # how many 15m candles to scan for swings
SWING_STRENGTH: int = 3            # candles each side to confirm a swing point
MIN_RRR_LIQUIDITY: float = 0.5     # minimum R:R to take the trade

# ==========================================
# Risk Management
# ==========================================

# Position sizing: when True, LOT_SIZE is ignored and the volume is derived
# from RISK_PERCENT of the account balance and the distance to the stop loss.
#
# Note for small accounts: BTCUSD has a 0.01 lot minimum, which on this broker
# is 0.01 BTC — about $0.01 of loss per $1 of adverse move.  With a $128
# balance, 1% risk ($1.29) is only reachable while the stop sits within ~$129
# of entry.  Beyond that the minimum lot takes over and the real risk exceeds
# RISK_PERCENT; the connector logs a warning when that happens.
USE_RISK_BASED_LOT: bool = False
RISK_PERCENT: float = 1.0  # % of balance risked per trade

# Skip entries while the spread is abnormally wide.  0 disables the filter.
# Measured on Exness-MT5Trial15 BTCUSD: 700 points (= $7.00) at a quiet moment,
# against 1m stop distances of roughly $60 — so the spread alone costs ~12% of
# the risk on every entry.  Watch the logged spread across a session, then set
# this above the normal range but below the news/rollover spikes.
MAX_SPREAD_POINTS: int = 0

# Move the stop loss to break-even once the trade is this many R in profit.
USE_BREAKEVEN: bool = False
BREAKEVEN_TRIGGER_R: float = 1.0

# Partial close.  When price reaches PARTIAL_TRIGGER_PCT of the TP distance,
# close PARTIAL_CLOSE_PCT of the position, move stop to entry, and shift TP
# to the next liquidity level beyond the original TP.
# Maximum risk per trade.  Two modes:
#   MAX_RISK_PCT > 0  →  risk capped at this % of account balance (preferred)
#   MAX_RISK_USD > 0  →  hard dollar cap (fallback if PCT is 0)
#   Both 0             →  no cap
# 0.5% tested on XAUUSD: passes 63% of trades, cuts outsized stops.
MAX_RISK_PCT: float = 0.0    # % of account balance (0 = disabled)
MAX_RISK_USD: float = 80.0   # hard dollar cap (0 = disabled)

USE_PARTIAL_CLOSE: bool = True
PARTIAL_TRIGGER_PCT: float = 0.80   # 80% of TP distance
PARTIAL_CLOSE_PCT: float = 0.80     # close 80% of position size
