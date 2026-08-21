"""Independent per-symbol trading context.

Each symbol gets its own SymbolContext — market data, strategy state, config,
and trade history.  A BTCUSD setup never interferes with XAUUSD state.
"""

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd

logger = logging.getLogger("smc_bot")


@dataclass
class SymbolSpec:
    """Broker-specific contract information, read from MT5."""
    symbol: str
    digits: int
    point: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_stops_level: int
    spread: int

    @classmethod
    def from_mt5(cls, info) -> "SymbolSpec":
        return cls(
            symbol=info.name,
            digits=info.digits,
            point=info.point,
            contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            trade_stops_level=info.trade_stops_level,
            spread=info.spread,
        )


@dataclass
class SymbolConfig:
    """Per-symbol configuration.  Loaded from config_<symbol>.py or defaults."""
    symbol: str
    enabled: bool = True
    lot_size: float = 0.01
    magic_number: int = 100200
    risk_percent: float = 0.5
    stop_mode: str = "window"
    max_risk_usd: float = 80.0
    max_risk_pct: float = 0.0
    night_start: int = 0
    night_end: int = 0
    blocked_days: list = field(default_factory=list)
    skip_weekends: bool = True
    rrr: float = 3.0
    invert_signals: bool = False
    min_rr_liquidity: float = 0.5
    use_trend_filter: bool = True
    trend_ema_period: int = 100
    use_partial_close: bool = True
    partial_trigger_pct: float = 0.80
    partial_close_pct: float = 0.80
    max_spread_points: int = 0
    # Phase 1: risk
    max_consecutive_losses: int = 3
    max_daily_loss_pct: float = 4.0
    daily_loss_buffer_pct: float = 0.5   # stop at 3.5% if limit is 4%
    max_trades_per_day: int = 10
    cooldown_candles: int = 3

    @classmethod
    def load(cls, symbol: str) -> "SymbolConfig":
        """Load from config_<symbol>.py, falling back to config.py defaults."""
        try:
            mod = importlib.import_module(f"config_{symbol.lower()}")
        except ModuleNotFoundError:
            mod = importlib.import_module("config")

        return cls(
            symbol=symbol,
            enabled=True,
            lot_size=getattr(mod, "LOT_SIZE", 0.01),
            magic_number=getattr(mod, "MAGIC_NUMBER", 100200),
            risk_percent=getattr(mod, "RISK_PERCENT", 0.5),
            stop_mode=getattr(mod, "STOP_MODE", "window"),
            max_risk_usd=getattr(mod, "MAX_RISK_USD", 80.0),
            max_risk_pct=getattr(mod, "MAX_RISK_PCT", 0.0),
            night_start=getattr(mod, "NIGHT_START_HOUR", 0),
            night_end=getattr(mod, "NIGHT_END_HOUR", 0),
            blocked_days=list(getattr(mod, "BLOCKED_DAYS", [])),
            skip_weekends=getattr(mod, "SKIP_WEEKENDS", True),
            rrr=getattr(mod, "RRR", 3.0),
            invert_signals=getattr(mod, "INVERT_SIGNALS", False),
            min_rr_liquidity=getattr(mod, "MIN_RRR_LIQUIDITY", 0.5),
            use_trend_filter=getattr(mod, "USE_TREND_FILTER", True),
            trend_ema_period=getattr(mod, "TREND_EMA_PERIOD", 100),
            use_partial_close=getattr(mod, "USE_PARTIAL_CLOSE", True),
            partial_trigger_pct=getattr(mod, "PARTIAL_TRIGGER_PCT", 0.80),
            partial_close_pct=getattr(mod, "PARTIAL_CLOSE_PCT", 0.80),
            max_spread_points=getattr(mod, "MAX_SPREAD_POINTS", 0),
            max_consecutive_losses=getattr(mod, "MAX_CONSECUTIVE_LOSSES", 3),
            max_daily_loss_pct=getattr(mod, "MAX_DAILY_LOSS_PCT", 4.0),
            daily_loss_buffer_pct=getattr(mod, "DAILY_LOSS_BUFFER_PCT", 0.5),
            max_trades_per_day=getattr(mod, "MAX_TRADES_PER_DAY", 10),
            cooldown_candles=getattr(mod, "COOLDOWN_CANDLES", 3),
        )


@dataclass
class SymbolState:
    """Mutable per-symbol state — strategy, risk, and trade tracking."""
    # Strategy state
    poi: Any = None
    poi_updated_at: Optional[str] = None
    trend: str = "NEUTRAL"
    watch_key: Optional[tuple] = None
    watch_start: Optional[pd.Timestamp] = None
    abandoned: bool = False

    # Risk state
    consecutive_losses: int = 0
    trades_today: int = 0
    daily_pnl: float = 0.0
    daily_reset_date: Optional[str] = None
    cooldown_until: Optional[pd.Timestamp] = None
    paused: bool = False
    pause_reason: str = ""

    # Rejection stats
    rejections: Dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: str):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def reset_daily(self, date_str: str):
        if self.daily_reset_date != date_str:
            self.daily_reset_date = date_str
            self.trades_today = 0
            self.daily_pnl = 0.0

    def record_trade_result(self, pnl: float):
        self.trades_today += 1
        self.daily_pnl += pnl
        if pnl <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0


class SymbolContext:
    """Everything the engine needs to trade one symbol independently."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.cfg = SymbolConfig.load(symbol)
        self.spec: Optional[SymbolSpec] = None
        self.state = SymbolState()
        self.trade_history: List[dict] = []

    def initialize(self) -> bool:
        """Load broker specs from MT5.  Returns False if symbol is unavailable."""
        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.error("Symbol %s not found in MT5.", self.symbol)
            return False
        if not info.visible:
            mt5.symbol_select(self.symbol, True)
            info = mt5.symbol_info(self.symbol)
            if info is None:
                return False
        self.spec = SymbolSpec.from_mt5(info)
        logger.info(
            "Loaded %s: digits=%s point=%s contract=%s vol=%s-%s step=%s",
            self.symbol, self.spec.digits, self.spec.point,
            self.spec.contract_size, self.spec.volume_min,
            self.spec.volume_max, self.spec.volume_step,
        )
        return True

    def is_trading_allowed(self, now) -> Optional[str]:
        """Returns rejection reason if trading is blocked, None if allowed."""
        # Weekend
        if self.cfg.skip_weekends and now.weekday() >= 5:
            return "WEEKEND"

        # Blocked days
        if self.cfg.blocked_days and now.weekday() in self.cfg.blocked_days:
            return "BLOCKED_DAY"

        # Night filter
        ns, ne = self.cfg.night_start, self.cfg.night_end
        if ns != ne:
            h = now.hour
            if ns > ne:  # wraps midnight, e.g. 21-13
                if h >= ns or h < ne:
                    return "NIGHT_SESSION"
            elif ns <= h < ne:
                return "NIGHT_SESSION"

        # Daily loss
        effective_limit = self.cfg.max_daily_loss_pct - self.cfg.daily_loss_buffer_pct
        if effective_limit > 0:
            account = mt5.account_info()
            if account and account.balance > 0:
                daily_loss_pct = abs(min(0, self.state.daily_pnl)) / account.balance * 100
                if daily_loss_pct >= effective_limit:
                    return "DAILY_LOSS_LIMIT"

        # Consecutive losses
        if self.state.consecutive_losses >= self.cfg.max_consecutive_losses:
            return "CONSECUTIVE_LOSSES"

        # Daily trade limit
        if self.state.trades_today >= self.cfg.max_trades_per_day:
            return "MAX_DAILY_TRADES"

        # Paused
        if self.state.paused:
            return self.state.pause_reason or "PAUSED"

        return None

    def log_status(self, prefix: str = ""):
        """Log current state for debugging."""
        s = self.state
        logger.info(
            "%s[%s] trend=%s poi=%s consec_loss=%d trades_today=%d daily_pnl=%.2f",
            prefix, self.symbol, s.trend,
            "YES" if s.poi else "NO",
            s.consecutive_losses, s.trades_today, s.daily_pnl,
        )
