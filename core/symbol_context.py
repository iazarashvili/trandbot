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
    use_sweep_filter: bool = False
    use_asian_range: bool = False
    use_premium_discount: bool = False
    # NOT WIRED: use_structure_shift, use_breaker_blocks, use_po3 and use_ifvg
    # are loaded below but no executing path reads them.  The detectors exist in
    # strategy.py and are exercised only by reports/test_ict_filters.py, so
    # setting the corresponding flag in a symbol config changes nothing live.
    use_structure_shift: bool = False
    use_breaker_blocks: bool = False
    use_po3: bool = False
    use_ifvg: bool = False
    use_breakeven: bool = False
    breakeven_r: float = 2.0
    use_liquidity_tp: bool = True
    use_partial_close: bool = True
    partial_trigger_pct: float = 0.80
    partial_close_pct: float = 0.80
    max_spread_points: int = 0
    # Session/strategy filters (from backtest analysis 2026-09-16)
    blocked_sb_hours: list = field(default_factory=list)  # UTC hours where Silver Bullet is skipped
    blocked_amd_hours: list = field(default_factory=list)  # UTC hours where AMD is skipped
    blocked_trends: list = field(default_factory=list)     # trend directions to skip ("BEARISH", "BULLISH")
    max_ob_risk_px: float = 0.0                            # max OB stop width in price; 0=disabled
    # Phase 1: risk
    max_consecutive_losses: int = 5
    max_daily_loss_pct: float = 4.0
    daily_loss_buffer_pct: float = 0.5   # stop at 3.5% if limit is 4%
    max_trades_per_day: int = 1
    cooldown_candles: int = 3
    account_type: str = "funded"  # "funded"=0.4%, "challenge"=0.6%, "custom"=use RISK_PERCENT
    max_weekly_profit_pct: float = 2.0
    max_monthly_profit_pct: float = 6.0
    # SMT divergence
    smt_correlated_symbol: str = ""  # e.g. "ETHUSD" for BTCUSD, "EURUSD" for GBPUSD
    require_smt: bool = True         # gate trades on SMT confirmation
    # News filter
    trade_only_news_days: list = field(default_factory=list)  # weekdays that require news (0=Mon,1=Tue)
    require_high_impact_news: bool = False  # only trade during high impact news windows
    # Liquidity resistance
    use_liquidity_resistance: bool = True
    low_resistance_risk_factor: float = 1.0    # full risk on low resistance targets
    high_resistance_risk_factor: float = 0.25  # 1/4 risk on high resistance targets

    @classmethod
    def load(cls, symbol: str) -> "SymbolConfig":
        """Load from config_<symbol>.py, falling back to config.py defaults."""
        try:
            mod = importlib.import_module(f"config_{symbol.lower()}")
        except ModuleNotFoundError:
            mod = importlib.import_module("config")

        cfg = cls(
            symbol=symbol,
            enabled=True,
            lot_size=getattr(mod, "LOT_SIZE", 0.01),
            magic_number=getattr(mod, "MAGIC_NUMBER", 100200),
            risk_percent=getattr(mod, "RISK_PERCENT", 0.4),
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
            use_sweep_filter=getattr(mod, "USE_SWEEP_FILTER", False),
            use_asian_range=getattr(mod, "USE_ASIAN_RANGE", False),
            use_premium_discount=getattr(mod, "USE_PREMIUM_DISCOUNT", False),
            use_structure_shift=getattr(mod, "USE_STRUCTURE_SHIFT", False),
            use_breaker_blocks=getattr(mod, "USE_BREAKER_BLOCKS", False),
            use_po3=getattr(mod, "USE_PO3", False),
            use_ifvg=getattr(mod, "USE_IFVG", False),
            use_breakeven=getattr(mod, "USE_BREAKEVEN", False),
            breakeven_r=getattr(mod, "BREAKEVEN_R", 2.0),
            use_liquidity_tp=getattr(mod, "USE_LIQUIDITY_TP", True),
            use_partial_close=getattr(mod, "USE_PARTIAL_CLOSE", True),
            partial_trigger_pct=getattr(mod, "PARTIAL_TRIGGER_PCT", 0.80),
            partial_close_pct=getattr(mod, "PARTIAL_CLOSE_PCT", 0.80),
            max_spread_points=getattr(mod, "MAX_SPREAD_POINTS", 0),
            blocked_sb_hours=list(getattr(mod, "BLOCKED_SB_HOURS", [])),
            blocked_amd_hours=list(getattr(mod, "BLOCKED_AMD_HOURS", [])),
            blocked_trends=list(getattr(mod, "BLOCKED_TRENDS", [])),
            max_ob_risk_px=getattr(mod, "MAX_OB_RISK_PX", 0.0),
            max_consecutive_losses=getattr(mod, "MAX_CONSECUTIVE_LOSSES", 3),
            max_daily_loss_pct=getattr(mod, "MAX_DAILY_LOSS_PCT", 4.0),
            daily_loss_buffer_pct=getattr(mod, "DAILY_LOSS_BUFFER_PCT", 0.5),
            max_trades_per_day=getattr(mod, "MAX_TRADES_PER_DAY", 1),
            cooldown_candles=getattr(mod, "COOLDOWN_CANDLES", 3),
            account_type=getattr(mod, "ACCOUNT_TYPE", "funded"),
            max_weekly_profit_pct=getattr(mod, "MAX_WEEKLY_PROFIT_PCT", 2.0),
            max_monthly_profit_pct=getattr(mod, "MAX_MONTHLY_PROFIT_PCT", 6.0),
            smt_correlated_symbol=getattr(mod, "SMT_CORRELATED_SYMBOL", ""),
            require_smt=getattr(mod, "REQUIRE_SMT", True),
            trade_only_news_days=list(getattr(mod, "TRADE_ONLY_NEWS_DAYS", [])),
            require_high_impact_news=getattr(mod, "REQUIRE_HIGH_IMPACT_NEWS", False),
            use_liquidity_resistance=getattr(mod, "USE_LIQUIDITY_RESISTANCE", True),
            low_resistance_risk_factor=getattr(mod, "LOW_RESISTANCE_RISK_FACTOR", 1.0),
            high_resistance_risk_factor=getattr(mod, "HIGH_RESISTANCE_RISK_FACTOR", 0.25),
        )

        # Resolve account type -> risk_percent
        _ACCOUNT_RISK = {"funded": 0.4, "challenge": 0.6}
        if cfg.account_type in _ACCOUNT_RISK:
            cfg.risk_percent = _ACCOUNT_RISK[cfg.account_type]

        return cfg


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

    # Weekly/monthly profit tracking
    weekly_pnl: float = 0.0
    weekly_reset_key: Optional[str] = None   # "2026-W42"
    monthly_pnl: float = 0.0
    monthly_reset_key: Optional[str] = None  # "2026-10"

    # Position tracking — used to detect when a position closes (SL/TP/manual)
    last_position_ticket: Optional[int] = None

    # Last DOL target (for Telegram status display)
    _last_dol: Optional[float] = None

    # Rejection stats
    rejections: Dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: str):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def reset_daily(self, date_str: str):
        if self.daily_reset_date != date_str:
            self.daily_reset_date = date_str
            self.trades_today = 0
            self.daily_pnl = 0.0
            # Reset consecutive losses daily to prevent permanent deadlock:
            # once the limit is hit, no trade can fire to produce a win,
            # so the counter would never clear without this reset.
            self.consecutive_losses = 0

    def reset_periodic(self, now):
        """Reset weekly/monthly P&L counters when the period rolls over."""
        week_key = now.strftime("%G-W%V")
        if self.weekly_reset_key != week_key:
            self.weekly_reset_key = week_key
            self.weekly_pnl = 0.0

        month_key = now.strftime("%Y-%m")
        if self.monthly_reset_key != month_key:
            self.monthly_reset_key = month_key
            self.monthly_pnl = 0.0

    def record_trade_result(self, pnl: float):
        self.trades_today += 1
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
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

        # News filter: certain days (Mon/Tue) require high-impact news
        if self.cfg.trade_only_news_days and now.weekday() in self.cfg.trade_only_news_days:
            from news_calendar import get_calendar
            cal = get_calendar()
            if not cal.has_high_impact_today(self.symbol, now):
                return "NO_HIGH_IMPACT_NEWS"

        # Require proximity to high-impact news (optional)
        if self.cfg.require_high_impact_news:
            from news_calendar import get_calendar
            cal = get_calendar()
            if not cal.is_near_high_impact(self.symbol, now, window_minutes=60):
                return "NOT_NEAR_NEWS"

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

        # Cooldown after a trade
        if self.state.cooldown_until is not None:
            if now < self.state.cooldown_until:
                return "COOLDOWN"
            self.state.cooldown_until = None  # expired, clear it

        # Weekly/Monthly profit caps
        if self.cfg.max_weekly_profit_pct > 0 or self.cfg.max_monthly_profit_pct > 0:
            acct = mt5.account_info()
            if acct and acct.balance > 0:
                if self.cfg.max_weekly_profit_pct > 0:
                    weekly_pct = self.state.weekly_pnl / acct.balance * 100
                    if weekly_pct >= self.cfg.max_weekly_profit_pct:
                        return "WEEKLY_PROFIT_CAP"
                if self.cfg.max_monthly_profit_pct > 0:
                    monthly_pct = self.state.monthly_pnl / acct.balance * 100
                    if monthly_pct >= self.cfg.max_monthly_profit_pct:
                        return "MONTHLY_PROFIT_CAP"

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
