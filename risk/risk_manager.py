"""Portfolio-level and per-symbol risk management.

Handles:
  - %-based position sizing (dynamic lot calculation)
  - Portfolio total open risk limit
  - Per-symbol risk validation
  - Spread validation
"""

import logging
from decimal import Decimal, ROUND_DOWN
from typing import Optional

import MetaTrader5 as mt5

from core.symbol_context import SymbolContext

logger = logging.getLogger("smc_bot")

# Global portfolio limit — total open risk across ALL symbols.
MAX_TOTAL_OPEN_RISK_PCT = 3.0  # % of account balance


def calculate_lot_size(
    ctx: SymbolContext,
    entry_price: float,
    sl_price: float,
) -> Optional[float]:
    """Calculate position size from risk percentage and SL distance.

    Returns the lot size rounded DOWN to the broker's volume step,
    or None if the trade should be skipped.
    """
    account = mt5.account_info()
    if account is None:
        logger.error("[%s] Cannot read account info for sizing.", ctx.symbol)
        return None

    spec = ctx.spec
    risk_pct = ctx.cfg.risk_percent
    equity = account.equity

    # Risk amount in dollars
    risk_amount = equity * risk_pct / 100.0

    # SL distance in price
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        return None

    # P&L per lot per price point = contract_size
    # For BTCUSD (contract=1): 1 lot, $1 move = $1
    # For XAUUSD (contract=100): 1 lot, $1 move = $100
    # For GBPUSD (contract=100000): 1 lot, 0.0001 move = $10
    pnl_per_lot = sl_distance * spec.contract_size

    if pnl_per_lot <= 0:
        return None

    raw_lots = risk_amount / pnl_per_lot

    # Round DOWN to volume step
    step = Decimal(str(spec.volume_step))
    lots = float(Decimal(str(raw_lots)).quantize(step, rounding=ROUND_DOWN))

    # Clamp to broker limits
    if lots < spec.volume_min:
        # Can't trade smaller than minimum — check if min lot risk is acceptable
        min_risk = spec.volume_min * pnl_per_lot
        if min_risk > risk_amount * 1.5:  # allow 50% over if forced to min lot
            logger.warning(
                "[%s] Min lot %.2f risks $%.2f, exceeds $%.2f — skip.",
                ctx.symbol, spec.volume_min, min_risk, risk_amount)
            return None
        lots = spec.volume_min

    lots = min(lots, spec.volume_max)

    actual_risk = lots * pnl_per_lot
    logger.info(
        "[%s] Sizing: %.1f%% of $%.0f = $%.2f risk | SL dist=%.5f | "
        "lots=%.2f | actual risk=$%.2f",
        ctx.symbol, risk_pct, equity, risk_amount,
        sl_distance, lots, actual_risk)

    return lots


def check_max_risk_usd(ctx: SymbolContext, lots: float,
                       entry: float, sl: float) -> Optional[str]:
    """Check hard dollar risk cap. Returns rejection reason or None."""
    sl_distance = abs(entry - sl)
    risk_usd = lots * sl_distance * ctx.spec.contract_size

    if ctx.cfg.max_risk_pct > 0:
        account = mt5.account_info()
        if account and account.balance > 0:
            max_allowed = account.balance * ctx.cfg.max_risk_pct / 100
            if risk_usd > max_allowed:
                return f"RISK_PCT_EXCEEDED (${risk_usd:.2f} > {ctx.cfg.max_risk_pct}% = ${max_allowed:.2f})"

    if ctx.cfg.max_risk_usd > 0 and risk_usd > ctx.cfg.max_risk_usd:
        return f"RISK_USD_EXCEEDED (${risk_usd:.2f} > ${ctx.cfg.max_risk_usd:.0f})"

    return None


def check_portfolio_risk() -> Optional[str]:
    """Check total open risk across all bot-managed positions.

    Returns rejection reason or None.
    """
    account = mt5.account_info()
    if account is None:
        return "NO_ACCOUNT_INFO"

    positions = mt5.positions_get()
    if positions is None:
        return None  # no positions = no risk

    total_risk = 0.0
    for pos in positions:
        if not pos.sl:
            continue
        sl_dist = abs(pos.price_open - pos.sl)
        risk = sl_dist * pos.volume * mt5.symbol_info(pos.symbol).trade_contract_size
        total_risk += risk

    risk_pct = total_risk / account.balance * 100 if account.balance > 0 else 0

    if risk_pct >= MAX_TOTAL_OPEN_RISK_PCT:
        return f"PORTFOLIO_RISK ({risk_pct:.1f}% >= {MAX_TOTAL_OPEN_RISK_PCT}%)"

    return None


def check_spread(ctx: SymbolContext) -> Optional[str]:
    """Check if current spread is acceptable. Returns rejection reason or None."""
    if ctx.cfg.max_spread_points <= 0:
        return None

    tick = mt5.symbol_info_tick(ctx.symbol)
    if tick is None:
        return "NO_TICK_DATA"

    spread = abs(tick.ask - tick.bid) / ctx.spec.point
    if spread > ctx.cfg.max_spread_points:
        return f"SPREAD_TOO_HIGH ({spread:.0f} > {ctx.cfg.max_spread_points})"

    return None


def validate_entry(ctx: SymbolContext, direction: str,
                   entry: float, sl: float, tp: float) -> Optional[str]:
    """Run all pre-trade risk checks. Returns rejection reason or None."""
    # 1. Time/session filters
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    ctx.state.reset_daily(now.strftime("%Y-%m-%d"))
    time_block = ctx.is_trading_allowed(now)
    if time_block:
        return time_block

    # 2. Spread
    spread_block = check_spread(ctx)
    if spread_block:
        return spread_block

    # 3. Portfolio risk
    port_block = check_portfolio_risk()
    if port_block:
        return port_block

    # 4. Calculate lot size
    lots = calculate_lot_size(ctx, entry, sl)
    if lots is None:
        return "SIZING_FAILED"

    # 5. Per-symbol risk cap
    risk_block = check_max_risk_usd(ctx, lots, entry, sl)
    if risk_block:
        return risk_block

    # 6. RR check
    risk_dist = abs(entry - sl)
    reward_dist = abs(tp - entry)
    rr = reward_dist / risk_dist if risk_dist > 0 else 0
    if rr < ctx.cfg.min_rr_liquidity:
        return f"RR_TOO_LOW ({rr:.2f} < {ctx.cfg.min_rr_liquidity})"

    return None  # all checks passed
