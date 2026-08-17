"""MetaTrader 5 Integration Module.

All broker-specific pedantry lives here — digit rounding, volume steps,
minimum stop distance, filling modes and retcode handling — so that the
strategy and the main loop can keep talking in plain prices.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional, Sequence

import MetaTrader5 as mt5
import pandas as pd

from config import DEVIATION, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

logger = logging.getLogger(__name__)

# Retcodes are resolved defensively: the numeric values are part of the MT5
# protocol and stable, but the constant names have moved around between
# releases of the Python package.
RETCODE_DONE = getattr(mt5, "TRADE_RETCODE_DONE", 10009)
RETCODE_DONE_PARTIAL = getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)
RETCODE_REQUOTE = getattr(mt5, "TRADE_RETCODE_REQUOTE", 10004)
RETCODE_INVALID_STOPS = getattr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016)
RETCODE_PRICE_CHANGED = getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020)
RETCODE_PRICE_OFF = getattr(mt5, "TRADE_RETCODE_PRICE_OFF", 10021)

# Worth one more attempt with a freshly quoted price.
_RETRYABLE = frozenset({RETCODE_REQUOTE, RETCODE_PRICE_CHANGED, RETCODE_PRICE_OFF})
_MAX_SEND_ATTEMPTS = 3


def _decimals(step: float) -> int:
    """Number of decimal places needed to express `step` exactly."""
    try:
        exponent = Decimal(str(step)).normalize().as_tuple().exponent
    except (InvalidOperation, ValueError):
        return 8
    return max(0, -int(exponent)) if isinstance(exponent, int) else 8


@dataclass(frozen=True)
class OrderLevels:
    """Entry/SL/TP already rounded and validated against broker constraints."""

    price: float
    sl: float
    tp: float


class MT5Connector:
    """Handles communication with the MetaTrader 5 terminal."""

    def __init__(self, symbol: str, magic_number: int):
        self.symbol = symbol
        self.magic_number = magic_number

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """Initializes the MT5 connection with explicit login credentials."""
        if not mt5.initialize(
            login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER
        ):
            logger.error("MT5 initialization failed: %s", mt5.last_error())
            return False

        if not mt5.symbol_select(self.symbol, True):
            logger.error("Failed to select symbol: %s", self.symbol)
            mt5.shutdown()
            return False

        info = self.symbol_info()
        if info is None:
            logger.error("Symbol %s has no info after selection.", self.symbol)
            mt5.shutdown()
            return False

        logger.info(
            "Connected to MT5. Target: %s | digits=%s point=%s "
            "volume=%s-%s step=%s stops_level=%s",
            self.symbol,
            info.digits,
            info.point,
            info.volume_min,
            info.volume_max,
            info.volume_step,
            getattr(info, "trade_stops_level", 0),
        )
        self.trading_enabled()  # surfaces the AutoTrading toggle at startup
        return True

    def is_connected(self) -> bool:
        """True while the terminal is reachable and the account is logged in."""
        return mt5.terminal_info() is not None and mt5.account_info() is not None

    def trading_enabled(self) -> bool:
        """True when the terminal and the account will actually accept orders.

        `order_check` does not test the terminal's AutoTrading toggle, so a bot
        can look perfectly healthy and still have every `order_send` bounce
        back with retcode 10027.
        """
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None or account is None:
            logger.error("Terminal or account info unavailable: %s", mt5.last_error())
            return False

        if not terminal.trade_allowed:
            logger.error(
                "🚫 AutoTrading is DISABLED in MetaTrader 5. Press the "
                "'Algo Trading' button in the toolbar (Ctrl+E) — until then "
                "every order is rejected with retcode 10027."
            )
            return False

        if not account.trade_expert:
            logger.error(
                "🚫 The broker has disabled expert/automated trading on "
                "account %s. Orders will be rejected.",
                account.login,
            )
            return False

        return True

    def reconnect(self) -> bool:
        """Tears the connection down and brings it back up."""
        logger.warning("MT5 connection lost — attempting to reconnect...")
        try:
            mt5.shutdown()
        except Exception:  # noqa: BLE001 - shutdown must never mask the retry
            logger.debug("shutdown() raised during reconnect", exc_info=True)
        return self.initialize()

    def shutdown(self) -> None:
        """Closes the MT5 connection."""
        mt5.shutdown()
        logger.info("MT5 connection closed.")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def symbol_info(self):
        """Returns the live SymbolInfo, or None if the terminal is unavailable."""
        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.warning(
                "symbol_info(%s) returned None: %s", self.symbol, mt5.last_error()
            )
        return info

    def get_tick(self):
        """Returns the latest tick, or None if no quote is available."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None or tick.ask <= 0 or tick.bid <= 0:
            logger.warning(
                "No valid tick for %s: %s", self.symbol, mt5.last_error()
            )
            return None
        return tick

    def get_spread_points(self) -> Optional[float]:
        """Current spread expressed in points."""
        tick = self.get_tick()
        info = self.symbol_info()
        if tick is None or info is None or info.point <= 0:
            return None
        return (tick.ask - tick.bid) / info.point

    def entry_price(self, order_type: int) -> Optional[float]:
        """The price a market order of `order_type` would execute at."""
        tick = self.get_tick()
        if tick is None:
            return None
        return tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    def fetch_rates(self, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """Fetches historical candlestick data as a Pandas DataFrame."""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(
                "Failed to fetch rates for timeframe %s: %s",
                timeframe,
                mt5.last_error(),
            )
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def normalize_price(self, price: float) -> float:
        """Rounds a price to the symbol's tick precision."""
        info = self.symbol_info()
        if info is None:
            return price
        return round(float(price), info.digits)

    def normalize_volume(self, volume: float) -> Optional[float]:
        """Snaps a volume to the broker's step and clamps it to min/max."""
        info = self.symbol_info()
        if info is None:
            return None

        step = info.volume_step or 0.01
        precision = _decimals(step)

        # Round *down* to the step so we never risk more than intended.
        snapped = int(float(volume) / step + 1e-9) * step
        snapped = round(snapped, precision)

        if snapped < info.volume_min:
            logger.warning(
                "Volume %.4f is below the broker minimum %.4f — using the "
                "minimum, which risks more than configured.",
                volume,
                info.volume_min,
            )
            snapped = info.volume_min
        if snapped > info.volume_max:
            logger.warning(
                "Volume %.4f exceeds the broker maximum %.4f — capping.",
                volume,
                info.volume_max,
            )
            snapped = info.volume_max

        return round(snapped, precision)

    def min_stop_distance(self) -> float:
        """Minimum allowed distance between price and SL/TP, in price units."""
        info = self.symbol_info()
        if info is None:
            return 0.0
        level = getattr(info, "trade_stops_level", 0) or 0
        return level * info.point

    def normalize_levels(
        self, order_type: int, price: float, sl: float, tp: float
    ) -> OrderLevels:
        """Rounds entry/SL/TP and pushes them outside the broker's stop level.

        Widening the stop loss raises the real risk of the trade, so size the
        position from the levels this returns — not from the raw strategy
        output.
        """
        min_dist = self.min_stop_distance()
        price = self.normalize_price(price)
        original_sl = sl

        if order_type == mt5.ORDER_TYPE_BUY:
            sl = min(sl, price - min_dist)
            tp = max(tp, price + min_dist)
        else:
            sl = max(sl, price + min_dist)
            tp = min(tp, price - min_dist)

        sl = self.normalize_price(sl)
        tp = self.normalize_price(tp)

        if min_dist > 0 and abs(sl - original_sl) > min_dist * 0.01:
            logger.warning(
                "SL widened from %.*f to %.*f to satisfy the broker's minimum "
                "stop distance (%.*f).",
                self._digits(), original_sl,
                self._digits(), sl,
                self._digits(), min_dist,
            )

        return OrderLevels(price=price, sl=sl, tp=tp)

    def _digits(self) -> int:
        info = self.symbol_info()
        return info.digits if info is not None else 2

    def _filling_mode(self) -> int:
        """Picks a filling mode the symbol actually supports.

        Hard-coding IOC is the single most common reason a bot's orders come
        back with retcode 10030 on a broker that only allows FOK.
        """
        info = self.symbol_info()
        allowed = getattr(info, "filling_mode", 0) if info is not None else 0

        if allowed & getattr(mt5, "SYMBOL_FILLING_IOC", 2):
            return mt5.ORDER_FILLING_IOC
        if allowed & getattr(mt5, "SYMBOL_FILLING_FOK", 1):
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_volume(
        self, order_type: int, entry: float, sl: float, risk_percent: float
    ) -> Optional[float]:
        """Volume that risks `risk_percent` of the balance between entry and SL."""
        account = mt5.account_info()
        if account is None:
            logger.error("account_info() unavailable: %s", mt5.last_error())
            return None
        if risk_percent <= 0:
            return None

        risk_amount = account.balance * risk_percent / 100.0
        loss_per_lot = mt5.order_calc_profit(order_type, self.symbol, 1.0, entry, sl)

        if loss_per_lot is None or loss_per_lot >= 0:
            logger.error(
                "order_calc_profit returned %s for %s %.2f->%.2f; cannot size "
                "the position.",
                loss_per_lot,
                self.symbol,
                entry,
                sl,
            )
            return None

        return self.normalize_volume(risk_amount / abs(loss_per_lot))

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def get_open_positions(self) -> Sequence:
        """Active positions on this symbol that belong to this bot."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return ()
        return tuple(p for p in positions if p.magic == self.magic_number)

    def get_open_positions_count(self) -> int:
        """Number of active positions managed by this bot."""
        return len(self.get_open_positions())

    def modify_position_sl(self, position, new_sl: float) -> bool:
        """Moves an open position's stop loss, keeping its take profit."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": position.ticket,
            "sl": self.normalize_price(new_sl),
            "tp": position.tp,
            "magic": self.magic_number,
        }
        result = mt5.order_send(request)
        if result is None:
            logger.error("SL modify returned None: %s", mt5.last_error())
            return False
        if result.retcode != RETCODE_DONE:
            logger.error(
                "SL modify failed on #%s! Retcode: %s, Comment: %s",
                position.ticket,
                result.retcode,
                result.comment,
            )
            return False

        logger.info("Position #%s stop loss moved to %s.", position.ticket, new_sl)
        return True

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def send_order(
        self,
        order_type: int,
        sl: float,
        tp: float,
        volume: float,
        comment: str = "SMC_Setup",
    ) -> bool:
        """Executes a market order at the live price with SL and TP.

        The price is re-quoted on every attempt — a market order sent at a
        stale candle close is what produces requotes and "invalid price"
        rejections.
        """
        if not self.trading_enabled():
            return False

        normalized_volume = self.normalize_volume(volume)
        if normalized_volume is None or normalized_volume <= 0:
            logger.error("Could not normalize volume %s — order aborted.", volume)
            return False

        for attempt in range(1, _MAX_SEND_ATTEMPTS + 1):
            price = self.entry_price(order_type)
            if price is None:
                logger.error("No live price available — order aborted.")
                return False

            levels = self.normalize_levels(order_type, price, sl, tp)
            result = self._deal(order_type, levels, normalized_volume, comment)

            if result is None:
                logger.error("order_send returned None: %s", mt5.last_error())
                return False

            if result.retcode in (RETCODE_DONE, RETCODE_DONE_PARTIAL):
                if result.retcode == RETCODE_DONE_PARTIAL:
                    logger.warning(
                        "Order only partially filled: %s of %s lots.",
                        result.volume,
                        normalized_volume,
                    )
                logger.info(
                    "Order executed: %s %s lots @ %s | SL: %s | TP: %s",
                    "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL",
                    normalized_volume,
                    result.price or levels.price,
                    levels.sl,
                    levels.tp,
                )
                return True

            if result.retcode == RETCODE_INVALID_STOPS:
                return self._send_then_attach_stops(
                    order_type, levels, normalized_volume, comment
                )

            if result.retcode in _RETRYABLE and attempt < _MAX_SEND_ATTEMPTS:
                logger.warning(
                    "Order rejected (retcode %s: %s) — retrying %s/%s with a "
                    "fresh quote.",
                    result.retcode,
                    result.comment,
                    attempt + 1,
                    _MAX_SEND_ATTEMPTS,
                )
                continue

            logger.error(
                "Order failed! Retcode: %s, Comment: %s",
                result.retcode,
                result.comment,
            )
            return False

        return False

    def _deal(
        self,
        order_type: int,
        levels: OrderLevels,
        volume: float,
        comment: str,
        with_stops: bool = True,
    ):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": levels.price,
            "deviation": DEVIATION,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }
        if with_stops:
            request["sl"] = levels.sl
            request["tp"] = levels.tp
        return mt5.order_send(request)

    def _send_then_attach_stops(
        self, order_type: int, levels: OrderLevels, volume: float, comment: str
    ) -> bool:
        """Fallback for brokers that reject SL/TP on the opening deal itself.

        Common on market-execution accounts: the entry has to go in naked and
        the stops are attached to the resulting position afterwards.
        """
        logger.warning(
            "Broker rejected the stops on the entry deal — sending the entry "
            "without stops and attaching them to the position."
        )
        result = self._deal(order_type, levels, volume, comment, with_stops=False)

        if result is None:
            logger.error("Naked entry returned None: %s", mt5.last_error())
            return False
        if result.retcode not in (RETCODE_DONE, RETCODE_DONE_PARTIAL):
            logger.error(
                "Naked entry failed! Retcode: %s, Comment: %s",
                result.retcode,
                result.comment,
            )
            return False

        position = self._find_position(getattr(result, "order", 0))
        if position is None:
            logger.critical(
                "Entry filled but the position could not be located — it is "
                "OPEN WITHOUT A STOP LOSS. Close or protect it manually."
            )
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": position.ticket,
            "sl": levels.sl,
            "tp": levels.tp,
            "magic": self.magic_number,
        }
        sltp_result = mt5.order_send(request)
        if sltp_result is None or sltp_result.retcode != RETCODE_DONE:
            retcode = getattr(sltp_result, "retcode", None)
            logger.critical(
                "Position #%s is OPEN WITHOUT A STOP LOSS (SLTP retcode: %s). "
                "Close or protect it manually.",
                position.ticket,
                retcode,
            )
            return False

        logger.info(
            "Order executed: %s lots | SL: %s | TP: %s (stops attached "
            "post-fill)",
            volume,
            levels.sl,
            levels.tp,
        )
        return True

    def _find_position(self, order_ticket: int):
        """Locates the position opened by `order_ticket`."""
        if order_ticket:
            found = mt5.positions_get(ticket=order_ticket)
            if found:
                return found[0]

        # Hedging accounts reuse the order ticket, netting accounts do not.
        # Fall back to the newest position carrying our magic number.
        ours = self.get_open_positions()
        return max(ours, key=lambda p: p.time, default=None)
