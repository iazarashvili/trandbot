"""Telegram dashboard — notifications, controls, and account monitoring.

/menu — main dashboard with buttons
/status — live positions + balance
/close BTCUSD — close specific symbol
/closeall — close all positions
/balance — account info
/trades — recent trade history
/help — command list
"""

import json
import logging
import os
import ssl
import threading
import time
from typing import Optional
from urllib.request import urlopen, Request

logger = logging.getLogger("smc_bot")

# SSL fix for some Windows builds
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# Load .env
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env):
    for _line in open(_env).readlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
API = f"https://api.telegram.org/bot{TOKEN}"


# ---------------------------------------------------------------- API
def _call(method: str, data: dict = None) -> Optional[dict]:
    try:
        url = f"{API}/{method}"
        if data:
            body = json.dumps(data).encode("utf-8")
            req = Request(url, data=body,
                          headers={"Content-Type": "application/json"})
        else:
            req = Request(url)
        with urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.debug("Telegram API error: %s", e)
        return None


def send_message(text: str, reply_markup: dict = None) -> bool:
    if not TOKEN or not CHAT_ID:
        return False
    data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    if reply_markup:
        data["reply_markup"] = reply_markup
    result = _call("sendMessage", data)
    return result is not None and result.get("ok", False)


def test_connection() -> bool:
    if not TOKEN:
        return False
    result = _call("getMe")
    if result and result.get("ok"):
        logger.info("Telegram connected: @%s", result["result"]["username"])
        return True
    return False


# ---------------------------------------------------------------- Menu
def send_menu():
    """Main dashboard menu with buttons."""
    text = "🤖 <b>What would you like?</b>"
    markup = {"inline_keyboard": [
        [{"text": "📊 Dashboard", "callback_data": "dashboard"}],
        [{"text": "🔧 Manage", "callback_data": "manage"},
         {"text": "📋 Trades", "callback_data": "trades"}],
        [{"text": "📈 Markets", "callback_data": "markets"},
         {"text": "📊 Info", "callback_data": "info"}],
        [{"text": "⚙️ Account", "callback_data": "account"},
         {"text": "📜 History", "callback_data": "history"}],
        [{"text": "❓ Help", "callback_data": "help"}],
    ]}
    send_message(text, reply_markup=markup)


def send_dashboard(balance, equity, positions, today_pnl=0, today_trades=0,
                   today_wins=0, today_losses=0, last_trade=""):
    """Full dashboard view."""
    hwm = max(balance, equity)
    dd_pct = (hwm - equity) / hwm * 100 if hwm > 0 else 0
    floating = equity - balance

    pos_count = len(positions) if positions else 0
    floating_pnl = sum(p.profit for p in positions) if positions else 0

    text = (
        f"📊 <b>DASHBOARD</b>\n\n"
        f"  Balance:  <b>${balance:,.2f}</b>\n"
        f"  Equity:   <b>${equity:,.2f}</b> ({floating:+.2f})\n"
        f"  HWM:      ${hwm:,.2f} (DD {dd_pct:.2f}%)\n"
        f"  Today:    ${today_pnl:+.2f} ({today_wins}W/{today_losses}L)\n"
        f"  Open:     {pos_count} positions  floating ${floating_pnl:+.2f}\n"
    )
    if last_trade:
        text += f"  Last:     {last_trade}\n"

    buttons = []
    if positions:
        buttons.append([
            {"text": "🔧 Manage positions", "callback_data": "manage"},
            {"text": "📋 Trades", "callback_data": "trades"},
        ])
        buttons.append([
            {"text": "🔴 Close ALL", "callback_data": "closeall"},
            {"text": "🔄 Refresh", "callback_data": "dashboard"},
        ])
    else:
        buttons.append([
            {"text": "📋 Trades", "callback_data": "trades"},
            {"text": "🔄 Refresh", "callback_data": "dashboard"},
        ])
    buttons.append([{"text": "🏠 Home", "callback_data": "menu"}])

    send_message(text, reply_markup={"inline_keyboard": buttons})


def send_manage(positions):
    """Position management view with close buttons per symbol."""
    if not positions:
        text = "🔧 <b>MANAGE POSITIONS</b>\n\nNo open positions."
        markup = {"inline_keyboard": [
            [{"text": "🔄 Refresh", "callback_data": "manage"}],
            [{"text": "🏠 Home", "callback_data": "menu"}],
        ]}
        send_message(text, reply_markup=markup)
        return

    lines = []
    buttons = []
    for p in positions:
        emoji = "🟢" if p.type == 0 else "🔴"
        direction = "BUY" if p.type == 0 else "SELL"
        lines.append(
            f"{emoji} <b>{p.symbol}</b> {direction} {p.volume} lots\n"
            f"    Entry: {p.price_open:.5f} | P&L: ${p.profit:+.2f}"
        )
        buttons.append([{
            "text": f"❌ Close {p.symbol} (${p.profit:+.2f})",
            "callback_data": f"close_{p.symbol}"
        }])

    text = f"🔧 <b>MANAGE POSITIONS</b>\n\n" + "\n\n".join(lines)
    buttons.append([{"text": "🔴 Close ALL", "callback_data": "closeall"}])
    buttons.append([
        {"text": "🔄 Refresh", "callback_data": "manage"},
        {"text": "🏠 Home", "callback_data": "menu"},
    ])
    send_message(text, reply_markup={"inline_keyboard": buttons})


def send_account(balance, equity, margin, free_margin, margin_level):
    """Account details view."""
    text = (
        f"⚙️ <b>ACCOUNT</b>\n\n"
        f"  Balance:      <b>${balance:,.2f}</b>\n"
        f"  Equity:       <b>${equity:,.2f}</b>\n"
        f"  Margin:       ${margin:,.2f}\n"
        f"  Free Margin:  ${free_margin:,.2f}\n"
        f"  Margin Level: {margin_level:.1f}%\n"
    )
    markup = {"inline_keyboard": [
        [{"text": "📊 Dashboard", "callback_data": "dashboard"}],
        [{"text": "🏠 Home", "callback_data": "menu"}],
    ]}
    send_message(text, reply_markup=markup)


def send_help():
    text = (
        "❓ <b>COMMANDS</b>\n\n"
        "/menu — Main dashboard\n"
        "/status — Positions & balance\n"
        "/balance — Account info\n"
        "/trades — Recent trade history\n"
        "/close BTCUSD — Close symbol\n"
        "/closeall — Close all positions\n"
        "/help — This message\n\n"
        "Or use the buttons below each message."
    )
    markup = {"inline_keyboard": [
        [{"text": "🏠 Home", "callback_data": "menu"}],
    ]}
    send_message(text, reply_markup=markup)


# ---------------------------------------------------------------- Notifications
def notify_trade_opened(symbol: str, direction: str, lots: float,
                        entry: float, sl: float, tp: float,
                        risk_usd: float, balance: float):
    emoji = "🟢" if direction == "BUY" else "🔴"
    risk_pct = risk_usd / balance * 100 if balance > 0 else 0
    risk_px = abs(entry - sl)
    reward_px = abs(tp - entry)
    rrr = round(reward_px / risk_px, 1) if risk_px > 0 else 0
    text = (
        f"{emoji} <b>TRADE ENTERED — LIVE</b>\n\n"
        f"<b>{direction} {symbol}</b>\n"
        f"Entry: {entry:.5f}\n"
        f"SL:    {sl:.5f}\n"
        f"TP:    {tp:.5f}\n\n"
        f"RRR: 1:{rrr}\n"
        f"Lots: {lots:.2f} | Risk: ${risk_usd:.2f} ({risk_pct:.1f}%)\n"
        f"Balance: ${balance:,.2f}"
    )
    markup = {"inline_keyboard": [
        [{"text": f"❌ Close {symbol}", "callback_data": f"close_{symbol}"}],
        [{"text": "📊 Dashboard", "callback_data": "dashboard"}],
    ]}
    send_message(text, reply_markup=markup)


def notify_trade_closed(symbol: str, direction: str, pnl: float,
                        exit_reason: str, balance: float, r_multiple: float = 0):
    if pnl > 0:
        emoji = "🟢"
        result = "WIN"
    elif pnl == 0:
        emoji = "⚪"
        result = "SCRATCH"
    else:
        emoji = "🔴"
        result = "LOSS"

    text = (
        f"{emoji} <b>{symbol} {direction} ({exit_reason})</b> | "
        f"${pnl:+.2f} ({r_multiple:+.2f}R)\n"
        f"Balance: ${balance:,.2f}"
    )
    send_message(text)


def notify_partial_close(symbol: str, closed_lots: float,
                         pnl: float, new_tp: float):
    text = (
        f"📊 <b>PARTIAL CLOSE</b>\n\n"
        f"Symbol: <b>{symbol}</b>\n"
        f"Closed: {closed_lots:.2f} lots\n"
        f"Locked P&L: ${pnl:+.2f}\n"
        f"Runner TP -> {new_tp:.5f}\n"
        f"SL -> Entry (breakeven)"
    )
    send_message(text)


def notify_waiting_for_trade(symbol: str, poi_type: str, zone_top: float,
                             zone_bottom: float, trend: str, filters: str = ""):
    emoji = "🟢" if poi_type == "BULLISH" else "🔴"
    direction = "BUY" if poi_type == "BULLISH" else "SELL"
    text = (
        f"👀 <b>WATCHING — {symbol}</b>\n\n"
        f"POI: {emoji} {poi_type}\n"
        f"Zone: {zone_top:.5f} — {zone_bottom:.5f}\n"
        f"Trend: {trend}\n"
        f"Looking for: <b>{direction}</b> setup\n"
    )
    if filters:
        text += f"Filters: {filters}\n"
    text += "\nWaiting for confirmation..."
    send_message(text)


def notify_signal_rejected(symbol: str, direction: str, reason: str):
    text = (
        f"🚫 <b>{symbol} {direction} REJECTED</b>\n"
        f"Reason: {reason}"
    )
    send_message(text)


def notify_error(message: str):
    send_message(f"⚠️ <b>ERROR</b>\n\n{message}")


def send_daily_summary(trades_today: int, wins: int, losses: int,
                       pnl: float, balance: float):
    emoji = "🟢" if pnl >= 0 else "🔴"
    good = "✅ Good day" if pnl >= 0 else "❌ Bad day"
    text = (
        f"{emoji} <b>DAILY SUMMARY</b>\n\n"
        f"P&L:     ${pnl:+.2f}\n"
        f"Trades:  {trades_today} ({wins}W/{losses}L)\n\n"
        f"Balance: ${balance:,.2f}\n\n"
        f"{good}"
    )
    send_message(text)


# ---------------------------------------------------------------- Command Listener
class TelegramCommandListener:
    """Polls for commands and button presses."""

    def __init__(self, on_close_symbol=None, on_close_all=None):
        self.on_close_symbol = on_close_symbol
        self.on_close_all = on_close_all
        self._offset = 0
        self._running = False
        self._thread = None

    def start(self):
        if not TOKEN or not CHAT_ID:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Telegram command listener started.")

    def stop(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            try:
                self._poll()
            except Exception:
                pass
            time.sleep(2)

    def _poll(self):
        result = _call("getUpdates", {"offset": self._offset, "timeout": 5})
        if not result or not result.get("ok"):
            return

        for update in result.get("result", []):
            self._offset = update["update_id"] + 1

            # Callback queries (button presses)
            if "callback_query" in update:
                cb = update["callback_query"]
                cb_data = cb.get("data", "")
                _call("answerCallbackQuery",
                      {"callback_query_id": cb["id"]})
                self._handle_callback(cb_data)

            # Text commands
            if "message" in update:
                text = update["message"].get("text", "").strip()
                chat_id = str(update["message"]["chat"]["id"])
                if chat_id != CHAT_ID:
                    continue
                self._handle_command(text)

    def _handle_callback(self, data: str):
        if data == "menu":
            send_menu()
        elif data == "dashboard":
            self._send_dashboard()
        elif data == "manage":
            self._send_manage()
        elif data == "account":
            self._send_account()
        elif data == "help":
            send_help()
        elif data == "trades" or data == "history":
            self._send_trades()
        elif data == "markets" or data == "info":
            self._send_markets()
        elif data == "closeall" and self.on_close_all:
            self.on_close_all()
            send_message("🔴 Closing all positions...")
        elif data.startswith("close_") and self.on_close_symbol:
            symbol = data[6:]
            self.on_close_symbol(symbol)
            send_message(f"🔴 Closing {symbol}...")

    def _handle_command(self, text: str):
        if text == "/menu" or text == "/start":
            send_menu()
        elif text == "/status" or text == "/dashboard":
            self._send_dashboard()
        elif text == "/balance" or text == "/account":
            self._send_account()
        elif text == "/trades" or text == "/history":
            self._send_trades()
        elif text == "/help":
            send_help()
        elif text == "/closeall" and self.on_close_all:
            self.on_close_all()
            send_message("🔴 Closing all positions...")
        elif text.startswith("/close ") and self.on_close_symbol:
            symbol = text[7:].strip().upper()
            self.on_close_symbol(symbol)
            send_message(f"🔴 Closing {symbol}...")

    def _send_dashboard(self):
        try:
            import MetaTrader5 as mt5
            account = mt5.account_info()
            if not account:
                send_message("⚠️ Cannot read account.")
                return
            positions = mt5.positions_get()
            send_dashboard(
                account.balance, account.equity,
                list(positions) if positions else [])
        except Exception as e:
            send_message(f"⚠️ {e}")

    def _send_manage(self):
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            send_manage(list(positions) if positions else [])
        except Exception as e:
            send_message(f"⚠️ {e}")

    def _send_account(self):
        try:
            import MetaTrader5 as mt5
            a = mt5.account_info()
            if not a:
                send_message("⚠️ Cannot read account.")
                return
            send_account(a.balance, a.equity, a.margin,
                         a.margin_free, a.margin_level or 0)
        except Exception as e:
            send_message(f"⚠️ {e}")

    def _send_trades(self):
        try:
            import MetaTrader5 as mt5
            from datetime import datetime, timedelta
            now = datetime.now()
            deals = mt5.history_deals_get(
                now - timedelta(days=7), now)
            if not deals or len(deals) == 0:
                send_message("📋 No trades in last 7 days.")
                return

            # Filter only closes with P&L
            closes = [d for d in deals if d.entry == 1 and d.profit != 0]
            if not closes:
                send_message("📋 No closed trades in last 7 days.")
                return

            lines = []
            for d in closes[-10:]:  # last 10
                emoji = "🟢" if d.profit > 0 else "🔴"
                dt = datetime.fromtimestamp(d.time)
                lines.append(
                    f"{emoji} {d.symbol} | ${d.profit:+.2f} | "
                    f"{dt.strftime('%m-%d %H:%M')}"
                )

            text = f"📋 <b>LAST TRADES</b>\n\n" + "\n".join(lines)
            markup = {"inline_keyboard": [
                [{"text": "🔄 Refresh", "callback_data": "trades"}],
                [{"text": "🏠 Home", "callback_data": "menu"}],
            ]}
            send_message(text, reply_markup=markup)
        except Exception as e:
            send_message(f"⚠️ {e}")

    def _send_markets(self):
        try:
            import MetaTrader5 as mt5
            symbols = ["BTCUSD", "XAUUSD", "GBPUSD", "EURUSD"]
            lines = []
            for sym in symbols:
                tick = mt5.symbol_info_tick(sym)
                if tick:
                    spread = abs(tick.ask - tick.bid)
                    info = mt5.symbol_info(sym)
                    spread_pts = spread / info.point if info else 0
                    lines.append(
                        f"<b>{sym}</b>\n"
                        f"  Bid: {tick.bid:.5f} | Ask: {tick.ask:.5f}\n"
                        f"  Spread: {spread_pts:.0f} pts"
                    )
            text = f"📈 <b>MARKETS</b>\n\n" + "\n\n".join(lines)
            markup = {"inline_keyboard": [
                [{"text": "🔄 Refresh", "callback_data": "markets"}],
                [{"text": "🏠 Home", "callback_data": "menu"}],
            ]}
            send_message(text, reply_markup=markup)
        except Exception as e:
            send_message(f"⚠️ {e}")
