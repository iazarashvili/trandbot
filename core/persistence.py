"""SQLite persistence for bot state — survives restarts and crashes.

Stores: trades, daily P&L, symbol states, risk events.
On startup: reconcile with MT5 positions.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("smc_bot")

DB_PATH = Path(__file__).resolve().parent.parent / "bot_state.db"


class BotDatabase:
    """Lightweight SQLite store for bot state persistence."""

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("Database ready: %s", path.name)

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                ticket INTEGER,
                direction TEXT NOT NULL,
                strategy TEXT DEFAULT '',
                entry_price REAL,
                sl REAL,
                tp REAL,
                volume REAL,
                pnl REAL,
                r_multiple REAL,
                setup_score INTEGER,
                rejection_reason TEXT,
                entry_time TEXT,
                exit_time TEXT,
                exit_reason TEXT,
                smt_confirmed INTEGER DEFAULT 0,
                news_context TEXT DEFAULT '',
                liq_resistance TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trades_count INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                UNIQUE(date, symbol)
            );

            CREATE TABLE IF NOT EXISTS symbol_states (
                symbol TEXT PRIMARY KEY,
                consecutive_losses INTEGER DEFAULT 0,
                paused INTEGER DEFAULT 0,
                pause_reason TEXT DEFAULT '',
                last_trade_time TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add columns that may be missing from an older schema."""
        cursor = self.conn.execute("PRAGMA table_info(trades)")
        existing = {row[1] for row in cursor.fetchall()}
        new_cols = {
            "strategy": "TEXT DEFAULT ''",
            "smt_confirmed": "INTEGER DEFAULT 0",
            "news_context": "TEXT DEFAULT ''",
            "liq_resistance": "TEXT DEFAULT ''",
            "notes": "TEXT DEFAULT ''",
        }
        for col, typedef in new_cols.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typedef}")
        self.conn.commit()

    # ---- Trades ----
    def record_trade(self, symbol: str, ticket: int, direction: str,
                     entry: float, sl: float, tp: float, volume: float,
                     pnl: float = 0, r_multiple: float = 0,
                     setup_score: int = 0, entry_time: str = "",
                     exit_time: str = "", exit_reason: str = "",
                     strategy: str = "", smt_confirmed: bool = False,
                     news_context: str = "", liq_resistance: str = ""):
        self.conn.execute(
            "INSERT INTO trades (symbol, ticket, direction, strategy, entry_price, sl, tp, "
            "volume, pnl, r_multiple, setup_score, entry_time, exit_time, exit_reason, "
            "smt_confirmed, news_context, liq_resistance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, ticket, direction, strategy, entry, sl, tp, volume, pnl,
             r_multiple, setup_score, entry_time, exit_time, exit_reason,
             int(smt_confirmed), news_context, liq_resistance))
        self.conn.commit()

    def record_rejection(self, symbol: str, reason: str, details: str = ""):
        self.conn.execute(
            "INSERT INTO trades (symbol, direction, rejection_reason, entry_time) "
            "VALUES (?, 'REJECTED', ?, ?)",
            (symbol, reason, datetime.utcnow().isoformat()))
        self.conn.commit()

    # ---- Daily metrics ----
    def update_daily(self, date: str, symbol: str, pnl: float, is_win: bool):
        self.conn.execute("""
            INSERT INTO daily_metrics (date, symbol, trades_count, wins, losses, pnl)
            VALUES (?, ?, 1, ?, ?, ?)
            ON CONFLICT(date, symbol) DO UPDATE SET
                trades_count = trades_count + 1,
                wins = wins + ?,
                losses = losses + ?,
                pnl = pnl + ?
        """, (date, symbol, int(is_win), int(not is_win), pnl,
              int(is_win), int(not is_win), pnl))
        self.conn.commit()

    def get_daily_pnl(self, date: str, symbol: str) -> float:
        row = self.conn.execute(
            "SELECT pnl FROM daily_metrics WHERE date=? AND symbol=?",
            (date, symbol)).fetchone()
        return row["pnl"] if row else 0.0

    def get_daily_trades(self, date: str, symbol: str) -> int:
        row = self.conn.execute(
            "SELECT trades_count FROM daily_metrics WHERE date=? AND symbol=?",
            (date, symbol)).fetchone()
        return row["trades_count"] if row else 0

    # ---- Symbol state ----
    def save_symbol_state(self, symbol: str, consecutive_losses: int,
                          paused: bool, pause_reason: str):
        self.conn.execute("""
            INSERT INTO symbol_states (symbol, consecutive_losses, paused, pause_reason, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                consecutive_losses = ?,
                paused = ?,
                pause_reason = ?,
                updated_at = ?
        """, (symbol, consecutive_losses, int(paused), pause_reason,
              datetime.utcnow().isoformat(),
              consecutive_losses, int(paused), pause_reason,
              datetime.utcnow().isoformat()))
        self.conn.commit()

    def load_symbol_state(self, symbol: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM symbol_states WHERE symbol=?", (symbol,)).fetchone()
        if row:
            return dict(row)
        return None

    # ---- Risk events ----
    def record_risk_event(self, symbol: str, event_type: str, details: str = ""):
        self.conn.execute(
            "INSERT INTO risk_events (symbol, event_type, details) VALUES (?, ?, ?)",
            (symbol, event_type, details))
        self.conn.commit()

    # ---- Stats ----
    def get_symbol_stats(self, symbol: str) -> dict:
        trades = self.conn.execute(
            "SELECT * FROM trades WHERE symbol=? AND pnl IS NOT NULL AND pnl != 0",
            (symbol,)).fetchall()
        if not trades:
            return {"trades": 0}
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 1),
            "total_pnl": round(sum(t["pnl"] for t in trades), 2),
        }

    def close(self):
        self.conn.close()
