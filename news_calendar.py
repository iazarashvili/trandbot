"""Economic news calendar — fetches high-impact events from MQL5.

Used by the engine to:
  1. Block Mon/Tue trades unless a high-impact event is scheduled
  2. Optionally require all trades to be near high-impact news windows

The calendar is cached for 6 hours to avoid hammering the API.
Falls back gracefully: if the fetch fails, trading is NOT blocked.
"""

import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger("smc_bot")

CACHE_PATH = Path(__file__).resolve().parent / ".news_cache.json"
CACHE_TTL = 6 * 3600  # 6 hours

# Currencies we care about — map symbols to their base/quote currencies
SYMBOL_CURRENCIES = {
    "BTCUSD": ["USD"],
    "XAUUSD": ["USD"],
    "GBPUSD": ["GBP", "USD"],
    "EURUSD": ["EUR", "USD"],
}

# MQL5 economic calendar importance levels
# 0 = holiday, 1 = low, 2 = medium, 3 = high
HIGH_IMPACT = 3


@dataclass
class NewsEvent:
    """A single economic calendar event."""
    time_utc: datetime.datetime
    currency: str
    importance: int  # 1=low, 2=medium, 3=high
    title: str
    country: str = ""


class NewsCalendar:
    """Fetches and caches economic news events."""

    def __init__(self, cache_path: Path = CACHE_PATH):
        self._cache_path = cache_path
        self._events: List[NewsEvent] = []
        self._last_fetch: float = 0.0

    def refresh_if_needed(self):
        """Fetch fresh data if cache is stale."""
        now = time.time()
        if now - self._last_fetch < CACHE_TTL and self._events:
            return

        # Try loading from disk cache first
        if self._load_cache():
            return

        # Fetch from MQL5
        self._fetch_mql5()

    def _load_cache(self) -> bool:
        """Load events from disk cache if fresh enough."""
        if not self._cache_path.exists():
            return False
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            cache_time = data.get("fetched_at", 0)
            if time.time() - cache_time > CACHE_TTL:
                return False
            self._events = [
                NewsEvent(
                    time_utc=datetime.datetime.fromisoformat(e["time_utc"]),
                    currency=e["currency"],
                    importance=e["importance"],
                    title=e["title"],
                    country=e.get("country", ""),
                )
                for e in data.get("events", [])
            ]
            self._last_fetch = cache_time
            logger.info("News calendar loaded from cache: %d events.", len(self._events))
            return True
        except Exception:
            logger.warning("Failed to load news cache, will re-fetch.")
            return False

    def _fetch_mql5(self):
        """Fetch this week's calendar from MQL5 economic calendar."""
        today = datetime.date.today()
        # Fetch Mon-Fri of current week
        monday = today - datetime.timedelta(days=today.weekday())
        friday = monday + datetime.timedelta(days=4)

        url = (
            f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        )

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            self._events = []
            for item in raw:
                try:
                    # ForexFactory JSON format
                    title = item.get("title", "")
                    currency = item.get("country", "")
                    impact = item.get("impact", "Low")
                    date_str = item.get("date", "")

                    importance = 1
                    if impact == "High":
                        importance = 3
                    elif impact == "Medium":
                        importance = 2

                    if date_str:
                        event_time = datetime.datetime.fromisoformat(
                            date_str.replace("Z", "+00:00"))
                    else:
                        continue

                    self._events.append(NewsEvent(
                        time_utc=event_time,
                        currency=currency.upper(),
                        importance=importance,
                        title=title,
                    ))
                except (ValueError, KeyError):
                    continue

            self._last_fetch = time.time()
            self._save_cache()
            logger.info("News calendar fetched: %d events this week.", len(self._events))

        except (URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to fetch news calendar: %s — trading NOT blocked.", exc)
            self._last_fetch = time.time()  # don't retry immediately

    def _save_cache(self):
        """Persist events to disk."""
        try:
            data = {
                "fetched_at": self._last_fetch,
                "events": [
                    {
                        "time_utc": e.time_utc.isoformat(),
                        "currency": e.currency,
                        "importance": e.importance,
                        "title": e.title,
                        "country": e.country,
                    }
                    for e in self._events
                ],
            }
            self._cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def has_high_impact_today(self, symbol: str, now: Optional[datetime.datetime] = None) -> bool:
        """Check if there is any high-impact news today for this symbol's currencies."""
        self.refresh_if_needed()
        if not self._events:
            return True  # fail-open: if we can't fetch, allow trading

        if now is None:
            now = datetime.datetime.now(datetime.timezone.utc)

        currencies = SYMBOL_CURRENCIES.get(symbol, ["USD"])
        today = now.date()

        for event in self._events:
            if event.importance >= HIGH_IMPACT and event.currency in currencies:
                if event.time_utc.date() == today:
                    return True
        return False

    def is_near_high_impact(self, symbol: str,
                            now: Optional[datetime.datetime] = None,
                            window_minutes: int = 60) -> bool:
        """Check if we're within window_minutes of a high-impact event.

        Returns True if a high-impact event for this symbol's currencies
        is within [-window, +window] minutes of now.
        """
        self.refresh_if_needed()
        if not self._events:
            return True  # fail-open

        if now is None:
            now = datetime.datetime.now(datetime.timezone.utc)

        currencies = SYMBOL_CURRENCIES.get(symbol, ["USD"])
        window = datetime.timedelta(minutes=window_minutes)

        for event in self._events:
            if event.importance >= HIGH_IMPACT and event.currency in currencies:
                # Make event_time timezone-aware if it isn't
                event_time = event.time_utc
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=datetime.timezone.utc)
                diff = abs(now - event_time)
                if diff <= window:
                    return True
        return False

    def get_todays_events(self, symbol: str,
                          now: Optional[datetime.datetime] = None) -> List[NewsEvent]:
        """Get all events for today for this symbol's currencies."""
        self.refresh_if_needed()
        if now is None:
            now = datetime.datetime.now(datetime.timezone.utc)

        currencies = SYMBOL_CURRENCIES.get(symbol, ["USD"])
        today = now.date()

        return [
            e for e in self._events
            if e.currency in currencies and e.time_utc.date() == today
        ]


# Module-level singleton
_calendar: Optional[NewsCalendar] = None


def get_calendar() -> NewsCalendar:
    global _calendar
    if _calendar is None:
        _calendar = NewsCalendar()
    return _calendar
