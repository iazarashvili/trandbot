"""Data loading for the multi-symbol backtester.

Pulls 6 months of M5, H1, and M15 history from MT5 for each symbol,
caches to disk so MT5 is only needed on the first run (or --refresh).

Usage:
    from back_test.data import load_symbol_data
    d = load_symbol_data("BTCUSD")
    # d = {"m5": DataFrame, "h1": DataFrame, "m15": DataFrame,
    #       "contract_size": float, "point": float}
"""

from pathlib import Path
from typing import Dict

import pandas as pd

HERE = Path(__file__).resolve().parent

# 5 years of data at each timeframe
M5_BARS = 525_600   # ~5 years of 5m bars  (5 * 365 * 24 * 12)
H1_BARS = 43_800    # ~5 years of 1h bars  (5 * 365 * 24)
M15_BARS = 175_200  # ~5 years of 15m bars (5 * 365 * 24 * 4)


def _cache_path(symbol: str) -> Path:
    return HERE / f"_cache_{symbol.lower()}.pkl"


def load_symbol_data(symbol: str, refresh: bool = False) -> Dict:
    """Load M5 / H1 / M15 history for *symbol*, cached to disk.

    Returns
    -------
    dict with keys: m5, h1, m15, contract_size, point
    """
    cache = _cache_path(symbol)

    if cache.exists() and not refresh:
        blob = pd.read_pickle(cache)
        if all(k in blob for k in ("m5", "h1", "m15", "contract_size", "point")):
            print(f"[{symbol}] loaded from cache: "
                  f"{len(blob['m5'])} M5, {len(blob['h1'])} H1, "
                  f"{len(blob['m15'])} M15 bars")
            return blob

    # ------------------------------------------------------------------
    # Pull from MT5
    # ------------------------------------------------------------------
    import MetaTrader5 as mt5

    if not mt5.terminal_info():
        if not mt5.initialize():
            raise RuntimeError(
                f"MT5 is not reachable — start the terminal and retry. "
                f"Last error: {mt5.last_error()}")

    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol {symbol} not found in MT5.")
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol {symbol} could not be selected.")

    contract_size = info.trade_contract_size
    point = info.point

    frames = {}
    for key, tf, n in (
        ("m5",  mt5.TIMEFRAME_M5,  M5_BARS),
        ("h1",  mt5.TIMEFRAME_H1,  H1_BARS),
        ("m15", mt5.TIMEFRAME_M15, M15_BARS),
    ):
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"No {key.upper()} data for {symbol}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        frames[key] = df

    blob = {
        **frames,
        "contract_size": contract_size,
        "point": point,
    }

    pd.to_pickle(blob, cache)
    print(f"[{symbol}] cached {len(frames['m5'])} M5 / "
          f"{len(frames['h1'])} H1 / {len(frames['m15'])} M15 bars "
          f"-> {cache.name}  (contract={contract_size}, point={point})")
    return blob
