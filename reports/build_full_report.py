"""Build HTML report — multi-symbol backtest with ALL live engine filters.

Mirrors core/engine.py gate-for-gate:
  1. Weekend / blocked days / night filter
  2. Macro windows (ICT killzones)
  3. Blocked trends (per-symbol)
  4. HTF trend + POI
  5. DOL gate (draw on liquidity required)
  6. Zone patience + LTF FVG confirmation
  7. Blocked strategy hours (per-symbol)
  8. Entry drift guard (>20% = reject)
  9. OB too wide filter (per-symbol)
  10. Liquidity resistance filter
  11. Risk caps (daily loss, consecutive losses, cooldown, max trades/day)
  12. Position sizing from shared balance

Usage:
    python reports/build_full_report.py                # default, needs MT5 for first run
    python reports/build_full_report.py --refresh      # re-pull history from MT5
"""

import argparse
import importlib
import sys
import time as _time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    HTF, HTF_CANDLES_LOOKBACK, LTF, LTF_CANDLES_LOOKBACK,
    LIQUIDITY_TF, MAX_LTF_WAIT_CANDLES, SWING_STRENGTH,
    TREND_EMA_PERIOD, MIN_RRR_LIQUIDITY,
)
from strategy import SMCStrategy

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- config
BALANCE = 10_000.0
SYMBOLS = ["BTCUSD", "XAUUSD", "GBPUSD"]

# ICT macro windows (UTC) — same as core/engine.py
MACRO_WINDOWS = [
    (2, 0, 5, 0),
    (7, 0, 11, 0),
    (12, 0, 13, 0),
    (13, 30, 16, 0),
    (19, 0, 20, 0),
]

M5_BARS = 100_000
H1_BARS = 10_000
M15_BARS = 50_000


def _in_macro_window(hour, minute):
    current = hour * 60 + minute
    for sh, sm, eh, em in MACRO_WINDOWS:
        if sh * 60 + sm <= current < eh * 60 + em:
            return True
    return False


def _load_cfg(symbol):
    """Load per-symbol config — mirrors SymbolConfig.load()."""
    try:
        mod = importlib.import_module(f"config_{symbol.lower()}")
    except ModuleNotFoundError:
        mod = importlib.import_module("config")
    cfg = {
        "night_start": getattr(mod, "NIGHT_START_HOUR", 0),
        "night_end": getattr(mod, "NIGHT_END_HOUR", 0),
        "blocked_days": list(getattr(mod, "BLOCKED_DAYS", [])),
        "skip_weekends": getattr(mod, "SKIP_WEEKENDS", True),
        "rrr": getattr(mod, "RRR", 2.0),
        "stop_mode": getattr(mod, "STOP_MODE", "swing"),
        "risk_percent": getattr(mod, "RISK_PERCENT", 0.4),
        "account_type": getattr(mod, "ACCOUNT_TYPE", "funded"),
        "max_risk_usd": getattr(mod, "MAX_RISK_USD", 0.0),
        "max_risk_pct": getattr(mod, "MAX_RISK_PCT", 0.0),
        "use_breakeven": getattr(mod, "USE_BREAKEVEN", False),
        "breakeven_r": getattr(mod, "BREAKEVEN_R", 2.0),
        "use_partial": getattr(mod, "USE_PARTIAL_CLOSE", True),
        "partial_trigger_pct": getattr(mod, "PARTIAL_TRIGGER_PCT", 0.80),
        "partial_close_pct": getattr(mod, "PARTIAL_CLOSE_PCT", 0.80),
        "blocked_trends": list(getattr(mod, "BLOCKED_TRENDS", [])),
        "max_ob_risk_px": getattr(mod, "MAX_OB_RISK_PX", 0.0),
        "use_liquidity_resistance": getattr(mod, "USE_LIQUIDITY_RESISTANCE", True),
        "high_resistance_risk_factor": getattr(mod, "HIGH_RESISTANCE_RISK_FACTOR", 0.25),
        "max_consecutive_losses": getattr(mod, "MAX_CONSECUTIVE_LOSSES", 3),
        "max_daily_loss_pct": getattr(mod, "MAX_DAILY_LOSS_PCT", 4.0),
        "daily_loss_buffer_pct": getattr(mod, "DAILY_LOSS_BUFFER_PCT", 0.5),
        "cooldown_candles": getattr(mod, "COOLDOWN_CANDLES", 3),
    }
    # Resolve account type
    _ACCOUNT_RISK = {"funded": 0.4, "challenge": 0.6}
    if cfg["account_type"] in _ACCOUNT_RISK:
        cfg["risk_percent"] = _ACCOUNT_RISK[cfg["account_type"]]
    return cfg


# ---------------------------------------------------------------- data
def load_data(refresh=False):
    """Fetch and cache M5/H1/M15 data for each symbol."""
    all_data = {}
    need_mt5 = False

    for sym in SYMBOLS:
        cache = HERE / f"_history_{sym.lower()}.pkl"
        if cache.exists() and not refresh:
            blob = pd.read_pickle(cache)
            if "m5" in blob:
                all_data[sym] = blob
                continue
        need_mt5 = True

    if need_mt5:
        import MetaTrader5 as mt5
        mt5.initialize()
        from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
        mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

        for sym in SYMBOLS:
            cache = HERE / f"_history_{sym.lower()}.pkl"
            if sym in all_data:
                continue
            info = mt5.symbol_info(sym)
            if info is None:
                print(f"WARNING: {sym} not available in MT5, skipping.")
                continue
            frames = {}
            for key, tf, n in [("m5", LTF, M5_BARS),
                                ("h1", HTF, H1_BARS),
                                ("m15", LIQUIDITY_TF, M15_BARS)]:
                rates = mt5.copy_rates_from_pos(sym, tf, 0, n)
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                frames[key] = df
            blob = {**frames, "contract": info.trade_contract_size, "point": info.point}
            pd.to_pickle(blob, cache)
            all_data[sym] = blob
            print(f"Cached {sym}: {len(frames['m5'])} M5 bars -> {cache.name}")

        mt5.shutdown()

    return all_data


# ---------------------------------------------------------------- backtest
def run_backtest(all_data):
    """Run multi-symbol backtest with shared balance and ALL live filters."""

    balance = BALANCE
    global_trades_today = 0
    global_trade_date = ""
    max_trades_per_day = 1  # from config.py

    # Per-symbol state
    sym_state = {}
    for sym in SYMBOLS:
        if sym not in all_data:
            continue
        sym_state[sym] = {
            "consecutive_losses": 0,
            "daily_pnl": 0.0,
            "daily_date": "",
            "cooldown_until": -1,  # bar index
            "trades_today": 0,
            "weekly_pnl": 0.0,
            "weekly_key": "",
            "monthly_pnl": 0.0,
            "monthly_key": "",
        }

    # Precompute arrays for each symbol
    sym_arrays = {}
    for sym in SYMBOLS:
        if sym not in all_data:
            continue
        d = all_data[sym]
        cfg = _load_cfg(sym)
        m5, h1, m15 = d["m5"], d["h1"], d["m15"]
        point, contract = d["point"], d["contract"]

        h1_t = h1["time"].values
        m5_t = m5["time"].values
        m15_t = m15["time"].values
        h_idx = np.searchsorted(h1_t, m5_t - np.timedelta64(60, "m"), side="right") - 1
        liq_idx = np.searchsorted(m15_t, m5_t - np.timedelta64(15, "m"), side="right") - 1

        sym_arrays[sym] = {
            "m5": m5, "h1": h1, "m15": m15,
            "o": m5["open"].to_numpy(float),
            "hi": m5["high"].to_numpy(float),
            "lo": m5["low"].to_numpy(float),
            "cl": m5["close"].to_numpy(float),
            "spread": m5["spread"].to_numpy(float) * point,
            "times": m5["time"],
            "h_idx": h_idx, "liq_idx": liq_idx,
            "weekdays": pd.to_datetime(m5["time"]).dt.weekday.to_numpy(),
            "hours": pd.to_datetime(m5["time"]).dt.hour.to_numpy(),
            "minutes": pd.to_datetime(m5["time"]).dt.minute.to_numpy(),
            "dates": pd.to_datetime(m5["time"]).dt.strftime("%Y-%m-%d").values,
            "contract": contract, "point": point, "cfg": cfg,
            # HTF cache
            "htf_cache": {}, "poi": None, "poi_h": -1,
            # Position
            "position": None,
            # Zone tracking
            "watch_key": None, "watch_start": None, "abandoned": False,
        }

    # Find common time range
    starts = {}
    for sym, sa in sym_arrays.items():
        s = LTF_CANDLES_LOOKBACK
        while s < len(sa["m5"]) and sa["h_idx"][s] < HTF_CANDLES_LOOKBACK:
            s += 1
        starts[sym] = s

    max_bars = max(len(sa["m5"]) for sa in sym_arrays.values())

    all_trades = []
    all_rejections = defaultdict(lambda: defaultdict(int))
    equity = [{"time": "", "balance": BALANCE}]
    t_start = _time.time()

    for t in range(min(starts.values()), max_bars - 1):
        if t % 20000 == 0:
            elapsed = _time.time() - t_start
            print(f"  ... bar {t}/{max_bars} ({elapsed:.0f}s, {len(all_trades)} trades, bal=${balance:.0f})")

        for sym, sa in sym_arrays.items():
            if t < starts[sym] or t >= len(sa["m5"]) - 1:
                continue

            cfg = sa["cfg"]
            o, hi, lo, cl = sa["o"], sa["hi"], sa["lo"], sa["cl"]
            spread = sa["spread"]
            times = sa["times"]
            h_idx, liq_idx = sa["h_idx"], sa["liq_idx"]
            m5, h1, m15 = sa["m5"], sa["h1"], sa["m15"]
            contract = sa["contract"]
            state = sym_state[sym]

            cur_date = sa["dates"][t]
            hour = int(sa["hours"][t])
            minute = int(sa["minutes"][t])
            weekday = int(sa["weekdays"][t])

            # Daily reset (including consecutive losses to prevent deadlock)
            if state["daily_date"] != cur_date:
                state["daily_date"] = cur_date
                state["daily_pnl"] = 0.0
                state["trades_today"] = 0
                state["consecutive_losses"] = 0

            # Weekly/monthly reset
            ts = pd.Timestamp(sa["times"].iloc[t])
            week_key = ts.strftime("%G-W%V")
            if state["weekly_key"] != week_key:
                state["weekly_key"] = week_key
                state["weekly_pnl"] = 0.0
            month_key = ts.strftime("%Y-%m")
            if state["monthly_key"] != month_key:
                state["monthly_key"] = month_key
                state["monthly_pnl"] = 0.0

            # Global daily reset
            if global_trade_date != cur_date:
                global_trade_date = cur_date
                global_trades_today = 0

            # ---- Manage open position ----
            if sa["position"] is not None:
                p = sa["position"]
                vol = p["vol"]

                # Breakeven
                if cfg["use_breakeven"] and not p.get("be") and not p.get("partial_done"):
                    td = p["risk_px"] * cfg["breakeven_r"]
                    if p["dir"] == "BUY" and hi[t] >= p["entry"] + td:
                        p["sl"] = p["entry"]; p["be"] = True
                    elif p["dir"] == "SELL" and lo[t] + spread[t] <= p["entry"] - td:
                        p["sl"] = p["entry"]; p["be"] = True

                # Partial close
                if cfg["use_partial"] and not p.get("partial_done"):
                    tp_dist = abs(p["tp"] - p["entry"])
                    trigger_dist = tp_dist * cfg["partial_trigger_pct"]
                    if p["dir"] == "BUY":
                        triggered = hi[t] >= p["entry"] + trigger_dist
                    else:
                        triggered = lo[t] + spread[t] <= p["entry"] - trigger_dist
                    if triggered:
                        close_vol = vol * cfg["partial_close_pct"]
                        remain_vol = vol - close_vol
                        partial_px = (p["entry"] + trigger_dist if p["dir"] == "BUY"
                                      else p["entry"] - trigger_dist)
                        partial_pnl = ((partial_px - p["entry"]) if p["dir"] == "BUY"
                                       else (p["entry"] - partial_px)) * close_vol * contract
                        balance += partial_pnl
                        p["partial_pnl"] = round(partial_pnl, 2)
                        p["partial_done"] = True
                        p["vol"] = remain_vol
                        p["sl"] = p["entry"]
                        # Next liquidity for runner TP
                        li = liq_idx[t]
                        if li >= 0:
                            liq_slice = m15.iloc[max(0, li - 200): li + 1]
                            next_liq = SMCStrategy.find_next_liquidity(
                                liq_slice, p["dir"], p["tp"],
                                strength=SWING_STRENGTH, use_closed_candles=False)
                            if next_liq is not None:
                                p["tp"] = round(next_liq, 5)

                # Check SL/TP hit
                if p["dir"] == "BUY":
                    p["mfe"] = max(p.get("mfe", 0), hi[t] - p["entry"])
                    p["mae"] = max(p.get("mae", 0), p["entry"] - lo[t])
                    hit_sl = lo[t] <= p["sl"]
                    hit_tp = hi[t] >= p["tp"]
                else:
                    p["mfe"] = max(p.get("mfe", 0), p["entry"] - (lo[t] + spread[t]))
                    p["mae"] = max(p.get("mae", 0), (hi[t] + spread[t]) - p["entry"])
                    hit_sl = hi[t] + spread[t] >= p["sl"]
                    hit_tp = lo[t] + spread[t] <= p["tp"]

                if hit_sl or hit_tp:
                    ep = p["sl"] if hit_sl else p["tp"]
                    remaining_pnl = ((ep - p["entry"]) if p["dir"] == "BUY"
                                     else (p["entry"] - ep)) * p["vol"] * contract
                    balance += remaining_pnl
                    total_pnl = remaining_pnl + p.get("partial_pnl", 0)
                    ru = p["risk_px"] * p.get("orig_vol", vol) * contract

                    trade = {
                        "symbol": sym, "strategy": p["strategy"], "dir": p["dir"],
                        "entry_time": p["entry_time"], "exit_time": str(times.iloc[t]),
                        "entry": p["entry"], "sl": p["sl"], "tp": p["tp"],
                        "vol": p.get("orig_vol", vol), "risk_px": p["risk_px"],
                        "risk_usd": round(ru, 2),
                        "pnl": round(total_pnl, 2),
                        "r_multiple": round(total_pnl / ru, 2) if ru > 0 else 0,
                        "exit_reason": "SL" if hit_sl else "TP",
                        "bars_held": t - p["entry_bar"],
                        "balance": round(balance, 2),
                        "entry_hour": p.get("eh", hour),
                        "entry_weekday": p.get("ew", weekday),
                        "mfe_r": round(p.get("mfe", 0) / p["risk_px"], 2) if p["risk_px"] > 0 else 0,
                        "mae_r": round(p.get("mae", 0) / p["risk_px"], 2) if p["risk_px"] > 0 else 0,
                        "dol_target": p.get("dol_target"),
                        "liq_resistance": p.get("liq_resistance", ""),
                        "partial_pnl": p.get("partial_pnl", 0),
                    }
                    all_trades.append(trade)
                    equity.append({"time": trade["exit_time"], "balance": round(balance, 2)})
                    sa["position"] = None
                    sa["watch_key"] = None
                    sa["abandoned"] = False

                    # Update risk state
                    state["daily_pnl"] += total_pnl
                    state["weekly_pnl"] += total_pnl
                    state["monthly_pnl"] += total_pnl
                    state["trades_today"] += 1
                    if total_pnl <= 0:
                        state["consecutive_losses"] += 1
                    else:
                        state["consecutive_losses"] = 0
                    # Cooldown
                    if cfg["cooldown_candles"] > 0:
                        state["cooldown_until"] = t + cfg["cooldown_candles"]

                    if balance <= 0:
                        break
                continue  # position open, don't scan

            # ========== GATES (same order as live engine) ==========

            # Gate 0: Global max trades per day
            if global_trades_today >= max_trades_per_day:
                all_rejections[sym]["GLOBAL_MAX_DAILY_TRADES"] += 1
                continue

            # Gate 1: Weekend
            if cfg["skip_weekends"] and weekday >= 5:
                continue

            # Gate 2: Blocked days
            if cfg["blocked_days"] and weekday in cfg["blocked_days"]:
                continue

            # Gate 3: Night filter
            ns, ne = cfg["night_start"], cfg["night_end"]
            if ns != ne:
                if ns > ne:
                    if hour >= ns or hour < ne:
                        continue
                elif ns <= hour < ne:
                    continue

            # Gate 4: Risk limits
            # Consecutive losses
            if state["consecutive_losses"] >= cfg["max_consecutive_losses"]:
                all_rejections[sym]["CONSECUTIVE_LOSSES"] += 1
                continue

            # Daily loss limit
            eff_limit = cfg["max_daily_loss_pct"] - cfg["daily_loss_buffer_pct"]
            if eff_limit > 0 and balance > 0:
                daily_loss_pct = abs(min(0, state["daily_pnl"])) / balance * 100
                if daily_loss_pct >= eff_limit:
                    all_rejections[sym]["DAILY_LOSS_LIMIT"] += 1
                    continue

            # Daily trade limit (per-symbol)
            if state["trades_today"] >= max_trades_per_day:
                all_rejections[sym]["MAX_DAILY_TRADES"] += 1
                continue

            # Cooldown
            if t < state["cooldown_until"]:
                all_rejections[sym]["COOLDOWN"] += 1
                continue

            # Weekly/monthly profit caps
            if balance > BALANCE:  # only cap profits, not during drawdown
                weekly_pnl = state.get("weekly_pnl", 0)
                monthly_pnl = state.get("monthly_pnl", 0)
                if balance > 0:
                    if weekly_pnl / balance * 100 >= 2.0:
                        all_rejections[sym]["WEEKLY_PROFIT_CAP"] += 1
                        continue
                    if monthly_pnl / balance * 100 >= 6.0:
                        all_rejections[sym]["MONTHLY_PROFIT_CAP"] += 1
                        continue

            # Gate 5: Macro windows
            if not _in_macro_window(hour, minute):
                continue

            # Gate 6: HTF trend + POI
            if h_idx[t] != sa["poi_h"]:
                sa["poi_h"] = h_idx[t]
                ck = sa["poi_h"]
                if ck in sa["htf_cache"]:
                    trend, poi = sa["htf_cache"][ck]
                else:
                    wh = h1.iloc[max(0, ck - HTF_CANDLES_LOOKBACK + 1): ck + 1]
                    trend = SMCStrategy.get_htf_trend(wh, TREND_EMA_PERIOD,
                                                       use_closed_candles=False)
                    poi = SMCStrategy.detect_htf_poi(
                        wh, use_trend_filter=True,
                        ema_period=TREND_EMA_PERIOD, use_closed_candles=False)
                    sa["htf_cache"][ck] = (trend, poi)
                if poi is None:
                    sa["watch_key"] = None
                    sa["abandoned"] = False
            else:
                cached = sa["htf_cache"].get(sa["poi_h"])
                trend, poi = cached if cached else ("NEUTRAL", None)

            # Blocked trends
            if trend in cfg["blocked_trends"]:
                all_rejections[sym]["BLOCKED_TREND"] += 1
                continue

            if trend == "NEUTRAL":
                continue

            if poi is None:
                continue

            # Gate 7: DOL — Draw on Liquidity (hard gate)
            li = liq_idx[t]
            dol_target = None
            if li >= 0:
                liq_slice = m15.iloc[max(0, li - 200): li + 1]
                direction = "BUY" if poi.type == "BULLISH" else "SELL"
                dol_target = SMCStrategy.find_nearest_liquidity(
                    liq_slice, direction, cl[t],
                    strength=SWING_STRENGTH, use_closed_candles=False)

            if dol_target is None:
                all_rejections[sym]["NO_DOL"] += 1
                continue

            # Gate 8: Zone patience — price in/near POI
            mid = cl[t] + spread[t] / 2.0
            ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]

            if not SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
                sa["watch_key"] = None
                sa["abandoned"] = False
                continue

            zone_key = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
            if zone_key != sa["watch_key"]:
                sa["watch_key"] = zone_key
                sa["watch_start"] = t
                sa["abandoned"] = False

            if sa["abandoned"]:
                continue

            if sa["watch_start"] is not None and (t - sa["watch_start"] + 1) > MAX_LTF_WAIT_CANDLES:
                sa["abandoned"] = True
                continue

            # Gate 9: LTF confirmation — 5m FVG inside POI
            # Live engine uses ONLY check_ltf_confirmation (no AMD/SB/Sweep).
            setup = SMCStrategy.check_ltf_confirmation(
                ltf, poi, cfg["rrr"], use_closed_candles=False,
                stop_mode=cfg["stop_mode"], buffer_atr=0.5)

            if setup is None:
                continue
            strategy_name = "PLAN"

            # Gate 10: Entry price and drift guard
            nt = t + 1
            entry = o[nt] + spread[nt] if setup["direction"] == "BUY" else o[nt]
            sl = setup["sl"]
            risk = abs(entry - sl)
            if risk <= 0:
                continue

            intended_entry = setup.get("entry", entry)
            if abs(entry - intended_entry) / risk > 0.20:
                all_rejections[sym]["ENTRY_DRIFT"] += 1
                continue

            # Gate 11: OB too wide
            if cfg["max_ob_risk_px"] > 0 and risk > cfg["max_ob_risk_px"]:
                all_rejections[sym]["OB_TOO_WIDE"] += 1
                continue

            # Gate 12: DOL-based TP
            tp = setup["tp"]
            dol_dist = abs(dol_target - entry)
            dol_rr = dol_dist / risk if risk > 0 else 0
            if dol_rr >= MIN_RRR_LIQUIDITY:
                tp = dol_target
            # else keep fixed RRR tp

            # Gate 13: Min RR check (from validate_entry)
            reward_dist = abs(tp - entry)
            actual_rr = reward_dist / risk if risk > 0 else 0
            if actual_rr < MIN_RRR_LIQUIDITY:
                all_rejections[sym]["RR_TOO_LOW"] += 1
                continue

            # Gate 14: Liquidity resistance
            liq_resistance = ""
            if cfg["use_liquidity_resistance"] and li >= 0:
                liq_slice = m15.iloc[max(0, li - 200): li + 1]
                liq_resistance = SMCStrategy.classify_liquidity_resistance(
                    liq_slice, tp, setup["direction"],
                    swing_strength=SWING_STRENGTH, use_closed_candles=False)
                if liq_resistance == "HIGH" and cfg["high_resistance_risk_factor"] <= 0:
                    all_rejections[sym]["HIGH_RESISTANCE"] += 1
                    continue

            # Gate 14: Position sizing
            risk_pct = cfg["risk_percent"]
            if balance <= 0:
                break
            risk_amount = balance * risk_pct / 100.0
            pnl_per_lot = risk * contract
            if pnl_per_lot <= 0:
                continue
            vol = max(0.01, int(risk_amount / pnl_per_lot * 100) / 100.0)

            # Risk caps
            risk_usd = risk * vol * contract
            if cfg["max_risk_pct"] > 0:
                max_allowed = balance * cfg["max_risk_pct"] / 100
                if risk_usd > max_allowed:
                    all_rejections[sym]["RISK_CAP"] += 1
                    continue
            elif cfg["max_risk_usd"] > 0 and risk_usd > cfg["max_risk_usd"]:
                all_rejections[sym]["RISK_CAP"] += 1
                continue

            # Liquidity resistance risk reduction
            if liq_resistance == "HIGH" and cfg["high_resistance_risk_factor"] < 1.0:
                vol = max(0.01, int(vol * cfg["high_resistance_risk_factor"] * 100) / 100.0)

            # ========== OPEN POSITION ==========
            sa["position"] = {
                "dir": setup["direction"],
                "entry_time": str(times.iloc[nt]),
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "risk_px": round(risk, 5),
                "vol": vol,
                "orig_vol": vol,
                "entry_bar": nt,
                "strategy": strategy_name,
                "eh": int(sa["hours"][nt]) if nt < len(sa["hours"]) else hour,
                "ew": int(sa["weekdays"][nt]) if nt < len(sa["weekdays"]) else weekday,
                "dol_target": round(dol_target, 5),
                "liq_resistance": liq_resistance or "LOW",
            }
            global_trades_today += 1
            sa["watch_key"] = None
            sa["abandoned"] = False

    # Sort all trades by exit time
    all_trades.sort(key=lambda t: t.get("exit_time", ""))

    # Recalculate shared balance
    shared_bal = BALANCE
    for t in all_trades:
        shared_bal += t["pnl"]
        t["shared_balance"] = round(shared_bal, 2)

    elapsed = _time.time() - t_start
    print(f"\nDone: {len(all_trades)} trades in {elapsed:.1f}s")
    return all_trades, equity, all_rejections


# ---------------------------------------------------------------- HTML report
def build_html(all_trades, equity, rejections):
    """Generate detailed HTML report."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    symbols = SYMBOLS

    total_t = len(all_trades)
    if total_t == 0:
        print("No trades — nothing to report.")
        return "<html><body><h1>No trades</h1></body></html>"

    total_w = len([t for t in all_trades if t["pnl"] > 0])
    total_l = total_t - total_w
    total_pnl = sum(t["pnl"] for t in all_trades)
    gw = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
    gl = -sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)
    pf = round(gw / gl, 2) if gl > 0 else 0
    final_bal = all_trades[-1]["shared_balance"] if all_trades else BALANCE

    # Max drawdown
    peak = BALANCE; max_dd = 0; max_dd_pct = 0
    for t in all_trades:
        peak = max(peak, t["shared_balance"])
        dd = peak - t["shared_balance"]
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak * 100

    # Streaks
    streak = 0; mws = 0; mls = 0
    for t in all_trades:
        if t["pnl"] > 0:
            streak = streak + 1 if streak > 0 else 1
            mws = max(mws, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            mls = max(mls, abs(streak))

    # Expectancy in R
    avg_r = np.mean([t["r_multiple"] for t in all_trades]) if all_trades else 0

    # Monthly
    monthly = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0})
    for t in all_trades:
        m = t["exit_time"][:7]
        monthly[m]["t"] += 1; monthly[m]["pnl"] += t["pnl"]
        if t["pnl"] > 0: monthly[m]["w"] += 1
    months_sorted = sorted(monthly.keys())

    # Strategy
    strategies = sorted(set(t["strategy"] for t in all_trades))
    strat_data = {}
    for s in strategies:
        st = [t for t in all_trades if t["strategy"] == s]
        w = len([t for t in st if t["pnl"] > 0])
        g = sum(t["pnl"] for t in st if t["pnl"] > 0)
        l_val = -sum(t["pnl"] for t in st if t["pnl"] <= 0)
        strat_data[s] = {"t": len(st), "w": w,
                         "wr": round(w / len(st) * 100, 1) if st else 0,
                         "pf": round(g / l_val, 2) if l_val > 0 else 0,
                         "net": round(sum(t["pnl"] for t in st), 2),
                         "avg_r": round(np.mean([t["r_multiple"] for t in st]), 3) if st else 0}

    # Symbol
    sym_data = {}
    for sym in symbols:
        st = [t for t in all_trades if t["symbol"] == sym]
        if not st:
            sym_data[sym] = {"t": 0, "w": 0, "wr": 0, "pf": 0, "net": 0, "avg_r": 0}
            continue
        w = len([t for t in st if t["pnl"] > 0])
        g = sum(t["pnl"] for t in st if t["pnl"] > 0)
        l_val = -sum(t["pnl"] for t in st if t["pnl"] <= 0)
        sym_data[sym] = {"t": len(st), "w": w,
                         "wr": round(w / len(st) * 100, 1),
                         "pf": round(g / l_val, 2) if l_val > 0 else 0,
                         "net": round(sum(t["pnl"] for t in st), 2),
                         "avg_r": round(np.mean([t["r_multiple"] for t in st]), 3)}

    # Hour
    hour_data = {}
    for h in range(24):
        ht = [t for t in all_trades if t["entry_hour"] == h]
        if not ht: continue
        w = len([t for t in ht if t["pnl"] > 0])
        hour_data[h] = {"t": len(ht), "w": w, "net": round(sum(t["pnl"] for t in ht), 2)}

    # Day
    day_data = {}
    for d in range(7):
        dt = [t for t in all_trades if t["entry_weekday"] == d]
        if not dt: continue
        w = len([t for t in dt if t["pnl"] > 0])
        day_data[d] = {"t": len(dt), "w": w, "net": round(sum(t["pnl"] for t in dt), 2)}

    # Matrix: symbol x strategy
    matrix = {}
    for sym in symbols:
        for s in strategies:
            st = [t for t in all_trades if t["symbol"] == sym and t["strategy"] == s]
            if st:
                w = len([t for t in st if t["pnl"] > 0])
                matrix[f"{sym}_{s}"] = {"t": len(st), "w": w,
                                         "wr": round(w / len(st) * 100, 1),
                                         "net": round(sum(t["pnl"] for t in st), 2)}

    # Daily P&L
    daily_pnl = defaultdict(float)
    for t in all_trades:
        daily_pnl[t["exit_time"][:10]] += t["pnl"]
    worst10 = sorted(daily_pnl.items(), key=lambda x: x[1])[:10]
    best10 = sorted(daily_pnl.items(), key=lambda x: x[1], reverse=True)[:10]

    # Direction
    dir_data = {}
    for d in ["BUY", "SELL"]:
        dt = [t for t in all_trades if t["dir"] == d]
        if dt:
            w = len([t for t in dt if t["pnl"] > 0])
            dir_data[d] = {"t": len(dt), "w": w,
                           "wr": round(w / len(dt) * 100, 1),
                           "net": round(sum(t["pnl"] for t in dt), 2)}

    # Equity JSON (sampled)
    eq_points = [{"time": "", "balance": BALANCE}]
    for t in all_trades:
        eq_points.append({"time": t["exit_time"][:16], "balance": t["shared_balance"]})
    step = max(1, len(eq_points) // 300)
    eq_json = ",".join([f'["{e["time"]}",{e["balance"]}]' for e in eq_points[::step]])

    # Monthly chart
    monthly_labels = ",".join([f'"{m}"' for m in months_sorted])
    monthly_vals = ",".join([f'{monthly[m]["pnl"]:.0f}' for m in months_sorted])
    monthly_colors = ",".join(['"#089981"' if monthly[m]["pnl"] >= 0 else '"#F23645"' for m in months_sorted])

    # Rejection summary
    rej_rows = ""
    for sym in symbols:
        for reason, count in sorted(rejections.get(sym, {}).items(), key=lambda x: -x[1]):
            rej_rows += f'<tr><td>{sym}</td><td>{reason}</td><td>{count:,}</td></tr>\n'

    # Period
    period_from = all_trades[0]["entry_time"][:10] if all_trades else "?"
    period_to = all_trades[-1]["exit_time"][:10] if all_trades else "?"

    # Badge class
    def badge_cls(s):
        if "AMD" in s: return "amd"
        if "SILVER" in s or "SB" in s: return "sb"
        if "SWEEP" in s: return "sweep"
        if "PLAN" in s: return "poi"
        return "poi"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMC Bot — Full Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0f1117; color:#d1d4dc; line-height:1.6; }}
.container {{ max-width:1400px; margin:0 auto; padding:20px; }}
h1 {{ color:#fff; font-size:28px; margin-bottom:5px; }}
h2 {{ color:#2962ff; font-size:20px; margin:30px 0 15px; border-bottom:1px solid #2a2e39; padding-bottom:8px; }}
.subtitle {{ color:#787b86; font-size:14px; margin-bottom:30px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:30px; }}
.card {{ background:#1e222d; border-radius:12px; padding:18px; text-align:center; }}
.card .value {{ font-size:26px; font-weight:bold; color:#fff; }}
.card .label {{ font-size:11px; color:#787b86; margin-top:4px; text-transform:uppercase; }}
.card.green .value {{ color:#089981; }}
.card.red .value {{ color:#F23645; }}
.card.blue .value {{ color:#2962ff; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:20px; background:#1e222d; border-radius:8px; overflow:hidden; }}
th {{ background:#2a2e39; color:#787b86; padding:10px 12px; text-align:left; font-size:11px; text-transform:uppercase; }}
td {{ padding:8px 12px; border-top:1px solid #2a2e39; font-size:13px; }}
tr:hover {{ background:#262a35; }}
.pos {{ color:#089981; }}
.neg {{ color:#F23645; }}
.chart-box {{ background:#1e222d; border-radius:12px; padding:20px; margin-bottom:20px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
.grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; }}
@media(max-width:900px) {{ .grid2,.grid3 {{ grid-template-columns:1fr; }} }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:4px; font-size:11px; font-weight:bold; }}
.badge-amd {{ background:#2962ff33; color:#2962ff; }}
.badge-sb {{ background:#08998133; color:#089981; }}
.badge-sweep {{ background:#f5a62333; color:#f5a623; }}
.badge-poi {{ background:#787b8633; color:#d1d4dc; }}
.filter-bar {{ margin:10px 0; display:flex; gap:8px; flex-wrap:wrap; }}
.filter-bar button {{ background:#2a2e39; color:#d1d4dc; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:12px; }}
.filter-bar button:hover,.filter-bar button.active {{ background:#2962ff; color:#fff; }}
.tag {{ font-size:10px; padding:2px 6px; border-radius:3px; }}
.tag-high {{ background:#F2364533; color:#F23645; }}
.tag-low {{ background:#08998133; color:#089981; }}
</style>
</head>
<body>
<div class="container">

<h1>SMC Bot — Full Backtest Report</h1>
<p class="subtitle">${BALANCE:,.0f} start | {', '.join(symbols)} | {period_from} to {period_to} | All live filters active</p>

<!-- KPI Cards -->
<div class="cards">
<div class="card {'green' if total_pnl>=0 else 'red'}"><div class="value">${final_bal:,.0f}</div><div class="label">Final Balance</div></div>
<div class="card {'green' if total_pnl>=0 else 'red'}"><div class="value">{(final_bal/BALANCE-1)*100:+.1f}%</div><div class="label">Total Return</div></div>
<div class="card blue"><div class="value">{total_t}</div><div class="label">Total Trades</div></div>
<div class="card"><div class="value">{round(total_w/total_t*100,1) if total_t else 0}%</div><div class="label">Win Rate</div></div>
<div class="card blue"><div class="value">{pf}</div><div class="label">Profit Factor</div></div>
<div class="card {'green' if avg_r>0 else 'red'}"><div class="value">{avg_r:+.3f}R</div><div class="label">Avg Expectancy</div></div>
<div class="card red"><div class="value">{max_dd_pct:.1f}%</div><div class="label">Max Drawdown</div></div>
<div class="card"><div class="value">{mws}W / {mls}L</div><div class="label">Max Streaks</div></div>
</div>

<!-- Equity Curve -->
<div class="chart-box">
<h2 style="margin-top:0">Equity Curve (shared balance)</h2>
<canvas id="equityChart" height="80"></canvas>
</div>

<!-- Monthly P&L -->
<div class="chart-box">
<h2 style="margin-top:0">Monthly P&L</h2>
<canvas id="monthlyChart" height="60"></canvas>
</div>

<!-- Symbol Performance -->
<h2>Symbol Performance</h2>
<table>
<tr><th>Symbol</th><th>Trades</th><th>Wins</th><th>Win Rate</th><th>PF</th><th>Avg R</th><th>Net P&L</th></tr>
{"".join(f'<tr><td><b>{s}</b></td><td>{sym_data[s]["t"]}</td><td>{sym_data[s]["w"]}</td><td>{sym_data[s]["wr"]}%</td><td>{sym_data[s]["pf"]}</td><td>{sym_data[s]["avg_r"]:+.3f}</td><td class="{"pos" if sym_data[s]["net"]>=0 else "neg"}">${sym_data[s]["net"]:+,.2f}</td></tr>' for s in symbols)}
</table>

<!-- Strategy Performance -->
<h2>Strategy Performance</h2>
<table>
<tr><th>Strategy</th><th>Trades</th><th>Wins</th><th>Win Rate</th><th>PF</th><th>Avg R</th><th>Net P&L</th></tr>
{"".join(f'<tr><td><span class="badge badge-{badge_cls(s)}">{s}</span></td><td>{strat_data[s]["t"]}</td><td>{strat_data[s]["w"]}</td><td>{strat_data[s]["wr"]}%</td><td>{strat_data[s]["pf"]}</td><td>{strat_data[s]["avg_r"]:+.3f}</td><td class="{"pos" if strat_data[s]["net"]>=0 else "neg"}">${strat_data[s]["net"]:+,.2f}</td></tr>' for s in strategies)}
</table>

<!-- Symbol x Strategy Matrix -->
<h2>Symbol x Strategy Matrix</h2>
<table>
<tr><th>Symbol</th><th>Strategy</th><th>Trades</th><th>Wins</th><th>WR%</th><th>Net P&L</th></tr>
{"".join(f'<tr><td>{k.split("_")[0]}</td><td><span class="badge badge-{badge_cls(k)}">{" ".join(k.split("_")[1:])}</span></td><td>{v["t"]}</td><td>{v["w"]}</td><td>{v["wr"]}%</td><td class="{"pos" if v["net"]>=0 else "neg"}">${v["net"]:+,.2f}</td></tr>' for k,v in sorted(matrix.items()))}
</table>

<div class="grid2">
<!-- Direction -->
<div>
<h2>Direction Analysis</h2>
<table>
<tr><th>Dir</th><th>Trades</th><th>Wins</th><th>WR%</th><th>Net P&L</th></tr>
{"".join(f'<tr><td><b>{d}</b></td><td>{dir_data[d]["t"]}</td><td>{dir_data[d]["w"]}</td><td>{dir_data[d]["wr"]}%</td><td class="{"pos" if dir_data[d]["net"]>=0 else "neg"}">${dir_data[d]["net"]:+,.2f}</td></tr>' for d in ["BUY","SELL"] if d in dir_data)}
</table>
</div>

<!-- Day of Week -->
<div>
<h2>Day of Week</h2>
<table>
<tr><th>Day</th><th>Trades</th><th>Wins</th><th>WR%</th><th>Net P&L</th></tr>
{"".join(f'<tr><td>{day_names[d]}</td><td>{day_data[d]["t"]}</td><td>{day_data[d]["w"]}</td><td>{round(day_data[d]["w"]/day_data[d]["t"]*100,1)}%</td><td class="{"pos" if day_data[d]["net"]>=0 else "neg"}">${day_data[d]["net"]:+,.2f}</td></tr>' for d in sorted(day_data))}
</table>
</div>
</div>

<!-- Hour of Day -->
<h2>Hour of Day (UTC)</h2>
<table>
<tr><th>Hour</th><th>Trades</th><th>Wins</th><th>WR%</th><th>Net P&L</th></tr>
{"".join(f'<tr><td>{h}:00</td><td>{hour_data[h]["t"]}</td><td>{hour_data[h]["w"]}</td><td>{round(hour_data[h]["w"]/hour_data[h]["t"]*100,1)}%</td><td class="{"pos" if hour_data[h]["net"]>=0 else "neg"}">${hour_data[h]["net"]:+,.2f}</td></tr>' for h in sorted(hour_data))}
</table>

<div class="grid2">
<div>
<h2>Worst 10 Days</h2>
<table>
<tr><th>Date</th><th>P&L</th></tr>
{"".join(f'<tr><td>{d}</td><td class="neg">${p:+,.2f}</td></tr>' for d,p in worst10)}
</table>
</div>
<div>
<h2>Best 10 Days</h2>
<table>
<tr><th>Date</th><th>P&L</th></tr>
{"".join(f'<tr><td>{d}</td><td class="pos">${p:+,.2f}</td></tr>' for d,p in best10)}
</table>
</div>
</div>

<!-- Rejection Analysis -->
<h2>Rejection Breakdown (filter activity)</h2>
<table>
<tr><th>Symbol</th><th>Reason</th><th>Count</th></tr>
{rej_rows}
</table>

<!-- Monthly Table -->
<h2>Monthly Breakdown</h2>
<table>
<tr><th>Month</th><th>Trades</th><th>Wins</th><th>Losses</th><th>WR%</th><th>Net P&L</th><th>Balance</th></tr>
"""

    running = BALANCE
    for m in months_sorted:
        d = monthly[m]
        running += d["pnl"]
        wr = round(d["w"] / d["t"] * 100, 1) if d["t"] else 0
        cls = "pos" if d["pnl"] >= 0 else "neg"
        html += (f'<tr><td>{m}</td><td>{d["t"]}</td><td>{d["w"]}</td>'
                 f'<td>{d["t"]-d["w"]}</td><td>{wr}%</td>'
                 f'<td class="{cls}">${d["pnl"]:+,.2f}</td>'
                 f'<td>${running:,.2f}</td></tr>\n')

    html += f"""</table>

<!-- Risk Stats -->
<h2>Risk Stats</h2>
<div class="cards">
<div class="card"><div class="value">${np.mean([t["risk_usd"] for t in all_trades]):.0f}</div><div class="label">Avg Risk/Trade</div></div>
<div class="card"><div class="value">{np.mean([t["bars_held"] for t in all_trades]):.0f} bars</div><div class="label">Avg Duration</div></div>
<div class="card green"><div class="value">${max(t["pnl"] for t in all_trades):,.0f}</div><div class="label">Best Trade</div></div>
<div class="card red"><div class="value">${min(t["pnl"] for t in all_trades):,.0f}</div><div class="label">Worst Trade</div></div>
<div class="card"><div class="value">{np.mean([t["mfe_r"] for t in all_trades]):.1f}R</div><div class="label">Avg MFE</div></div>
<div class="card"><div class="value">{np.mean([t["mae_r"] for t in all_trades]):.1f}R</div><div class="label">Avg MAE</div></div>
</div>

<!-- All Trades -->
<h2>All Trades ({total_t} total)</h2>
<div class="filter-bar">
<button class="active" onclick="filterTrades('ALL')">All</button>
{"".join(f'<button onclick="filterTrades(this.textContent)">{s}</button>' for s in symbols)}
{"".join(f'<button onclick="filterTrades(this.textContent)">{s}</button>' for s in strategies)}
</div>
<div style="max-height:700px;overflow-y:auto;border-radius:8px;">
<table id="tradesTable">
<tr><th>#</th><th>Date</th><th>Symbol</th><th>Strategy</th><th>Dir</th><th>Entry</th><th>SL</th><th>TP</th><th>Lots</th><th>DOL</th><th>Resist.</th><th>Result</th><th>P&L</th><th>R</th><th>MFE</th><th>MAE</th><th>Bars</th><th>Balance</th></tr>
"""

    for i, t in enumerate(all_trades):
        cls = "pos" if t["pnl"] > 0 else "neg"
        badge = badge_cls(t["strategy"])
        icon = "&#10004;" if t["pnl"] > 0 else "&#10008;"
        res_tag = f'<span class="tag tag-{"low" if t.get("liq_resistance","")!="HIGH" else "high"}">{t.get("liq_resistance","?")}</span>'
        dol_str = f'{t["dol_target"]:.5f}' if t.get("dol_target") else "-"
        html += (
            f'<tr data-sym="{t["symbol"]}" data-strat="{t["strategy"]}">'
            f'<td>{i+1}</td><td>{t["entry_time"][:16]}</td><td>{t["symbol"]}</td>'
            f'<td><span class="badge badge-{badge}">{t["strategy"]}</span></td>'
            f'<td>{t["dir"]}</td><td>{t["entry"]}</td><td>{t["sl"]}</td><td>{t["tp"]}</td>'
            f'<td>{t["vol"]:.2f}</td><td style="font-size:11px">{dol_str}</td><td>{res_tag}</td>'
            f'<td class="{cls}">{icon} {t["exit_reason"]}</td>'
            f'<td class="{cls}">${t["pnl"]:+,.2f}</td>'
            f'<td class="{cls}">{t["r_multiple"]:+.2f}R</td>'
            f'<td>{t["mfe_r"]:.1f}R</td><td>{t["mae_r"]:.1f}R</td>'
            f'<td>{t["bars_held"]}</td>'
            f'<td>${t["shared_balance"]:,.2f}</td></tr>\n'
        )

    html += f"""</table></div>

</div>

<script>
// Equity curve
const eqData = [{eq_json}];
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: eqData.map(d => d[0]),
    datasets: [{{
      data: eqData.map(d => d[1]),
      borderColor: '#2962ff',
      backgroundColor: 'rgba(41,98,255,0.1)',
      fill: true, tension: 0.1, pointRadius: 0, borderWidth: 2
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ display: false }},
      y: {{ ticks: {{ color: '#787b86', callback: v => '$'+v.toLocaleString() }}, grid: {{ color: '#2a2e39' }} }}
    }}
  }}
}});

// Monthly P&L
new Chart(document.getElementById('monthlyChart'), {{
  type: 'bar',
  data: {{
    labels: [{monthly_labels}],
    datasets: [{{ data: [{monthly_vals}], backgroundColor: [{monthly_colors}], borderRadius: 4 }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#787b86' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#787b86', callback: v => '$'+v.toLocaleString() }}, grid: {{ color: '#2a2e39' }} }}
    }}
  }}
}});

// Trade filter
function filterTrades(val) {{
  document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('#tradesTable tr[data-sym]').forEach(row => {{
    if (val === 'ALL') {{ row.style.display = ''; return; }}
    const show = row.dataset.sym === val || row.dataset.strat === val;
    row.style.display = show ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    return html


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="Re-pull history from MT5")
    args = ap.parse_args()

    print(f"Loading data for {', '.join(SYMBOLS)}...")
    all_data = load_data(refresh=args.refresh)

    print(f"Running backtest (${BALANCE:,.0f}, all live filters)...")
    trades, equity, rejections = run_backtest(all_data)

    html = build_html(trades, equity, rejections)
    out = HERE / "multi_strategy_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nReport saved: {out}")
    print(f"Open: file:///{str(out).replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
