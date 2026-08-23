"""Backtest engine for the SMC bot.

Calls the same SMCStrategy functions the live bot calls — the strategy is
never reimplemented here, only driven.

    python engine.py                      # baseline, signals as generated
    python engine.py --invert             # signals traded backwards
    python engine.py --rrr 1.5            # a different target multiple
    python engine.py --sweep              # expectancy across RRR, both modes
    python engine.py --refresh            # re-pull history from MT5

Fidelity rules:
  * Decisions on the close of bar t, execution at the open of bar t+1 — no
    lookahead.
  * MT5 candles are BID.  A long fills at ask and exits on bid; a short fills
    at bid and exits on ask, so the spread asymmetry is carried through.
  * One position at a time, exactly like main.py.
  * A bar touching both stop and target is scored as the loss.

Not modelled: commission, swap, slippage beyond the spread, intrabar tick order.
"""

import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    BLOCKED_DAYS, HTF, HTF_CANDLES_LOOKBACK, LTF, LTF_CANDLES_LOOKBACK,
    MAX_LTF_WAIT_CANDLES,
    MAX_RISK_PCT, MAX_RISK_USD, NIGHT_END_HOUR, NIGHT_START_HOUR,
    PARTIAL_CLOSE_PCT, PARTIAL_TRIGGER_PCT, RISK_PERCENT,
    TIME_STOP_BARS, TIME_STOP_MIN_PCT,
    TRAILING_STOP_TRIGGER_PCT, TRAILING_STOP_DISTANCE_PCT,
    USE_RISK_BASED_LOT,
    SKIP_WEEKENDS, STOP_MODE, SYMBOL, TREND_EMA_PERIOD, USE_TREND_FILTER,
    LIQUIDITY_TF, SWING_STRENGTH, MIN_RRR_LIQUIDITY,
)
from strategy import SMCStrategy  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_history.pkl"

START_BALANCE = 100_000.0
VOLUME = 1.0
M5_BARS = 100_000    # ~11 months of 5m data
H1_BARS = 10_000     # ~13 months of 1h data
M15_BARS = 50_000    # ~6 months of 15m data (for liquidity levels)


# ---------------------------------------------------------------- data
def load_data(refresh: bool = False):
    """Returns (m5, h1, m15_liq, contract_size, point), cached on disk."""
    if CACHE.exists() and not refresh:
        blob = pd.read_pickle(CACHE)
        # Support old cache format gracefully
        if "m5" in blob:
            return blob["m5"], blob["h1"], blob["m15_liq"], blob["contract"], blob["point"]

    import MetaTrader5 as mt5
    from mt5_connector import MT5Connector

    c = MT5Connector(symbol=SYMBOL, magic_number=0)
    if not c.initialize():
        raise RuntimeError("MT5 is not reachable — start the terminal and retry.")
    info = c.symbol_info()
    contract, point = info.trade_contract_size, info.point

    frames = {}
    for key, tf, n in (("m5", LTF, M5_BARS), ("h1", HTF, H1_BARS),
                        ("m15_liq", LIQUIDITY_TF, M15_BARS)):
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        frames[key] = df
    c.shutdown()

    pd.to_pickle({**frames, "contract": contract, "point": point}, CACHE)
    print(f"cached {len(frames['m5'])} M5 / {len(frames['h1'])} H1 / "
          f"{len(frames['m15_liq'])} M15 bars -> {CACHE.name}")
    return frames["m5"], frames["h1"], frames["m15_liq"], contract, point


# ---------------------------------------------------------------- engine
_HTF_CACHE = {}


def run(m5, h1, m15_liq, contract, point, rrr=3.0, invert=False,
        one_shot_per_zone=False, no_poi=False, no_trend=False,
        stop_mode=None, buffer_atr=0.5, use_liq_tp=True,
        breakeven_r=0.0, use_partial=True, progress=False):
    """Replays the strategy bar by bar and returns {summary, trades, equity}.

    breakeven_r: move stop to entry once price reaches this many R in profit.
                 0.0 disables the feature.
    use_partial: at 80% of TP, close 80% of position, move SL to entry,
                 move TP to next liquidity level.
    """
    if stop_mode is None:
        stop_mode = STOP_MODE
    from strategy import ZonePOI
    any_zones = (ZonePOI(type="BULLISH", top=1e12, bottom=-1e12),
                 ZonePOI(type="BEARISH", top=1e12, bottom=-1e12))

    h1_t, m5_t = h1["time"].values, m5["time"].values
    m15_t = m15_liq["time"].values
    # Map each M5 bar to the last closed H1 bar
    h_idx = np.searchsorted(h1_t, m5_t - np.timedelta64(60, "m"), side="right") - 1
    # Map each M5 bar to the last closed M15 bar (for liquidity)
    liq_idx = np.searchsorted(m15_t, m5_t - np.timedelta64(15, "m"), side="right") - 1

    o = m5["open"].to_numpy(float)
    hi = m5["high"].to_numpy(float)
    lo = m5["low"].to_numpy(float)
    cl = m5["close"].to_numpy(float)
    spread = m5["spread"].to_numpy(float) * point
    times = m5["time"]

    start = LTF_CANDLES_LOOKBACK
    while start < len(m5) and h_idx[start] < HTF_CANDLES_LOOKBACK:
        start += 1

    trades, equity = [], []
    balance = START_BALANCE
    poi, poi_h, trend = None, -1, "NEUTRAL"
    watch_key, watch_start, abandoned = None, None, False
    dead_zones = set()
    position = None
    t_start = _time.time()

    _weekdays = pd.to_datetime(m5["time"]).dt.weekday.to_numpy()
    _hours = pd.to_datetime(m5["time"]).dt.hour.to_numpy()

    for t in range(start, len(m5) - 1):
        if progress and t % 10000 == 0:
            print(f"  ... {t}/{len(m5)} ({_time.time()-t_start:.0f}s, {len(trades)} trades)")

        # ---- manage an open position ----
        if position is not None:
            p = position
            vol = p.get("vol", VOLUME)

            # Partial close: at 80% of TP distance, close 80%, SL→entry, TP→next liq
            if use_partial and not p.get("partial_done"):
                tp_dist = abs(p["tp"] - p["entry"])
                trigger_dist = tp_dist * PARTIAL_TRIGGER_PCT
                if p["dir"] == "BUY":
                    triggered = hi[t] >= p["entry"] + trigger_dist
                else:
                    triggered = lo[t] + spread[t] <= p["entry"] - trigger_dist

                if triggered:
                    # Book profit on the closed portion
                    close_vol = vol * PARTIAL_CLOSE_PCT
                    remain_vol = vol - close_vol
                    partial_px = (p["entry"] + trigger_dist if p["dir"] == "BUY"
                                  else p["entry"] - trigger_dist)
                    partial_pnl = ((partial_px - p["entry"]) if p["dir"] == "BUY"
                                   else (p["entry"] - partial_px)) * close_vol * contract
                    balance += partial_pnl
                    p["partial_pnl"] = round(partial_pnl, 2)
                    p["partial_done"] = True
                    p["vol"] = remain_vol

                    # Move SL to entry
                    p["sl"] = p["entry"]

                    # Move TP to next liquidity level
                    li = liq_idx[t]
                    if li >= 0:
                        liq_slice = m15_liq.iloc[max(0, li - 200): li + 1]
                        next_liq = SMCStrategy.find_next_liquidity(
                            liq_slice, p["dir"], p["tp"],
                            strength=SWING_STRENGTH, use_closed_candles=False)
                        if next_liq is not None:
                            p["tp"] = round(next_liq, 2)
                            p["tp_extended"] = True

            # Breakeven (standalone, if no partial close)
            if breakeven_r > 0 and not p.get("be_moved") and not p.get("partial_done"):
                trigger_dist = p["risk_px"] * breakeven_r
                if p["dir"] == "BUY":
                    if hi[t] >= p["entry"] + trigger_dist:
                        p["sl"] = p["entry"]
                        p["be_moved"] = True
                else:
                    if lo[t] + spread[t] <= p["entry"] - trigger_dist:
                        p["sl"] = p["entry"]
                        p["be_moved"] = True

            # Trailing stop: once price reaches X% of TP, trail SL behind peak
            if TRAILING_STOP_TRIGGER_PCT > 0 and not p.get("partial_done"):
                tp_dist = abs(p["tp"] - p["entry"])
                trail_trigger = tp_dist * TRAILING_STOP_TRIGGER_PCT
                trail_dist = tp_dist * TRAILING_STOP_DISTANCE_PCT

                if p["dir"] == "BUY":
                    peak = p.get("peak_price", p["entry"])
                    peak = max(peak, hi[t])
                    p["peak_price"] = peak
                    if peak >= p["entry"] + trail_trigger:
                        new_sl = peak - trail_dist
                        if new_sl > p["sl"]:
                            p["sl"] = round(new_sl, 2)
                else:
                    trough = p.get("trough_price", p["entry"])
                    trough = min(trough, lo[t] + spread[t])
                    p["trough_price"] = trough
                    if trough <= p["entry"] - trail_trigger:
                        new_sl = trough + trail_dist
                        if new_sl < p["sl"]:
                            p["sl"] = round(new_sl, 2)

            # Time stop: if trade hasn't progressed enough, force close
            if TIME_STOP_BARS > 0 and not p.get("partial_done"):
                bars_held = t - p["entry_bar"]
                if bars_held >= TIME_STOP_BARS:
                    tp_dist = abs(p["tp"] - p["entry"])
                    if p["dir"] == "BUY":
                        best_move = hi[t] - p["entry"]
                    else:
                        best_move = p["entry"] - (lo[t] + spread[t])
                    peak_move = p.get("peak_price", p["entry"]) - p["entry"] if p["dir"] == "BUY" \
                        else p["entry"] - p.get("trough_price", p["entry"])
                    pct_reached = peak_move / tp_dist * 100 if tp_dist > 0 else 0
                    if pct_reached < TIME_STOP_MIN_PCT:
                        # Force close at current price
                        exit_px = cl[t] if p["dir"] == "BUY" else cl[t] + spread[t]
                        pnl = ((exit_px - p["entry"]) if p["dir"] == "BUY"
                               else (p["entry"] - exit_px)) * vol * contract
                        balance += pnl
                        total_pnl = pnl + p.get("partial_pnl", 0)
                        p.update(
                            exit_time=str(times.iloc[t]), exit_price=round(exit_px, 2),
                            exit_reason="TIME_STOP",
                            bars_held=t - p.pop("entry_bar"),
                            pnl=round(total_pnl, 2),
                            r_multiple=round(total_pnl / p["risk_usd"], 2) if p["risk_usd"] else 0,
                            balance=round(balance, 2),
                        )
                        trades.append(p)
                        equity.append({"time": p["exit_time"], "balance": round(balance, 2)})
                        position, watch_key, abandoned = None, None, False
                        if balance <= 0:
                            break
                        continue

            if p["dir"] == "BUY":
                hit_sl, hit_tp = lo[t] <= p["sl"], hi[t] >= p["tp"]
            else:
                hit_sl = hi[t] + spread[t] >= p["sl"]
                hit_tp = lo[t] + spread[t] <= p["tp"]

            if hit_sl or hit_tp:
                exit_px = p["sl"] if hit_sl else p["tp"]
                remaining_pnl = ((exit_px - p["entry"]) if p["dir"] == "BUY"
                                 else (p["entry"] - exit_px)) * vol * contract
                balance += remaining_pnl
                total_pnl = remaining_pnl + p.get("partial_pnl", 0)
                p.update(
                    exit_time=str(times.iloc[t]), exit_price=round(exit_px, 2),
                    exit_reason="SL" if hit_sl else "TP",
                    bars_held=t - p.pop("entry_bar"),
                    pnl=round(total_pnl, 2),
                    r_multiple=round(total_pnl / p["risk_usd"], 2) if p["risk_usd"] else 0,
                    balance=round(balance, 2),
                )
                if hit_sl and one_shot_per_zone:
                    dead_zones.add(p["zone"])
                trades.append(p)
                equity.append({"time": p["exit_time"], "balance": round(balance, 2)})
                position, watch_key, abandoned = None, None, False
                if balance <= 0:
                    break
            continue

        # ---- weekend & night filter ----
        if SKIP_WEEKENDS and _weekdays[t] >= 5:
            continue
        h = _hours[t]
        if NIGHT_START_HOUR > NIGHT_END_HOUR:
            if h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR:
                continue
        elif NIGHT_START_HOUR <= h < NIGHT_END_HOUR:
            continue

        # ---- ablation: no HTF zone, just the LTF trigger ----
        if no_poi:
            ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
            setup = None
            for z in any_zones:
                setup = SMCStrategy.check_ltf_confirmation(
                    ltf, z, rrr, use_closed_candles=False,
                    stop_mode=stop_mode, buffer_atr=buffer_atr)
                if setup is not None:
                    break
            if setup is None:
                continue
            zone = ("ANY", 0.0, 0.0)
            trend = "NEUTRAL"
            # Build M15 slice for liquidity
            li = liq_idx[t]
            m15_slice = m15_liq.iloc[max(0, li - 200): li + 1] if use_liq_tp and li >= 0 else None
            position = _open(setup, invert, rrr, o, spread, t + 1, times,
                             trend, "ANY", zone, len(trades), contract,
                             m15_slice=m15_slice, balance=balance)
            continue

        # ---- refresh the HTF view once per closed H1 bar ----
        if h_idx[t] != poi_h:
            poi_h = h_idx[t]
            use_trend = USE_TREND_FILTER and not no_trend
            ck = (poi_h, use_trend)
            if ck in _HTF_CACHE:
                trend, poi = _HTF_CACHE[ck]
            else:
                wh = h1.iloc[max(0, poi_h - HTF_CANDLES_LOOKBACK + 1): poi_h + 1]
                trend = (SMCStrategy.get_htf_trend(wh, TREND_EMA_PERIOD,
                                                   use_closed_candles=False)
                         if use_trend else "NEUTRAL")
                poi = SMCStrategy.detect_htf_poi(
                    wh, use_trend_filter=use_trend,
                    ema_period=TREND_EMA_PERIOD, use_closed_candles=False)
                _HTF_CACHE[ck] = (trend, poi)
            if poi is None:
                watch_key, abandoned = None, False

        if poi is None:
            continue

        zone = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
        if zone in dead_zones:
            continue

        mid = cl[t] + spread[t] / 2.0
        if not (poi.bottom <= mid <= poi.top
                or (lo[t] <= poi.top and hi[t] >= poi.bottom)):
            watch_key, abandoned = None, False
            continue

        ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
        if not SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
            watch_key, abandoned = None, False
            continue

        if zone != watch_key:
            watch_key, watch_start, abandoned = zone, t, False
        if abandoned:
            continue
        if (t - watch_start + 1) > MAX_LTF_WAIT_CANDLES:
            abandoned = True
            continue

        # ---- consolidation filter ----
        if SMCStrategy.is_consolidating(ltf, use_closed_candles=False):
            continue

        setup = SMCStrategy.check_ltf_confirmation(
            ltf, poi, rrr, use_closed_candles=False,
            stop_mode=stop_mode, buffer_atr=buffer_atr)
        if setup is None:
            continue

        # Build M15 slice for liquidity TP
        li = liq_idx[t]
        m15_slice = m15_liq.iloc[max(0, li - 200): li + 1] if use_liq_tp and li >= 0 else None

        position = _open(setup, invert, rrr, o, spread, t + 1, times, trend,
                         poi.type, zone, len(trades), contract,
                         m15_slice=m15_slice, balance=balance)

    for tr in trades:
        tr.pop("zone", None)
    return summarize(trades, equity, balance, times, start, rrr, invert,
                     len(m5), m5, point, _time.time() - t_start)


def _open(setup, invert, rrr, o, spread, nt, times, trend, poi_type, zone,
          n_done, contract, m15_slice=None, balance=None):
    """Turns a confirmed setup into a position filled at bar `nt`'s open."""
    if invert:
        setup = SMCStrategy.invert(setup, rrr)

    entry = o[nt] + spread[nt] if setup["direction"] == "BUY" else o[nt]
    sl = setup["sl"]
    risk = abs(entry - sl)
    if risk <= 0:
        return None

    # %-based position sizing: calculate volume from risk
    if USE_RISK_BASED_LOT and balance is not None and balance > 0:
        risk_amount = balance * RISK_PERCENT / 100.0
        pnl_per_lot = risk * contract
        if pnl_per_lot <= 0:
            return None
        vol = risk_amount / pnl_per_lot
        # Floor to volume step (0.01)
        vol = max(0.01, int(vol * 100) / 100.0)
    else:
        vol = VOLUME

    # Risk cap
    risk_usd = risk * vol * contract
    if MAX_RISK_PCT > 0 and balance is not None:
        max_allowed = balance * MAX_RISK_PCT / 100
        if risk_usd > max_allowed:
            return None
    elif MAX_RISK_USD > 0 and risk_usd > MAX_RISK_USD:
        return None

    # Try liquidity-based TP first
    tp = None
    tp_source = "fixed"
    if m15_slice is not None and len(m15_slice) > 0:
        liq = SMCStrategy.find_nearest_liquidity(
            m15_slice, setup["direction"], entry,
            strength=SWING_STRENGTH, use_closed_candles=False)
        if liq is not None:
            liq_dist = abs(liq - entry)
            liq_rrr = liq_dist / risk if risk > 0 else 0
            if liq_rrr >= MIN_RRR_LIQUIDITY:
                tp = liq
                tp_source = "liquidity"
            # Liquidity too close — fall through to fixed RRR

    if tp is None:
        tp = entry + risk * rrr if setup["direction"] == "BUY" else entry - risk * rrr

    return {
        "n": n_done + 1, "dir": setup["direction"], "poi": poi_type,
        "trend": trend, "entry_time": str(times.iloc[nt]),
        "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
        "tp_source": tp_source,
        "risk_px": round(risk, 2),
        "risk_usd": round(risk * vol * contract, 2),
        "vol": vol,
        "spread_px": round(spread[nt], 2), "entry_bar": nt, "zone": zone,
    }


def summarize(trades, equity, balance, times, start, rrr, invert,
              n_bars, m5, point, runtime):
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = -sum(t["pnl"] for t in losses)

    peak, max_dd, max_dd_pct = START_BALANCE, 0.0, 0.0
    for e in equity:
        peak = max(peak, e["balance"])
        if peak - e["balance"] > max_dd:
            max_dd = peak - e["balance"]
            max_dd_pct = max_dd / peak * 100

    streak = best_w = best_l = 0
    for t in trades:
        if t["pnl"] > 0:
            streak = streak + 1 if streak > 0 else 1
            best_w = max(best_w, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            best_l = min(best_l, streak)

    n = len(trades)
    return {
        "summary": {
            "mode": "INVERTED" if invert else "AS-SIGNALLED",
            "symbol": SYMBOL, "rrr": rrr,
            "period_from": str(times.iloc[start]), "period_to": str(times.iloc[-1]),
            "days": round((times.iloc[-1] - times.iloc[start]).total_seconds() / 86400, 1),
            "m5_bars": n_bars, "start_balance": START_BALANCE, "volume": VOLUME,
            "spread_points": int(np.median(m5["spread"].to_numpy())),
            "end_balance": round(balance, 2),
            "net_pnl": round(balance - START_BALANCE, 2),
            "return_pct": round((balance / START_BALANCE - 1) * 100, 2),
            "trades": n, "wins": len(wins), "losses": len(losses),
            "win_rate": round(len(wins) / n * 100, 1) if n else 0,
            "profit_factor": round(gross_w / gross_l, 2) if gross_l else None,
            "avg_win": round(gross_w / len(wins), 2) if wins else 0,
            "avg_loss": round(-gross_l / len(losses), 2) if losses else 0,
            "expectancy": round((balance - START_BALANCE) / n, 2) if n else 0,
            "expectancy_r": round(sum(t["r_multiple"] for t in trades) / n, 3) if n else 0,
            "total_r": round(sum(t["r_multiple"] for t in trades), 1),
            "best_trade": round(max((t["pnl"] for t in trades), default=0), 2),
            "worst_trade": round(min((t["pnl"] for t in trades), default=0), 2),
            "max_drawdown": round(max_dd, 2), "max_drawdown_pct": round(max_dd_pct, 2),
            "longest_win_streak": best_w, "longest_loss_streak": abs(best_l),
            "avg_bars_held": round(np.mean([t["bars_held"] for t in trades]), 1) if n else 0,
            "avg_risk_usd": round(np.mean([t["risk_usd"] for t in trades]), 2) if n else 0,
            "runtime_s": round(runtime, 1),
        },
        "trades": trades,
        "equity": equity,
    }


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrr", type=float, default=3.0)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--one-shot", action="store_true",
                    help="a zone that produced a loss is not traded again")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ablate", action="store_true",
                    help="measure what each layer of the strategy contributes")
    ap.add_argument("--stops", action="store_true",
                    help="compare stop placement rules across RRR")
    ap.add_argument("--stop-mode", choices=("window", "swing", "zone"),
                    default="window")
    ap.add_argument("--buffer-atr", type=float, default=0.5)
    ap.add_argument("--no-poi", action="store_true")
    ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--no-liq-tp", action="store_true",
                    help="use fixed RRR instead of liquidity TP")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    m5, h1, m15_liq, contract, point = load_data(refresh=args.refresh)
    print(f"{len(m5)} M5 bars  {m5['time'].iloc[0]} .. {m5['time'].iloc[-1]}")

    use_liq = not args.no_liq_tp

    if args.sweep:
        print(f"\n{'RRR':>5} {'mode':<14}{'trades':>7}{'win%':>7}{'totR':>7}"
              f"{'expR':>7}{'net$':>9}{'PF':>6}{'maxDD%':>8}")
        print("-" * 68)
        rows = []
        for invert in (False, True):
            for rrr in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
                s = run(m5, h1, m15_liq, contract, point, rrr=rrr,
                        invert=invert, use_liq_tp=use_liq)["summary"]
                rows.append(s)
                print(f"{rrr:>5.1f} {s['mode']:<14}{s['trades']:>7}"
                      f"{s['win_rate']:>7.1f}{s['total_r']:>7.1f}"
                      f"{s['expectancy_r']:>7.3f}{s['net_pnl']:>9.2f}"
                      f"{str(s['profit_factor']):>6}{s['max_drawdown_pct']:>8.2f}")
            print("-" * 68)
        (HERE / "sweep_rrr.json").write_text(json.dumps(rows, indent=1), "utf-8")
        return

    if args.stops:
        print(f"\n{'stop rule':<12}{'RRR':>5} {'mode':<14}{'trades':>7}{'win%':>7}"
              f"{'totR':>7}{'expR':>7}{'net$':>9}{'avgRisk$':>10}{'PF':>6}")
        print("-" * 84)
        rows = []
        for mode in ("window", "swing", "zone"):
            for rrr in (1.5, 2.0, 2.5, 3.0):
                s = run(m5, h1, m15_liq, contract, point, rrr=rrr, invert=True,
                        stop_mode=mode, buffer_atr=args.buffer_atr,
                        use_liq_tp=use_liq)["summary"]
                s["stop_mode"] = mode
                rows.append(s)
                print(f"{mode:<12}{rrr:>5.1f} {s['mode']:<14}{s['trades']:>7}"
                      f"{s['win_rate']:>7.1f}{s['total_r']:>7.1f}"
                      f"{s['expectancy_r']:>7.3f}{s['net_pnl']:>9.2f}"
                      f"{s['avg_risk_usd']:>10.2f}{str(s['profit_factor']):>6}")
            print("-" * 84)
        (HERE / "stop_rules.json").write_text(json.dumps(rows, indent=1), "utf-8")
        return

    if args.ablate:
        layers = [
            ("full strategy",        dict()),
            ("no trend filter",      dict(no_trend=True)),
            ("no HTF zone",          dict(no_poi=True)),
            ("no zone, no trend",    dict(no_poi=True, no_trend=True)),
            ("one shot per zone",    dict(one_shot_per_zone=True)),
        ]
        print(f"\n{'layer':<20}{'mode':<14}{'trades':>7}{'win%':>7}{'totR':>7}"
              f"{'expR':>7}{'net$':>9}{'PF':>6}")
        print("-" * 77)
        rows = []
        for label, opts in layers:
            for invert in (False, True):
                s = run(m5, h1, m15_liq, contract, point, rrr=args.rrr,
                        invert=invert, use_liq_tp=use_liq, **opts)["summary"]
                s["layer"] = label
                rows.append(s)
                print(f"{label:<20}{s['mode']:<14}{s['trades']:>7}"
                      f"{s['win_rate']:>7.1f}{s['total_r']:>7.1f}"
                      f"{s['expectancy_r']:>7.3f}{s['net_pnl']:>9.2f}"
                      f"{str(s['profit_factor']):>6}")
            print("-" * 77)
        (HERE / "ablation.json").write_text(json.dumps(rows, indent=1), "utf-8")
        return

    res = run(m5, h1, m15_liq, contract, point, rrr=args.rrr, invert=args.invert,
              one_shot_per_zone=args.one_shot, no_poi=args.no_poi,
              no_trend=args.no_trend, use_liq_tp=use_liq, progress=True)
    out = HERE / (args.out or
                  f"bt_{'inv' if args.invert else 'sig'}_rrr{args.rrr:g}.json")
    out.write_text(json.dumps(res, indent=1), "utf-8")
    for k, v in res["summary"].items():
        print(f"{k:>20}: {v}")
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
