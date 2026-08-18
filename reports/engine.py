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
    HTF, HTF_CANDLES_LOOKBACK, LTF, LTF_CANDLES_LOOKBACK, MAX_LTF_WAIT_CANDLES,
    SKIP_WEEKENDS, SYMBOL, TREND_EMA_PERIOD, USE_TREND_FILTER,
)
from strategy import SMCStrategy  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_history.pkl"

START_BALANCE = 1000.0
VOLUME = 0.05
M1_BARS = 135_000   # ~90 days of 1m data (3 months)
M15_BARS = 18_000   # ~90 days of 15m data


# ---------------------------------------------------------------- data
def load_data(refresh: bool = False):
    """Returns (m1, m15, contract_size, point), cached on disk after the first pull."""
    if CACHE.exists() and not refresh:
        blob = pd.read_pickle(CACHE)
        return blob["m1"], blob["m15"], blob["contract"], blob["point"]

    import MetaTrader5 as mt5
    from mt5_connector import MT5Connector

    c = MT5Connector(symbol=SYMBOL, magic_number=0)
    if not c.initialize():
        raise RuntimeError("MT5 is not reachable — start the terminal and retry.")
    info = c.symbol_info()
    contract, point = info.trade_contract_size, info.point

    frames = {}
    for key, tf, n in (("m1", LTF, M1_BARS), ("m15", HTF, M15_BARS)):
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        frames[key] = df
    c.shutdown()

    pd.to_pickle({**frames, "contract": contract, "point": point}, CACHE)
    print(f"cached {len(frames['m1'])} M1 / {len(frames['m15'])} M15 bars -> {CACHE.name}")
    return frames["m1"], frames["m15"], contract, point


# ---------------------------------------------------------------- engine
# The HTF view depends only on which M15 bar has closed — never on RRR, on the
# direction mode, or on whether a position is open. Caching it lets a parameter
# sweep reuse one pass of the expensive backward scan across every run.
_HTF_CACHE = {}


def run(m1, m15, contract, point, rrr=3.0, invert=False,
        one_shot_per_zone=False, no_poi=False, no_trend=False,
        stop_mode="window", buffer_atr=0.5, progress=False):
    """Replays the strategy bar by bar and returns {summary, trades, equity}.

    `no_poi` and `no_trend` are ablations: they strip the 15m zone layer and
    the EMA trend filter respectively, to measure how much either contributes.
    """
    from strategy import ZonePOI
    any_zones = (ZonePOI(type="BULLISH", top=1e12, bottom=-1e12),
                 ZonePOI(type="BEARISH", top=1e12, bottom=-1e12))
    m15_t, m1_t = m15["time"].values, m1["time"].values
    h_idx = np.searchsorted(m15_t, m1_t - np.timedelta64(15, "m"), side="right") - 1

    o = m1["open"].to_numpy(float)
    hi = m1["high"].to_numpy(float)
    lo = m1["low"].to_numpy(float)
    cl = m1["close"].to_numpy(float)
    spread = m1["spread"].to_numpy(float) * point
    times = m1["time"]

    start = LTF_CANDLES_LOOKBACK
    while start < len(m1) and h_idx[start] < HTF_CANDLES_LOOKBACK:
        start += 1

    trades, equity = [], []
    balance = START_BALANCE
    poi, poi_h, trend = None, -1, "NEUTRAL"
    watch_key, watch_start, abandoned = None, None, False
    dead_zones = set()
    position = None
    t_start = _time.time()

    # Pre-compute weekend mask (Saturday=5, Sunday=6) so the inner loop is cheap.
    _weekdays = pd.to_datetime(m1["time"]).dt.weekday.to_numpy()

    for t in range(start, len(m1) - 1):
        if progress and t % 10000 == 0:
            print(f"  ... {t}/{len(m1)} ({_time.time()-t_start:.0f}s, {len(trades)} trades)")

        # ---- manage an open position ----
        if position is not None:
            p = position
            if p["dir"] == "BUY":
                hit_sl, hit_tp = lo[t] <= p["sl"], hi[t] >= p["tp"]
            else:
                hit_sl = hi[t] + spread[t] >= p["sl"]
                hit_tp = lo[t] + spread[t] <= p["tp"]

            if hit_sl or hit_tp:
                exit_px = p["sl"] if hit_sl else p["tp"]
                pnl = ((exit_px - p["entry"]) if p["dir"] == "BUY"
                       else (p["entry"] - exit_px)) * VOLUME * contract
                balance += pnl
                p.update(
                    exit_time=str(times.iloc[t]), exit_price=round(exit_px, 2),
                    exit_reason="SL" if hit_sl else "TP",
                    bars_held=t - p.pop("entry_bar"), pnl=round(pnl, 2),
                    r_multiple=round(pnl / p["risk_usd"], 2) if p["risk_usd"] else 0,
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

        # ---- weekend filter: no new entries on Saturday/Sunday ----
        if SKIP_WEEKENDS and _weekdays[t] >= 5:
            continue

        # ---- ablation: no 15m zone at all, just the 1m trigger ----
        if no_poi:
            ltf = m1.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
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
            position = _open(setup, invert, rrr, o, spread, t + 1, times,
                             trend, "ANY", zone, len(trades), contract)
            continue

        # ---- refresh the HTF view once per closed M15 bar ----
        if h_idx[t] != poi_h:
            poi_h = h_idx[t]
            use_trend = USE_TREND_FILTER and not no_trend
            ck = (poi_h, use_trend)
            if ck in _HTF_CACHE:
                trend, poi = _HTF_CACHE[ck]
            else:
                w15 = m15.iloc[max(0, poi_h - HTF_CANDLES_LOOKBACK + 1): poi_h + 1]
                trend = (SMCStrategy.get_htf_trend(w15, TREND_EMA_PERIOD,
                                                   use_closed_candles=False)
                         if use_trend else "NEUTRAL")
                poi = SMCStrategy.detect_htf_poi(
                    w15, use_trend_filter=use_trend,
                    ema_period=TREND_EMA_PERIOD, use_closed_candles=False)
                _HTF_CACHE[ck] = (trend, poi)
            if poi is None:
                watch_key, abandoned = None, False

        if poi is None:
            continue

        zone = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
        if zone in dead_zones:
            continue

        # Cheap scalar pre-filter — the authoritative check is is_zone_in_play
        # below; this only avoids building a DataFrame slice 50k times.
        mid = cl[t] + spread[t] / 2.0
        if not (poi.bottom <= mid <= poi.top
                or (lo[t] <= poi.top and hi[t] >= poi.bottom)):
            watch_key, abandoned = None, False
            continue

        ltf = m1.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
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

        setup = SMCStrategy.check_ltf_confirmation(
            ltf, poi, rrr, use_closed_candles=False,
            stop_mode=stop_mode, buffer_atr=buffer_atr)
        if setup is None:
            continue

        position = _open(setup, invert, rrr, o, spread, t + 1, times, trend,
                         poi.type, zone, len(trades), contract)

    for tr in trades:
        tr.pop("zone", None)
    return summarize(trades, equity, balance, times, start, rrr, invert,
                     len(m1), m1, point, _time.time() - t_start)


def _open(setup, invert, rrr, o, spread, nt, times, trend, poi_type, zone,
          n_done, contract):
    """Turns a confirmed setup into a position filled at bar `nt`'s open."""
    if invert:
        setup = SMCStrategy.invert(setup, rrr)

    entry = o[nt] + spread[nt] if setup["direction"] == "BUY" else o[nt]
    sl = setup["sl"]
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    tp = entry + risk * rrr if setup["direction"] == "BUY" else entry - risk * rrr

    return {
        "n": n_done + 1, "dir": setup["direction"], "poi": poi_type,
        "trend": trend, "entry_time": str(times.iloc[nt]),
        "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
        "risk_px": round(risk, 2),
        "risk_usd": round(risk * VOLUME * contract, 2),
        "spread_px": round(spread[nt], 2), "entry_bar": nt, "zone": zone,
    }


def summarize(trades, equity, balance, times, start, rrr, invert,
              n_bars, m1, point, runtime):
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
            "m1_bars": n_bars, "start_balance": START_BALANCE, "volume": VOLUME,
            "spread_points": int(np.median(m1["spread"].to_numpy())),
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
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    m1, m15, contract, point = load_data(refresh=args.refresh)
    print(f"{len(m1)} M1 bars  {m1['time'].iloc[0]} .. {m1['time'].iloc[-1]}")

    if args.sweep:
        print(f"\n{'RRR':>5} {'mode':<14}{'trades':>7}{'win%':>7}{'totR':>7}"
              f"{'expR':>7}{'net$':>9}{'PF':>6}{'maxDD%':>8}")
        print("-" * 68)
        rows = []
        for invert in (False, True):
            for rrr in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
                s = run(m1, m15, contract, point, rrr=rrr, invert=invert)["summary"]
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
                s = run(m1, m15, contract, point, rrr=rrr, invert=True,
                        stop_mode=mode, buffer_atr=args.buffer_atr)["summary"]
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
            ("no 15m zone",          dict(no_poi=True)),
            ("no zone, no trend",    dict(no_poi=True, no_trend=True)),
            ("one shot per zone",    dict(one_shot_per_zone=True)),
        ]
        print(f"\n{'layer':<20}{'mode':<14}{'trades':>7}{'win%':>7}{'totR':>7}"
              f"{'expR':>7}{'net$':>9}{'PF':>6}")
        print("-" * 77)
        rows = []
        for label, opts in layers:
            for invert in (False, True):
                s = run(m1, m15, contract, point, rrr=args.rrr,
                        invert=invert, **opts)["summary"]
                s["layer"] = label
                rows.append(s)
                print(f"{label:<20}{s['mode']:<14}{s['trades']:>7}"
                      f"{s['win_rate']:>7.1f}{s['total_r']:>7.1f}"
                      f"{s['expectancy_r']:>7.3f}{s['net_pnl']:>9.2f}"
                      f"{str(s['profit_factor']):>6}")
            print("-" * 77)
        (HERE / "ablation.json").write_text(json.dumps(rows, indent=1), "utf-8")
        return

    res = run(m1, m15, contract, point, rrr=args.rrr, invert=args.invert,
              one_shot_per_zone=args.one_shot, no_poi=args.no_poi,
              no_trend=args.no_trend, progress=True)
    out = HERE / (args.out or
                  f"bt_{'inv' if args.invert else 'sig'}_rrr{args.rrr:g}.json")
    out.write_text(json.dumps(res, indent=1), "utf-8")
    for k, v in res["summary"].items():
        print(f"{k:>20}: {v}")
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
