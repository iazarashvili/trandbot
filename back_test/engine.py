"""Multi-symbol backtest engine for the SMC trading bot.

Drives the same SMCStrategy functions the live engine uses — strategy logic is
never reimplemented here, only driven.  Each symbol gets its own independent
$10,000 starting balance, its own SymbolConfig, and runs all three active
strategies (AMD, Silver Bullet, Sweep+FVG) with first-match-wins semantics.

Fidelity rules (same as reports/engine.py and the live engine):
  * Decisions on the close of bar t, execution at the open of bar t+1.
  * MT5 candles are BID.  Long fills at ask (open+spread); short fills at bid.
  * One position per symbol at a time.
  * A bar touching both SL and TP is scored as the loss.
  * Partial close, breakeven, and liquidity TP are driven by SymbolConfig.

Usage:
    python -m back_test.engine                           # all 4 symbols
    python -m back_test.engine --symbols BTCUSD XAUUSD   # subset
    python -m back_test.engine --refresh                 # re-pull history
"""

import argparse
import json
import sys
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so config imports work.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# config.py imports MetaTrader5 at module level, and SymbolConfig.load() imports
# config_<symbol>.py which does `from config import *`.  We need MT5 importable
# (though not necessarily connected) before loading configs.
import MetaTrader5 as mt5  # noqa: E402

from core.symbol_context import SymbolConfig  # noqa: E402
from strategy import SMCStrategy, ZonePOI  # noqa: E402
from back_test.data import load_symbol_data  # noqa: E402

HERE = Path(__file__).resolve().parent

DEFAULT_SYMBOLS = ["BTCUSD", "GBPUSD"]
START_BALANCE = 10_000.0

# Lookback constants (from config.py defaults)
HTF_CANDLES_LOOKBACK = 200
LTF_CANDLES_LOOKBACK = 100
MAX_LTF_WAIT_CANDLES = 15
SWING_STRENGTH = 3
MIN_RRR_LIQUIDITY = 0.5


# =====================================================================
# Logging helpers
# =====================================================================

def _ts(times, t: int) -> str:
    """Format a timestamp for log output."""
    return str(times.iloc[t])


# =====================================================================
# Engine
# =====================================================================

def run_symbol(
    symbol: str,
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    m15: pd.DataFrame,
    contract_size: float,
    point: float,
    cfg: SymbolConfig,
) -> dict:
    """Run a full bar-by-bar backtest for one symbol.

    Returns dict with keys: summary, trades, equity, logs.
    """
    logs: list[str] = []

    def log(msg: str):
        logs.append(msg)

    log(f"=== BACKTEST START: {symbol} ===")
    log(f"Config: stop_mode={cfg.stop_mode} rrr={cfg.rrr} "
        f"use_sweep_filter={cfg.use_sweep_filter} "
        f"blocked_days={cfg.blocked_days} night={cfg.night_start}-{cfg.night_end} "
        f"use_breakeven={cfg.use_breakeven} breakeven_r={cfg.breakeven_r} "
        f"use_liquidity_tp={cfg.use_liquidity_tp} "
        f"use_partial_close={cfg.use_partial_close} "
        f"partial_trigger={cfg.partial_trigger_pct} partial_close={cfg.partial_close_pct} "
        f"use_premium_discount={cfg.use_premium_discount}")
    log(f"Data: {len(m5)} M5, {len(h1)} H1, {len(m15)} M15 bars")
    log(f"Contract size: {contract_size}, Point: {point}")
    log(f"M5 range: {m5['time'].iloc[0]} .. {m5['time'].iloc[-1]}")

    # ----- numpy arrays for speed -----
    h1_t = h1["time"].values
    m5_t = m5["time"].values
    m15_t = m15["time"].values

    # Map each M5 bar to the last *closed* H1 and M15 bar
    h_idx = np.searchsorted(h1_t, m5_t - np.timedelta64(60, "m"), side="right") - 1
    liq_idx = np.searchsorted(m15_t, m5_t - np.timedelta64(15, "m"), side="right") - 1

    o = m5["open"].to_numpy(float)
    hi = m5["high"].to_numpy(float)
    lo = m5["low"].to_numpy(float)
    cl = m5["close"].to_numpy(float)
    spread = m5["spread"].to_numpy(float) * point
    times = m5["time"]

    _weekdays = pd.to_datetime(m5["time"]).dt.weekday.to_numpy()
    _hours = pd.to_datetime(m5["time"]).dt.hour.to_numpy()
    _minutes = pd.to_datetime(m5["time"]).dt.minute.to_numpy()

    # Find start index: need enough lookback for both HTF and LTF
    start = LTF_CANDLES_LOOKBACK
    while start < len(m5) and h_idx[start] < HTF_CANDLES_LOOKBACK:
        start += 1

    # ----- state -----
    trades: list[dict] = []
    equity: list[dict] = []
    balance = START_BALANCE
    poi: Optional[ZonePOI] = None
    poi_h = -1
    trend = "NEUTRAL"
    watch_key = None
    watch_start = None
    abandoned = False
    dead_zones: set = set()
    position: Optional[dict] = None
    t_start = _time.time()

    # HTF cache: avoid recomputing POI for the same H1 bar
    htf_cache: dict = {}

    # Rejection counters
    rejections: dict[str, int] = {}

    def record_rejection(reason: str, t: int):
        rejections[reason] = rejections.get(reason, 0) + 1

    log(f"Starting main loop from bar {start} to {len(m5) - 2}")

    for t in range(start, len(m5) - 1):
        if t % 10000 == 0:
            elapsed = _time.time() - t_start
            log(f"  ... bar {t}/{len(m5)} ({elapsed:.0f}s, "
                f"{len(trades)} trades, balance=${balance:.2f})")

        # ================================================================
        # MANAGE OPEN POSITION
        # ================================================================
        if position is not None:
            p = position
            vol = p["vol"]

            # -- Partial close --
            if cfg.use_partial_close and not p.get("partial_done"):
                tp_dist = abs(p["tp"] - p["entry"])
                trigger_dist = tp_dist * cfg.partial_trigger_pct
                if p["dir"] == "BUY":
                    triggered = hi[t] >= p["entry"] + trigger_dist
                else:
                    triggered = lo[t] + spread[t] <= p["entry"] - trigger_dist

                if triggered:
                    close_vol = vol * cfg.partial_close_pct
                    remain_vol = vol - close_vol
                    partial_px = (p["entry"] + trigger_dist if p["dir"] == "BUY"
                                  else p["entry"] - trigger_dist)
                    partial_pnl = ((partial_px - p["entry"]) if p["dir"] == "BUY"
                                   else (p["entry"] - partial_px)) * close_vol * contract_size
                    balance += partial_pnl
                    p["partial_pnl"] = round(partial_pnl, 2)
                    p["partial_done"] = True
                    p["vol"] = remain_vol
                    vol = remain_vol

                    # Move SL to entry
                    p["sl"] = p["entry"]

                    # Move TP to next liquidity level
                    li = liq_idx[t]
                    if li >= 0:
                        liq_slice = m15.iloc[max(0, li - 200): li + 1]
                        next_liq = SMCStrategy.find_next_liquidity(
                            liq_slice, p["dir"], p["tp"],
                            strength=SWING_STRENGTH, use_closed_candles=False)
                        if next_liq is not None:
                            p["tp"] = round(next_liq, 5)
                            p["tp_extended"] = True

                    log(f"  [{_ts(times,t)}] PARTIAL CLOSE: {p['dir']} "
                        f"closed {close_vol:.2f} lots @ {partial_px:.5f}, "
                        f"P&L=${partial_pnl:.2f}, SL->entry, "
                        f"remain={remain_vol:.2f} lots, "
                        f"new TP={p['tp']:.5f}")

            # -- Breakeven (standalone, if no partial close) --
            if (cfg.use_breakeven and not p.get("be_moved")
                    and not p.get("partial_done")):
                risk_px = p["risk_px"]
                trigger_dist = risk_px * cfg.breakeven_r
                if p["dir"] == "BUY":
                    if hi[t] >= p["entry"] + trigger_dist:
                        p["sl"] = p["entry"]
                        p["be_moved"] = True
                        log(f"  [{_ts(times,t)}] BREAKEVEN: {p['dir']} "
                            f"SL moved to entry @ {cfg.breakeven_r}R")
                else:
                    if lo[t] + spread[t] <= p["entry"] - trigger_dist:
                        p["sl"] = p["entry"]
                        p["be_moved"] = True
                        log(f"  [{_ts(times,t)}] BREAKEVEN: {p['dir']} "
                            f"SL moved to entry @ {cfg.breakeven_r}R")

            # -- SL / TP hit --
            if p["dir"] == "BUY":
                hit_sl = lo[t] <= p["sl"]
                hit_tp = hi[t] >= p["tp"]
            else:
                hit_sl = hi[t] + spread[t] >= p["sl"]
                hit_tp = lo[t] + spread[t] <= p["tp"]

            if hit_sl or hit_tp:
                # Bar touching both = loss
                exit_reason = "SL" if hit_sl else "TP"
                exit_px = p["sl"] if hit_sl else p["tp"]
                remaining_pnl = ((exit_px - p["entry"]) if p["dir"] == "BUY"
                                 else (p["entry"] - exit_px)) * vol * contract_size
                balance += remaining_pnl
                total_pnl = remaining_pnl + p.get("partial_pnl", 0)

                p.update(
                    exit_time=_ts(times, t),
                    exit_price=round(exit_px, 5),
                    exit_reason=exit_reason,
                    bars_held=t - p.pop("entry_bar"),
                    pnl=round(total_pnl, 2),
                    r_multiple=round(total_pnl / p["risk_usd"], 2) if p["risk_usd"] else 0,
                    balance=round(balance, 2),
                )
                p.pop("zone", None)
                trades.append(p)
                equity.append({"time": p["exit_time"], "balance": round(balance, 2)})

                log(f"  [{_ts(times,t)}] EXIT {exit_reason}: {p['dir']} "
                    f"@ {exit_px:.5f}, P&L=${total_pnl:.2f}, "
                    f"R={p['r_multiple']:.2f}, balance=${balance:.2f} "
                    f"(strategy={p.get('strategy','?')})")

                position = None
                watch_key = None
                abandoned = False

                if balance <= 0:
                    log(f"  [{_ts(times,t)}] BLOWN — balance <= 0, stopping.")
                    break

            continue  # if position is open, skip signal detection

        # ================================================================
        # FILTERS
        # ================================================================

        # Weekend
        if cfg.skip_weekends and _weekdays[t] >= 5:
            continue

        # Blocked days
        if cfg.blocked_days and _weekdays[t] in cfg.blocked_days:
            record_rejection("BLOCKED_DAY", t)
            continue

        # Night session
        h = _hours[t]
        ns, ne = cfg.night_start, cfg.night_end
        if ns != ne:
            if ns > ne:
                if h >= ns or h < ne:
                    record_rejection("NIGHT_SESSION", t)
                    continue
            elif ns <= h < ne:
                record_rejection("NIGHT_SESSION", t)
                continue

        # ================================================================
        # HTF TREND + POI (refresh once per closed H1 bar)
        # ================================================================
        if h_idx[t] != poi_h:
            poi_h = h_idx[t]
            ck = poi_h
            if ck in htf_cache:
                trend, poi = htf_cache[ck]
            else:
                wh = h1.iloc[max(0, poi_h - HTF_CANDLES_LOOKBACK + 1): poi_h + 1]
                trend = (SMCStrategy.get_htf_trend(
                    wh, cfg.trend_ema_period, use_closed_candles=False)
                    if cfg.use_trend_filter else "NEUTRAL")
                poi = SMCStrategy.detect_htf_poi(
                    wh, use_trend_filter=cfg.use_trend_filter,
                    ema_period=cfg.trend_ema_period, use_closed_candles=False)
                htf_cache[ck] = (trend, poi)

                if poi is not None:
                    log(f"  [{_ts(times,t)}] POI found: {poi.type} "
                        f"{poi.top:.5f}-{poi.bottom:.5f} (trend={trend})")
                # Don't log every None — too noisy

            if poi is None:
                watch_key = None
                abandoned = False

        # Blocked trend filter
        if trend in cfg.blocked_trends:
            record_rejection("BLOCKED_TREND", t)
            continue

        # ================================================================
        # STRATEGY ATTEMPTS  (first valid signal wins)
        # ================================================================
        setup = None
        strategy_name = None

        ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
        cur_hour = int(_hours[t])
        cur_min = int(_minutes[t])
        mid_price = cl[t] + spread[t] / 2.0

        # --- AMD (skip if use_sweep_filter or blocked hour) ---
        if not cfg.use_sweep_filter and cur_hour not in cfg.blocked_amd_hours:
            asian = SMCStrategy.get_asian_range(
                ltf, use_closed_candles=False)
            if asian is not None:
                amd = SMCStrategy.check_amd_setup(
                    ltf, asian.high, asian.low,
                    swing_strength=SWING_STRENGTH,
                    rrr_fallback=cfg.rrr,
                    use_closed_candles=False)
                if amd is not None:
                    # Premium/discount filter
                    if cfg.use_premium_discount:
                        pd_zone = SMCStrategy.get_premium_discount(
                            ltf, lookback=50, use_closed_candles=False)
                        if not SMCStrategy.is_premium_discount_aligned(
                                pd_zone, amd["direction"]):
                            record_rejection("AMD_PD_MISMATCH", t)
                            log(f"  [{_ts(times,t)}] AMD: {amd['direction']} "
                                f"rejected — P/D zone mismatch "
                                f"({pd_zone['zone'] if pd_zone else 'N/A'})")
                            amd = None

                    if amd is not None:
                        setup = amd
                        strategy_name = "AMD"
                        log(f"  [{_ts(times,t)}] AMD SIGNAL: {amd['direction']} "
                            f"entry={amd['entry']:.5f} sl={amd['sl']:.5f} "
                            f"tp={amd['tp']:.5f}")
                else:
                    # Log reason AMD failed (only occasionally to avoid noise)
                    if t % 500 == 0:
                        log(f"  [{_ts(times,t)}] AMD: asian range found "
                            f"({asian.high:.5f}-{asian.low:.5f}) but no setup")
            else:
                if t % 2000 == 0:
                    log(f"  [{_ts(times,t)}] AMD: no Asian range")

        # --- Silver Bullet (skip if use_sweep_filter or blocked hour) ---
        if setup is None and not cfg.use_sweep_filter and cur_hour not in cfg.blocked_sb_hours:
            li = liq_idx[t]
            if li >= 0:
                m15_slice = m15.iloc[max(0, li - 200): li + 1]
                sb = SMCStrategy.check_silver_bullet(
                    ltf, m15_slice,
                    current_hour_utc=cur_hour,
                    current_minute=cur_min,
                    swing_strength=SWING_STRENGTH,
                    rrr_fallback=cfg.rrr,
                    use_closed_candles=False)
                if sb is not None:
                    # Premium/discount filter
                    if cfg.use_premium_discount:
                        pd_zone = SMCStrategy.get_premium_discount(
                            ltf, lookback=50, use_closed_candles=False)
                        if not SMCStrategy.is_premium_discount_aligned(
                                pd_zone, sb["direction"]):
                            record_rejection("SB_PD_MISMATCH", t)
                            log(f"  [{_ts(times,t)}] SB: {sb['direction']} "
                                f"rejected — P/D zone mismatch")
                            sb = None

                    if sb is not None:
                        setup = sb
                        strategy_name = "SILVER_BULLET"
                        log(f"  [{_ts(times,t)}] SILVER_BULLET SIGNAL: "
                            f"{sb['direction']} entry={sb['entry']:.5f} "
                            f"sl={sb['sl']:.5f} tp={sb['tp']:.5f} "
                            f"window={sb.get('window','?')}")
                else:
                    # Log SB time window checks
                    if cur_hour in (8, 15, 19) and cur_min == 0:
                        log(f"  [{_ts(times,t)}] SB: in time window "
                            f"{cur_hour}:00 UTC but no setup")

        # --- Sweep+FVG (always available) ---
        if setup is None and poi is not None:
            zone = (poi.type, round(float(poi.top), 5),
                    round(float(poi.bottom), 5))
            if zone not in dead_zones:
                # Zone in play?
                in_zone = (poi.bottom <= mid_price <= poi.top
                           or (lo[t] <= poi.top and hi[t] >= poi.bottom))

                if in_zone and SMCStrategy.is_zone_in_play(
                        poi, ltf, current_price=mid_price):
                    # Zone patience
                    if zone != watch_key:
                        watch_key = zone
                        watch_start = t
                        abandoned = False
                        log(f"  [{_ts(times,t)}] ZONE ENTERED: {poi.type} "
                            f"{poi.top:.5f}-{poi.bottom:.5f} "
                            f"(price={mid_price:.5f})")

                    if not abandoned:
                        if (t - watch_start + 1) > MAX_LTF_WAIT_CANDLES:
                            abandoned = True
                            log(f"  [{_ts(times,t)}] ZONE ABANDONED: "
                                f"waited {MAX_LTF_WAIT_CANDLES} bars, no setup")
                        else:
                            # Consolidation check
                            if SMCStrategy.is_consolidating(
                                    ltf, use_closed_candles=False):
                                record_rejection("CONSOLIDATING", t)
                            else:
                                # Require liquidity sweep aligned with POI
                                sweep = SMCStrategy.detect_liquidity_sweep(
                                    ltf, swing_strength=SWING_STRENGTH,
                                    use_closed_candles=False)
                                if sweep is None:
                                    record_rejection("NO_SWEEP", t)
                                elif ((poi.type == "BULLISH"
                                       and sweep.direction != "BULLISH")
                                      or (poi.type == "BEARISH"
                                          and sweep.direction != "BEARISH")):
                                    record_rejection("SWEEP_DIRECTION", t)
                                    log(f"  [{_ts(times,t)}] SWEEP+FVG: sweep "
                                        f"{sweep.direction} != POI {poi.type}")
                                else:
                                    # LTF confirmation
                                    confirm = SMCStrategy.check_ltf_confirmation(
                                        ltf, poi, cfg.rrr,
                                        use_closed_candles=False,
                                        stop_mode=cfg.stop_mode,
                                        buffer_atr=0.5)
                                    if confirm is not None:
                                        # Premium/discount filter
                                        if cfg.use_premium_discount:
                                            pd_zone = SMCStrategy.get_premium_discount(
                                                ltf, lookback=50,
                                                use_closed_candles=False)
                                            if not SMCStrategy.is_premium_discount_aligned(
                                                    pd_zone,
                                                    confirm["direction"]):
                                                record_rejection(
                                                    "SWEEP_PD_MISMATCH", t)
                                                log(f"  [{_ts(times,t)}] "
                                                    f"SWEEP+FVG: P/D mismatch")
                                                confirm = None

                                        if confirm is not None:
                                            setup = confirm
                                            strategy_name = "SWEEP_FVG"
                                            log(f"  [{_ts(times,t)}] "
                                                f"SWEEP+FVG SIGNAL: "
                                                f"{confirm['direction']} "
                                                f"entry={confirm['entry']:.5f} "
                                                f"sl={confirm['sl']:.5f} "
                                                f"tp={confirm['tp']:.5f}")
                                    else:
                                        record_rejection("NO_LTF_CONFIRM", t)
                else:
                    if zone == watch_key:
                        watch_key = None
                        abandoned = False

        # ================================================================
        # OPEN POSITION
        # ================================================================
        if setup is None:
            continue

        # Fill at the NEXT bar's open (no lookahead)
        nt = t + 1
        entry = o[nt] + spread[nt] if setup["direction"] == "BUY" else o[nt]
        sl = setup["sl"]
        risk = abs(entry - sl)
        if risk <= 0:
            log(f"  [{_ts(times,t)}] SKIP: zero risk (entry={entry:.5f} "
                f"sl={sl:.5f})")
            continue

        # Max OB risk width filter
        if cfg.max_ob_risk_px > 0 and risk > cfg.max_ob_risk_px:
            record_rejection("OB_TOO_WIDE", t)
            log(f"  [{_ts(times,t)}] OB_TOO_WIDE: risk {risk:.2f} > "
                f"max {cfg.max_ob_risk_px:.2f}")
            continue

        # Entry drift guard: >20% slippage from intended entry
        intended_entry = setup.get("entry")
        if intended_entry is not None and risk > 0:
            slippage_pct = abs(entry - intended_entry) / risk
            if slippage_pct > 0.20:
                record_rejection("ENTRY_DRIFT", t)
                log(f"  [{_ts(times,t)}] ENTRY_DRIFT: market {entry:.5f} "
                    f"drifted {slippage_pct:.0%} from intended "
                    f"{intended_entry:.5f}")
                continue

        # Risk-based lot sizing: RISK_PERCENT of current balance
        risk_amount = balance * cfg.risk_percent / 100.0
        pnl_per_lot = risk * contract_size
        if pnl_per_lot <= 0:
            continue
        vol = risk_amount / pnl_per_lot
        vol = max(0.01, int(vol * 100) / 100.0)  # floor to 0.01 step

        risk_usd = risk * vol * contract_size

        # Dollar cap check
        if cfg.max_risk_usd > 0 and risk_usd > cfg.max_risk_usd:
            # Try reducing lot to fit cap
            max_vol = cfg.max_risk_usd / pnl_per_lot
            max_vol = max(0.01, int(max_vol * 100) / 100.0)
            if max_vol < 0.01:
                record_rejection("RISK_USD_EXCEEDED", t)
                log(f"  [{_ts(times,t)}] RISK_USD: ${risk_usd:.2f} > "
                    f"${cfg.max_risk_usd:.0f} cap, can't reduce lots")
                continue
            vol = max_vol
            risk_usd = risk * vol * contract_size

        if cfg.max_risk_pct > 0:
            max_allowed = balance * cfg.max_risk_pct / 100
            if risk_usd > max_allowed:
                record_rejection("RISK_PCT_EXCEEDED", t)
                log(f"  [{_ts(times,t)}] RISK_PCT: ${risk_usd:.2f} > "
                    f"{cfg.max_risk_pct}% of ${balance:.2f}")
                continue

        # Liquidity TP (for Sweep+FVG — AMD/SB set their own TP)
        tp = setup["tp"]
        tp_source = "fixed"
        if (cfg.use_liquidity_tp and strategy_name == "SWEEP_FVG"):
            li = liq_idx[nt]
            if li >= 0:
                m15_slice = m15.iloc[max(0, li - 200): li + 1]
                liq = SMCStrategy.find_nearest_liquidity(
                    m15_slice, setup["direction"], entry,
                    strength=SWING_STRENGTH, use_closed_candles=False)
                if liq is not None:
                    liq_dist = abs(liq - entry)
                    liq_rr = liq_dist / risk if risk > 0 else 0
                    if liq_rr >= MIN_RRR_LIQUIDITY:
                        tp = liq
                        tp_source = "liquidity"

        # Ensure TP is on the correct side
        if setup["direction"] == "BUY" and tp <= entry:
            tp = entry + risk * cfg.rrr
        elif setup["direction"] == "SELL" and tp >= entry:
            tp = entry - risk * cfg.rrr

        position = {
            "n": len(trades) + 1,
            "symbol": symbol,
            "strategy": strategy_name,
            "dir": setup["direction"],
            "trend": trend,
            "entry_time": _ts(times, nt),
            "entry": round(entry, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "tp_source": tp_source,
            "risk_px": round(risk, 5),
            "risk_usd": round(risk_usd, 2),
            "vol": vol,
            "spread_px": round(spread[nt], 5),
            "entry_bar": nt,
            "zone": watch_key,
        }

        log(f"  [{_ts(times,nt)}] ENTRY: {setup['direction']} "
            f"{vol:.2f} lots @ {entry:.5f} "
            f"SL={sl:.5f} TP={tp:.5f} ({tp_source}) "
            f"risk=${risk_usd:.2f} strategy={strategy_name}")

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = _time.time() - t_start
    log(f"=== BACKTEST COMPLETE: {symbol} ===")
    log(f"Runtime: {elapsed:.1f}s, {len(trades)} trades, "
        f"final balance=${balance:.2f}")
    log(f"Rejections: {json.dumps(rejections, indent=2)}")

    summary = _build_summary(symbol, trades, equity, balance, times, start,
                             cfg, m5, point, elapsed)
    summary["rejections"] = rejections

    return {
        "summary": summary,
        "trades": trades,
        "equity": equity,
        "logs": logs,
    }


def _build_summary(symbol, trades, equity, balance, times, start,
                    cfg, m5, point, runtime):
    """Build the summary statistics dict."""
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = -sum(t["pnl"] for t in losses)

    peak, max_dd, max_dd_pct = START_BALANCE, 0.0, 0.0
    for e in equity:
        peak = max(peak, e["balance"])
        dd = peak - e["balance"]
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak * 100

    streak = best_w = best_l = 0
    for t in trades:
        if t["pnl"] > 0:
            streak = streak + 1 if streak > 0 else 1
            best_w = max(best_w, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            best_l = min(best_l, streak)

    n = len(trades)

    # Per-strategy breakdown
    strat_counts = {}
    strat_pnl = {}
    for tr in trades:
        s = tr.get("strategy", "?")
        strat_counts[s] = strat_counts.get(s, 0) + 1
        strat_pnl[s] = strat_pnl.get(s, 0) + tr["pnl"]

    return {
        "symbol": symbol,
        "stop_mode": cfg.stop_mode,
        "rrr": cfg.rrr,
        "period_from": str(times.iloc[start]),
        "period_to": str(times.iloc[-1]),
        "days": round((times.iloc[-1] - times.iloc[start]).total_seconds() / 86400, 1),
        "m5_bars": len(m5),
        "start_balance": START_BALANCE,
        "end_balance": round(balance, 2),
        "net_pnl": round(balance - START_BALANCE, 2),
        "return_pct": round((balance / START_BALANCE - 1) * 100, 2),
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 1) if n else 0,
        "profit_factor": round(gross_w / gross_l, 2) if gross_l else None,
        "avg_win": round(gross_w / len(wins), 2) if wins else 0,
        "avg_loss": round(-gross_l / len(losses), 2) if losses else 0,
        "expectancy": round((balance - START_BALANCE) / n, 2) if n else 0,
        "expectancy_r": round(sum(t["r_multiple"] for t in trades) / n, 3) if n else 0,
        "total_r": round(sum(t["r_multiple"] for t in trades), 1),
        "best_trade": round(max((t["pnl"] for t in trades), default=0), 2),
        "worst_trade": round(min((t["pnl"] for t in trades), default=0), 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "longest_win_streak": best_w,
        "longest_loss_streak": abs(best_l),
        "avg_bars_held": round(float(np.mean([t["bars_held"] for t in trades])), 1) if n else 0,
        "avg_risk_usd": round(float(np.mean([t["risk_usd"] for t in trades])), 2) if n else 0,
        "median_spread": int(np.median(m5["spread"].to_numpy())),
        "strategy_breakdown": {
            s: {"trades": strat_counts[s], "net_pnl": round(strat_pnl[s], 2)}
            for s in strat_counts
        },
        "runtime_s": round(runtime, 1),
    }


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Multi-symbol SMC backtest engine")
    ap.add_argument("--symbols", nargs="+", default=None,
                    help=f"Symbols to test (default: {DEFAULT_SYMBOLS})")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-pull history from MT5 (ignore cache)")
    ap.add_argument("--no-logs", action="store_true",
                    help="Suppress per-bar log output in JSON")
    args = ap.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS

    # Initialize MT5 (needed for data loading and config imports)
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        print("Start MetaTrader 5 and retry.")
        sys.exit(1)

    # Login
    from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 login failed: {mt5.last_error()}")
        sys.exit(1)

    print(f"Connected to MT5. Symbols: {symbols}")
    print(f"Each symbol gets an independent ${START_BALANCE:,.0f} starting balance.\n")

    # ---- Header ----
    hdr = (f"{'Symbol':<10}{'Trades':>7}{'Wins':>6}{'Win%':>7}"
           f"{'PF':>7}{'ExpR':>8}{'TotalR':>8}"
           f"{'Net$':>10}{'Ret%':>8}{'MaxDD%':>8}"
           f"{'Strategies':>30}")
    print(hdr)
    print("-" * len(hdr))

    all_results = {}

    for sym in symbols:
        print(f"\n{'='*60}")
        print(f" Loading {sym}...")
        print(f"{'='*60}")

        try:
            data = load_symbol_data(sym, refresh=args.refresh)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue

        cfg = SymbolConfig.load(sym)

        result = run_symbol(
            symbol=sym,
            m5=data["m5"],
            h1=data["h1"],
            m15=data["m15"],
            contract_size=data["contract_size"],
            point=data["point"],
            cfg=cfg,
        )

        s = result["summary"]

        # Strategy breakdown string
        strat_str = ", ".join(
            f"{k}:{v['trades']}" for k, v in s.get("strategy_breakdown", {}).items()
        )

        print(f"{sym:<10}{s['trades']:>7}{s['wins']:>6}{s['win_rate']:>7.1f}"
              f"{str(s['profit_factor']):>7}{s['expectancy_r']:>8.3f}"
              f"{s['total_r']:>8.1f}"
              f"{s['net_pnl']:>10.2f}{s['return_pct']:>8.2f}"
              f"{s['max_drawdown_pct']:>8.2f}"
              f"  {strat_str}")

        # Save per-symbol JSON
        out_path = HERE / f"results_{sym.lower()}.json"
        save_data = {
            "summary": result["summary"],
            "trades": result["trades"],
            "equity": result["equity"],
        }
        if not args.no_logs:
            save_data["logs"] = result["logs"]

        out_path.write_text(json.dumps(save_data, indent=1), "utf-8")
        print(f"  -> {out_path.name}")

        all_results[sym] = result

    # ---- Grand summary ----
    print(f"\n{'='*60}")
    print(" GRAND SUMMARY")
    print(f"{'='*60}")
    total_trades = sum(r["summary"]["trades"] for r in all_results.values())
    total_pnl = sum(r["summary"]["net_pnl"] for r in all_results.values())
    total_start = START_BALANCE * len(all_results)
    total_end = sum(r["summary"]["end_balance"] for r in all_results.values())
    print(f"  Symbols tested: {len(all_results)}")
    print(f"  Total trades:   {total_trades}")
    print(f"  Total P&L:      ${total_pnl:+,.2f}")
    print(f"  Combined start: ${total_start:,.2f}")
    print(f"  Combined end:   ${total_end:,.2f}")
    print(f"  Combined return: {(total_end/total_start - 1)*100:+.2f}%")

    mt5.shutdown()


if __name__ == "__main__":
    main()
