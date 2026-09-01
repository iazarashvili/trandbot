"""Multi-strategy backtest: all 5 strategies on all 3 symbols, $10000, 1% risk."""
import sys
import importlib
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from strategy import SMCStrategy

mt5.initialize()
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

BALANCE = 10000.0
RISK_PCT = 1.0

symbols_cfg = [
    ("BTCUSD", "config_btcusd"),
    ("XAUUSD", "config_xauusd"),
    ("GBPUSD", "config_gbpusd"),
]

all_data = {}
for sym, cfg_name in symbols_cfg:
    info = mt5.symbol_info(sym)
    frames = {}
    for key, tf, n in [("m5", mt5.TIMEFRAME_M5, 100000),
                        ("h1", mt5.TIMEFRAME_H1, 10000),
                        ("m15", mt5.TIMEFRAME_M15, 50000)]:
        rates = mt5.copy_rates_from_pos(sym, tf, 0, n)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        frames[key] = df
    all_data[sym] = {
        "frames": frames,
        "contract": info.trade_contract_size,
        "point": info.point,
        "cfg_name": cfg_name,
    }
    print(f"{sym}: {frames['m5']['time'].iloc[0].date()} to {frames['m5']['time'].iloc[-1].date()}")

mt5.shutdown()


def run_multi_strategy(sym, data):
    """Run all strategies on one symbol, priority order."""
    cfg = importlib.import_module(data["cfg_name"])

    from config import (HTF_CANDLES_LOOKBACK, LTF_CANDLES_LOOKBACK,
                        TREND_EMA_PERIOD, MAX_LTF_WAIT_CANDLES,
                        SKIP_WEEKENDS, SWING_STRENGTH)
    stop_mode = getattr(cfg, "STOP_MODE", "window")
    night_start = getattr(cfg, "NIGHT_START_HOUR", 0)
    night_end = getattr(cfg, "NIGHT_END_HOUR", 0)
    blocked = list(getattr(cfg, "BLOCKED_DAYS", []))
    rrr = getattr(cfg, "RRR", 3.0)
    be_r = getattr(cfg, "BREAKEVEN_R", 2.0) if getattr(cfg, "USE_BREAKEVEN", False) else 0.0

    m5 = data["frames"]["m5"]
    h1 = data["frames"]["h1"]
    m15 = data["frames"]["m15"]
    point = data["point"]
    contract = data["contract"]

    h1_t = h1["time"].values
    m5_t = m5["time"].values
    h_idx = np.searchsorted(h1_t, m5_t - np.timedelta64(60, "m"), side="right") - 1
    m15_t = m15["time"].values
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

    sb = LTF_CANDLES_LOOKBACK
    while sb < len(m5) and h_idx[sb] < HTF_CANDLES_LOOKBACK:
        sb += 1

    htf_cache = {}
    poi = None
    poi_h = -1
    wk = ws = None
    ab = False
    pos = None
    trades = []
    balance = BALANCE

    for t in range(sb, len(m5) - 1):
        # --- Manage position ---
        if pos is not None:
            p = pos
            vol = p["vol"]

            # Breakeven
            if be_r > 0 and not p.get("be"):
                td = p["risk_px"] * be_r
                if p["dir"] == "BUY" and hi[t] >= p["entry"] + td:
                    p["sl"] = p["entry"]; p["be"] = True
                elif p["dir"] == "SELL" and lo[t] + spread[t] <= p["entry"] - td:
                    p["sl"] = p["entry"]; p["be"] = True

            if p["dir"] == "BUY":
                hs = lo[t] <= p["sl"]; ht = hi[t] >= p["tp"]
            else:
                hs = hi[t] + spread[t] >= p["sl"]; ht = lo[t] + spread[t] <= p["tp"]

            if hs or ht:
                ep = p["sl"] if hs else p["tp"]
                pnl = ((ep - p["entry"]) if p["dir"] == "BUY"
                       else (p["entry"] - ep)) * vol * contract
                balance += pnl
                risk_usd = p["risk_px"] * vol * contract
                trades.append({
                    "pnl": round(pnl, 2), "balance": round(balance, 2),
                    "entry_time": p["entry_time"], "exit_time": str(times.iloc[t]),
                    "dir": p["dir"], "strategy": p["strategy"],
                    "exit_reason": "SL" if hs else "TP",
                    "r_multiple": round(pnl / risk_usd, 2) if risk_usd > 0 else 0,
                })
                pos = None; wk = None; ab = False
                if balance <= 0: break
            continue

        # --- Filters ---
        if SKIP_WEEKENDS and _weekdays[t] >= 5: continue
        if blocked and _weekdays[t] in blocked: continue
        h = _hours[t]
        if night_start > night_end:
            if h >= night_start or h < night_end: continue
        elif night_start != night_end and night_start <= h < night_end: continue

        # --- HTF ---
        if h_idx[t] != poi_h:
            poi_h = h_idx[t]
            if poi_h in htf_cache:
                trend, poi = htf_cache[poi_h]
            else:
                wh = h1.iloc[max(0, poi_h - HTF_CANDLES_LOOKBACK + 1): poi_h + 1]
                trend = SMCStrategy.get_htf_trend(wh, TREND_EMA_PERIOD, use_closed_candles=False)
                poi = SMCStrategy.detect_htf_poi(wh, use_trend_filter=True,
                        ema_period=TREND_EMA_PERIOD, use_closed_candles=False)
                htf_cache[poi_h] = (trend, poi)
            if poi is None: wk = None; ab = False

        ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
        mid = cl[t] + spread[t] / 2.0

        # --- Try all strategies ---
        setup = None
        strat_name = None

        # 1. AMD
        if setup is None:
            asian = SMCStrategy.get_asian_range(ltf, use_closed_candles=False)
            if asian is not None:
                amd = SMCStrategy.check_amd_setup(
                    ltf, asian.high, asian.low,
                    swing_strength=3, rrr_fallback=rrr,
                    use_closed_candles=False)
                if amd:
                    setup = amd
                    strat_name = "AMD"

        # 2. Silver Bullet
        if setup is None:
            li = liq_idx[t]
            if li >= 0:
                liq_slice = m15.iloc[max(0, li - 200): li + 1]
                sb_setup = SMCStrategy.check_silver_bullet(
                    ltf, liq_slice,
                    current_hour_utc=int(_hours[t]),
                    current_minute=int(_minutes[t]),
                    swing_strength=3, rrr_fallback=rrr,
                    use_closed_candles=False)
                if sb_setup:
                    setup = sb_setup
                    strat_name = "SILVER_BULLET"

        # 3. Sweep + FVG (needs POI)
        if setup is None and poi is not None:
            if poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom):
                if SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
                    sweep = SMCStrategy.detect_liquidity_sweep(
                        ltf, swing_strength=3, use_closed_candles=False)
                    if sweep is not None:
                        dir_match = ((poi.type == "BULLISH" and sweep.direction == "BULLISH") or
                                     (poi.type == "BEARISH" and sweep.direction == "BEARISH"))
                        if dir_match:
                            s = SMCStrategy.check_ltf_confirmation(
                                ltf, poi, rrr, use_closed_candles=False,
                                stop_mode=stop_mode, buffer_atr=0.5)
                            if s:
                                setup = s
                                strat_name = "SWEEP_FVG"

        # 4. P/D + FVG (needs POI)
        if setup is None and poi is not None:
            if poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom):
                if SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
                    s = SMCStrategy.check_ltf_confirmation(
                        ltf, poi, rrr, use_closed_candles=False,
                        stop_mode=stop_mode, buffer_atr=0.5)
                    if s:
                        pd_zone = SMCStrategy.get_premium_discount(
                            ltf, lookback=50, use_closed_candles=False)
                        if SMCStrategy.is_premium_discount_aligned(pd_zone, s["direction"]):
                            setup = s
                            strat_name = "PD_FVG"

        # 5. Base FVG (needs POI)
        if setup is None and poi is not None:
            if poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom):
                if SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
                    s = SMCStrategy.check_ltf_confirmation(
                        ltf, poi, rrr, use_closed_candles=False,
                        stop_mode=stop_mode, buffer_atr=0.5)
                    if s:
                        setup = s
                        strat_name = "FVG"

        if setup is None:
            continue

        # --- Open position ---
        entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
        sl = setup["sl"]
        risk = abs(entry - sl)
        if risk <= 0: continue
        tp = setup["tp"]

        # Dynamic lot: 1% risk
        risk_amount = balance * RISK_PCT / 100.0
        pnl_per_lot = risk * contract
        if pnl_per_lot <= 0: continue
        vol = risk_amount / pnl_per_lot
        vol = max(0.01, int(vol * 100) / 100.0)

        pos = {
            "dir": setup["direction"],
            "entry_time": str(times.iloc[t + 1]),
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "risk_px": round(risk, 2),
            "vol": vol,
            "entry_bar": t + 1,
            "strategy": strat_name,
        }

    return trades


# Run all symbols
print(f"\n{'='*100}")
print(f"  MULTI-STRATEGY BACKTEST | $10,000 | 1% Risk | All Strategies")
print(f"{'='*100}\n")

all_trades = []
symbol_trades = defaultdict(list)

for sym, cfg_name in symbols_cfg:
    if cfg_name in sys.modules:
        del sys.modules[cfg_name]
    trades = run_multi_strategy(sym, all_data[sym])
    for t in trades:
        t["symbol"] = sym
        all_trades.append(t)
        symbol_trades[sym].append(t)

all_trades.sort(key=lambda t: t.get("exit_time", ""))

# Replay on shared balance
balance = BALANCE
for t in all_trades:
    balance += t["pnl"]
    t["shared_balance"] = round(balance, 2)

# Strategy breakdown
strat_stats = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0, "gw": 0.0, "gl": 0.0})
for t in all_trades:
    s = strat_stats[t["strategy"]]
    s["t"] += 1
    s["pnl"] += t["pnl"]
    if t["pnl"] > 0:
        s["w"] += 1
        s["gw"] += t["pnl"]
    else:
        s["gl"] -= t["pnl"]

print(f"{'Strategy':<18} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PF':>6} {'Net':>12}")
print("-" * 60)
for strat in ["AMD", "SILVER_BULLET", "SWEEP_FVG", "PD_FVG", "FVG"]:
    s = strat_stats[strat]
    if s["t"] == 0: continue
    wr = round(s["w"] / s["t"] * 100, 1)
    pf = round(s["gw"] / s["gl"], 2) if s["gl"] > 0 else "-"
    sign = "+" if s["pnl"] >= 0 else ""
    print(f"{strat:<18} {s['t']:>7} {s['w']:>5} {wr:>5.1f}% {str(pf):>6} {sign}${s['pnl']:>10,.2f}")

# Per symbol + strategy
print(f"\n{'Symbol':<10} {'Strategy':<18} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'Net':>12}")
print("-" * 65)
for sym in ["BTCUSD", "XAUUSD", "GBPUSD"]:
    sym_strats = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0})
    for t in symbol_trades[sym]:
        ss = sym_strats[t["strategy"]]
        ss["t"] += 1
        ss["pnl"] += t["pnl"]
        if t["pnl"] > 0: ss["w"] += 1
    for strat in ["AMD", "SILVER_BULLET", "SWEEP_FVG", "PD_FVG", "FVG"]:
        ss = sym_strats[strat]
        if ss["t"] == 0: continue
        wr = round(ss["w"] / ss["t"] * 100, 1)
        sign = "+" if ss["pnl"] >= 0 else ""
        print(f"{sym:<10} {strat:<18} {ss['t']:>7} {ss['w']:>5} {wr:>5.1f}% {sign}${ss['pnl']:>10,.2f}")

# Monthly P&L
monthly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
for t in all_trades:
    month = t["exit_time"][:7]
    monthly[month]["trades"] += 1
    monthly[month]["pnl"] += t["pnl"]
    if t["pnl"] > 0: monthly[month]["wins"] += 1

print(f"\n{'Month':<10} {'Trades':>7} {'Wins':>5} {'Loss':>6} {'Net':>12} {'Balance':>12}")
print("-" * 55)
running = BALANCE
for m in sorted(monthly):
    d = monthly[m]
    running += d["pnl"]
    losses = d["trades"] - d["wins"]
    sign = "+" if d["pnl"] >= 0 else ""
    print(f"{m:<10} {d['trades']:>7} {d['wins']:>5} {losses:>6} {sign}${d['pnl']:>10,.2f} ${running:>10,.2f}")

# Totals
print("-" * 55)
total_t = len(all_trades)
total_w = len([t for t in all_trades if t["pnl"] > 0])
total_pnl = balance - BALANCE
gw = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
gl = -sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)
pf = round(gw / gl, 2) if gl > 0 else "-"
peak = BALANCE; max_dd = 0
for t in all_trades:
    peak = max(peak, t["shared_balance"])
    dd = peak - t["shared_balance"]
    max_dd = max(max_dd, dd)
max_dd_pct = round(max_dd / peak * 100, 1) if peak > 0 else 0

print(f"\n{'='*55}")
print(f"  ${BALANCE:,.0f} -> ${balance:,.2f} ({(balance/BALANCE-1)*100:+.1f}%)")
print(f"  Trades: {total_t} | Wins: {total_w} | WR: {round(total_w/total_t*100,1) if total_t else 0}%")
print(f"  PF: {pf} | MaxDD: ${max_dd:,.2f} ({max_dd_pct}%)")
print(f"{'='*55}")
