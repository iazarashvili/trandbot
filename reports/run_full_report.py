"""Comprehensive multi-strategy report with all statistics."""
import sys
import importlib
from pathlib import Path
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

mt5.shutdown()


def run_multi_strategy(sym, data):
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
    pos = None
    trades = []
    balance = BALANCE

    for t in range(sb, len(m5) - 1):
        if pos is not None:
            p = pos
            vol = p["vol"]
            if be_r > 0 and not p.get("be"):
                td = p["risk_px"] * be_r
                if p["dir"] == "BUY" and hi[t] >= p["entry"] + td:
                    p["sl"] = p["entry"]; p["be"] = True
                elif p["dir"] == "SELL" and lo[t] + spread[t] <= p["entry"] - td:
                    p["sl"] = p["entry"]; p["be"] = True

            if p["dir"] == "BUY":
                # Track MFE/MAE
                p["mfe"] = max(p.get("mfe", 0), hi[t] - p["entry"])
                p["mae"] = max(p.get("mae", 0), p["entry"] - lo[t])
                hs = lo[t] <= p["sl"]; ht = hi[t] >= p["tp"]
            else:
                p["mfe"] = max(p.get("mfe", 0), p["entry"] - (lo[t] + spread[t]))
                p["mae"] = max(p.get("mae", 0), (hi[t] + spread[t]) - p["entry"])
                hs = hi[t] + spread[t] >= p["sl"]; ht = lo[t] + spread[t] <= p["tp"]

            if hs or ht:
                ep = p["sl"] if hs else p["tp"]
                pnl = ((ep - p["entry"]) if p["dir"] == "BUY"
                       else (p["entry"] - ep)) * vol * contract
                balance += pnl
                risk_usd = p["risk_px"] * vol * contract
                mfe_r = round(p.get("mfe", 0) / p["risk_px"], 2) if p["risk_px"] > 0 else 0
                mae_r = round(p.get("mae", 0) / p["risk_px"], 2) if p["risk_px"] > 0 else 0
                trades.append({
                    "pnl": round(pnl, 2), "balance": round(balance, 2),
                    "entry_time": p["entry_time"], "exit_time": str(times.iloc[t]),
                    "dir": p["dir"], "strategy": p["strategy"],
                    "exit_reason": "SL" if hs else "TP",
                    "r_multiple": round(pnl / risk_usd, 2) if risk_usd > 0 else 0,
                    "risk_usd": round(risk_usd, 2),
                    "entry": p["entry"], "sl": p["sl"], "tp": p["tp"],
                    "risk_px": p["risk_px"], "vol": vol,
                    "bars_held": t - p["entry_bar"],
                    "mfe_r": mfe_r, "mae_r": mae_r,
                    "entry_hour": p["entry_hour"],
                    "entry_weekday": p["entry_weekday"],
                    "symbol": sym,
                })
                pos = None
                if balance <= 0: break
            continue

        if SKIP_WEEKENDS and _weekdays[t] >= 5: continue
        if blocked and _weekdays[t] in blocked: continue
        h = _hours[t]
        if night_start > night_end:
            if h >= night_start or h < night_end: continue
        elif night_start != night_end and night_start <= h < night_end: continue

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

        ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
        mid = cl[t] + spread[t] / 2.0

        setup = None
        strat_name = None

        # 1. AMD
        if setup is None:
            asian = SMCStrategy.get_asian_range(ltf, use_closed_candles=False)
            if asian is not None:
                amd = SMCStrategy.check_amd_setup(ltf, asian.high, asian.low,
                    swing_strength=3, rrr_fallback=rrr, use_closed_candles=False)
                if amd:
                    setup = amd; strat_name = "AMD"

        # 2. Silver Bullet
        if setup is None:
            li = liq_idx[t]
            if li >= 0:
                liq_slice = m15.iloc[max(0, li - 200): li + 1]
                sb_s = SMCStrategy.check_silver_bullet(ltf, liq_slice,
                    current_hour_utc=int(_hours[t]), current_minute=int(_minutes[t]),
                    swing_strength=3, rrr_fallback=rrr, use_closed_candles=False)
                if sb_s:
                    setup = sb_s; strat_name = "SILVER_BULLET"

        # 3. Sweep + FVG
        if setup is None and poi is not None:
            if poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom):
                if SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
                    sweep = SMCStrategy.detect_liquidity_sweep(ltf, swing_strength=3, use_closed_candles=False)
                    if sweep is not None:
                        dm = ((poi.type == "BULLISH" and sweep.direction == "BULLISH") or
                              (poi.type == "BEARISH" and sweep.direction == "BEARISH"))
                        if dm:
                            s = SMCStrategy.check_ltf_confirmation(ltf, poi, rrr,
                                use_closed_candles=False, stop_mode=stop_mode, buffer_atr=0.5)
                            if s:
                                setup = s; strat_name = "SWEEP_FVG"

        if setup is None:
            continue

        entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
        sl = setup["sl"]; risk = abs(entry - sl)
        if risk <= 0: continue
        tp = setup["tp"]

        risk_amount = balance * RISK_PCT / 100.0
        pnl_per_lot = risk * contract
        if pnl_per_lot <= 0: continue
        vol = risk_amount / pnl_per_lot
        vol = max(0.01, int(vol * 100) / 100.0)

        pos = {
            "dir": setup["direction"], "entry_time": str(times.iloc[t + 1]),
            "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "risk_px": round(risk, 2), "vol": vol, "entry_bar": t + 1,
            "strategy": strat_name,
            "entry_hour": int(_hours[t + 1]) if t + 1 < len(_hours) else int(_hours[t]),
            "entry_weekday": int(_weekdays[t + 1]) if t + 1 < len(_weekdays) else int(_weekdays[t]),
        }

    return trades


# ============ RUN ============
all_trades = []
for sym, cfg_name in symbols_cfg:
    if cfg_name in sys.modules: del sys.modules[cfg_name]
    trades = run_multi_strategy(sym, all_data[sym])
    all_trades.extend(trades)

all_trades.sort(key=lambda t: t.get("exit_time", ""))
balance = BALANCE
for t in all_trades:
    balance += t["pnl"]
    t["shared_balance"] = round(balance, 2)


# ============ REPORT ============
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

# --- Header ---
section("FULL MULTI-STRATEGY REPORT")
print(f"  Balance: ${BALANCE:,.0f} | Risk: {RISK_PCT}% per trade")
print(f"  Strategies: AMD, Silver Bullet, Sweep+FVG")
print(f"  Symbols: BTCUSD, XAUUSD, GBPUSD")

# --- Overall ---
section("OVERALL RESULTS")
total_t = len(all_trades)
total_w = len([t for t in all_trades if t["pnl"] > 0])
total_pnl = sum(t["pnl"] for t in all_trades)
gw = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
gl = -sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)
pf = round(gw / gl, 2) if gl > 0 else "-"
peak = BALANCE; max_dd = 0; max_dd_bal = 0
for t in all_trades:
    peak = max(peak, t["shared_balance"])
    dd = peak - t["shared_balance"]
    if dd > max_dd:
        max_dd = dd; max_dd_bal = t["shared_balance"]

print(f"  ${BALANCE:,.0f} -> ${balance:,.2f} ({(balance/BALANCE-1)*100:+.1f}%)")
print(f"  Trades: {total_t} | Wins: {total_w} | Losses: {total_t - total_w}")
print(f"  Win Rate: {round(total_w/total_t*100,1)}% | PF: {pf}")
print(f"  MaxDD: ${max_dd:,.2f} ({round(max_dd/peak*100,1)}%)")
print(f"  Avg Win: ${gw/total_w:,.2f} | Avg Loss: ${-gl/(total_t-total_w):,.2f}" if total_w and total_t > total_w else "")
print(f"  Best Trade: ${max(t['pnl'] for t in all_trades):,.2f}")
print(f"  Worst Trade: ${min(t['pnl'] for t in all_trades):,.2f}")
print(f"  Avg Risk/Trade: ${np.mean([t['risk_usd'] for t in all_trades]):,.2f}")
print(f"  Avg Bars Held: {np.mean([t['bars_held'] for t in all_trades]):.0f}")

# --- Strategy Breakdown ---
section("STRATEGY BREAKDOWN")
print(f"{'Strategy':<18} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PF':>6} {'Net':>12} {'AvgR':>7}")
print("-" * 65)
for strat in ["AMD", "SILVER_BULLET", "SWEEP_FVG"]:
    st = [t for t in all_trades if t["strategy"] == strat]
    if not st: continue
    w = len([t for t in st if t["pnl"] > 0])
    wr = round(w / len(st) * 100, 1)
    gw_s = sum(t["pnl"] for t in st if t["pnl"] > 0)
    gl_s = -sum(t["pnl"] for t in st if t["pnl"] <= 0)
    pf_s = round(gw_s / gl_s, 2) if gl_s > 0 else "-"
    net = sum(t["pnl"] for t in st)
    avg_r = round(np.mean([t["r_multiple"] for t in st]), 2)
    sign = "+" if net >= 0 else ""
    print(f"{strat:<18} {len(st):>7} {w:>5} {wr:>5.1f}% {str(pf_s):>6} {sign}${net:>10,.2f} {avg_r:>+6.2f}")

# --- Per Symbol ---
section("PER SYMBOL BREAKDOWN")
print(f"{'Symbol':<10} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PF':>6} {'Net':>12}")
print("-" * 50)
for sym in ["BTCUSD", "XAUUSD", "GBPUSD"]:
    st = [t for t in all_trades if t["symbol"] == sym]
    if not st: continue
    w = len([t for t in st if t["pnl"] > 0])
    wr = round(w / len(st) * 100, 1)
    gw_s = sum(t["pnl"] for t in st if t["pnl"] > 0)
    gl_s = -sum(t["pnl"] for t in st if t["pnl"] <= 0)
    pf_s = round(gw_s / gl_s, 2) if gl_s > 0 else "-"
    net = sum(t["pnl"] for t in st)
    sign = "+" if net >= 0 else ""
    print(f"{sym:<10} {len(st):>7} {w:>5} {wr:>5.1f}% {str(pf_s):>6} {sign}${net:>10,.2f}")

# --- Per Symbol + Strategy ---
section("SYMBOL x STRATEGY MATRIX")
print(f"{'Symbol':<10} {'Strategy':<18} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'Net':>12}")
print("-" * 60)
for sym in ["BTCUSD", "XAUUSD", "GBPUSD"]:
    for strat in ["AMD", "SILVER_BULLET", "SWEEP_FVG"]:
        st = [t for t in all_trades if t["symbol"] == sym and t["strategy"] == strat]
        if not st: continue
        w = len([t for t in st if t["pnl"] > 0])
        wr = round(w / len(st) * 100, 1)
        net = sum(t["pnl"] for t in st)
        sign = "+" if net >= 0 else ""
        print(f"{sym:<10} {strat:<18} {len(st):>7} {w:>5} {wr:>5.1f}% {sign}${net:>10,.2f}")

# --- Monthly ---
section("MONTHLY P&L")
monthly = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0})
for t in all_trades:
    m = t["exit_time"][:7]
    monthly[m]["t"] += 1; monthly[m]["pnl"] += t["pnl"]
    if t["pnl"] > 0: monthly[m]["w"] += 1

print(f"{'Month':<10} {'Trades':>7} {'Wins':>5} {'Loss':>6} {'Net':>12} {'Balance':>12}")
print("-" * 55)
running = BALANCE
for m in sorted(monthly):
    d = monthly[m]; running += d["pnl"]
    sign = "+" if d["pnl"] >= 0 else ""
    print(f"{m:<10} {d['t']:>7} {d['w']:>5} {d['t']-d['w']:>6} "
          f"{sign}${d['pnl']:>10,.2f} ${running:>10,.2f}")

# --- Day of Week ---
section("DAY OF WEEK ANALYSIS")
print(f"{'Day':<6} {'Trades':>7} {'Wins':>5} {'Loss':>6} {'WR%':>6} {'Net':>12}")
print("-" * 50)
for d in range(7):
    dt = [t for t in all_trades if t["entry_weekday"] == d]
    if not dt: continue
    w = len([t for t in dt if t["pnl"] > 0])
    wr = round(w / len(dt) * 100, 1)
    net = sum(t["pnl"] for t in dt)
    sign = "+" if net >= 0 else ""
    print(f"{day_names[d]:<6} {len(dt):>7} {w:>5} {len(dt)-w:>6} {wr:>5.1f}% {sign}${net:>10,.2f}")

# --- Hour of Day ---
section("HOUR OF DAY ANALYSIS")
print(f"{'Hour':>6} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'Net':>12}")
print("-" * 45)
for h in range(24):
    ht = [t for t in all_trades if t["entry_hour"] == h]
    if not ht: continue
    w = len([t for t in ht if t["pnl"] > 0])
    wr = round(w / len(ht) * 100, 1)
    net = sum(t["pnl"] for t in ht)
    sign = "+" if net >= 0 else ""
    print(f"{h:>4}:00 {len(ht):>7} {w:>5} {wr:>5.1f}% {sign}${net:>10,.2f}")

# --- BUY vs SELL ---
section("DIRECTION ANALYSIS")
for d in ["BUY", "SELL"]:
    dt = [t for t in all_trades if t["dir"] == d]
    if not dt: continue
    w = len([t for t in dt if t["pnl"] > 0])
    wr = round(w / len(dt) * 100, 1)
    net = sum(t["pnl"] for t in dt)
    sign = "+" if net >= 0 else ""
    print(f"{d}: {len(dt)} trades | {w}W/{len(dt)-w}L | WR {wr}% | {sign}${net:,.2f}")

# --- Risk Analysis ---
section("RISK ANALYSIS")
risks = [t["risk_usd"] for t in all_trades]
print(f"  Min Risk: ${min(risks):,.2f}")
print(f"  Max Risk: ${max(risks):,.2f}")
print(f"  Avg Risk: ${np.mean(risks):,.2f}")
print(f"  Median Risk: ${np.median(risks):,.2f}")

# --- MFE/MAE Analysis (losses that were in profit) ---
section("MFE ANALYSIS - LOSSES THAT REACHED 50%+ OF TP")
losses_with_profit = [t for t in all_trades if t["pnl"] <= 0 and t["mfe_r"] >= 1.0]
losses_with_profit.sort(key=lambda t: t["mfe_r"], reverse=True)
print(f"Total losses: {len([t for t in all_trades if t['pnl'] <= 0])}")
print(f"Losses that reached 1R+: {len(losses_with_profit)}")
print(f"Losses that reached 2R+: {len([t for t in losses_with_profit if t['mfe_r'] >= 2.0])}")
if losses_with_profit:
    cost = sum(t["pnl"] for t in losses_with_profit)
    print(f"Cost of reversals (1R+): ${cost:,.2f}")
    print()
    print(f"{'Date':<18} {'Sym':<8} {'Strat':<16} {'Dir':>4} {'MFE':>6} {'P&L':>10}")
    print("-" * 65)
    for t in losses_with_profit[:20]:
        print(f"{t['entry_time'][:16]:<18} {t['symbol']:<8} {t['strategy']:<16} "
              f"{t['dir']:>4} {t['mfe_r']:>5.1f}R ${t['pnl']:>+9.2f}")

# --- Consecutive Losses ---
section("STREAK ANALYSIS")
streak = 0; max_win_streak = 0; max_loss_streak = 0
cur_loss_streak_pnl = 0; max_loss_streak_pnl = 0
for t in all_trades:
    if t["pnl"] > 0:
        streak = streak + 1 if streak > 0 else 1
        max_win_streak = max(max_win_streak, streak)
        cur_loss_streak_pnl = 0
    else:
        streak = streak - 1 if streak < 0 else -1
        max_loss_streak = max(max_loss_streak, abs(streak))
        cur_loss_streak_pnl += t["pnl"]
        max_loss_streak_pnl = min(max_loss_streak_pnl, cur_loss_streak_pnl)

print(f"  Max Win Streak:  {max_win_streak}")
print(f"  Max Loss Streak: {max_loss_streak}")
print(f"  Max Loss Streak $: ${max_loss_streak_pnl:,.2f}")

# --- Worst Days ---
section("WORST 10 DAYS")
daily_pnl = defaultdict(float)
for t in all_trades:
    daily_pnl[t["exit_time"][:10]] += t["pnl"]
worst = sorted(daily_pnl.items(), key=lambda x: x[1])[:10]
print(f"{'Date':<12} {'P&L':>12}")
print("-" * 26)
for day, pnl in worst:
    print(f"{day:<12} ${pnl:>+10,.2f}")

# --- Best Days ---
section("BEST 10 DAYS")
best = sorted(daily_pnl.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"{'Date':<12} {'P&L':>12}")
print("-" * 26)
for day, pnl in best:
    print(f"{day:<12} ${pnl:>+10,.2f}")

# --- Per Symbol: Max SL in points ---
section("MAX STOP LOSS PER SYMBOL (in price points)")
for sym in ["BTCUSD", "XAUUSD", "GBPUSD"]:
    st = [t for t in all_trades if t["symbol"] == sym]
    if not st: continue
    risks_px = [t["risk_px"] for t in st]
    risks_usd = [t["risk_usd"] for t in st]
    print(f"{sym}: max_SL={max(risks_px):.2f} pts (${max(risks_usd):.2f}) | "
          f"avg_SL={np.mean(risks_px):.2f} pts (${np.mean(risks_usd):.2f}) | "
          f"min_SL={min(risks_px):.2f} pts (${min(risks_usd):.2f})")

print(f"\n{'='*80}")
print(f"  END OF REPORT")
print(f"{'='*80}")
