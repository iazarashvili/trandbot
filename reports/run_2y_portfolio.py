"""2-year portfolio backtest, 3 symbols, $1000, risk-based lot sizing."""
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

END = datetime.now()
START = END - timedelta(days=730)
BALANCE = 1000.0
RISK_PCT = 1.0  # 1% risk per trade

symbols_cfg = [
    ("BTCUSD", "config_btcusd"),
    ("XAUUSD", "config_xauusd"),
    ("GBPUSD", "config_gbpusd"),
]

# Pull data
all_data = {}
for sym, cfg_name in symbols_cfg:
    info = mt5.symbol_info(sym)
    if info is None:
        print(f"WARNING: {sym} not found")
        continue
    frames = {}
    for key, tf in [("m5", mt5.TIMEFRAME_M5), ("h1", mt5.TIMEFRAME_H1)]:
        r = mt5.copy_rates_range(sym, tf, START, END)
        if r is None or len(r) == 0:
            r = mt5.copy_rates_from_pos(sym, tf, 0, 100000)
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        frames[key] = df
    all_data[sym] = {
        "frames": frames,
        "contract": info.trade_contract_size,
        "point": info.point,
        "cfg_name": cfg_name,
    }
    print(f"{sym}: {len(frames['m5'])} M5 bars, {frames['m5']['time'].iloc[0]} to {frames['m5']['time'].iloc[-1]}")

mt5.shutdown()


def run_symbol(sym, data, shared_balance_fn):
    """Run one symbol with dynamic lot sizing based on shared balance."""
    cfg_mod = importlib.import_module(data["cfg_name"])

    from config import HTF_CANDLES_LOOKBACK, LTF_CANDLES_LOOKBACK, TREND_EMA_PERIOD
    max_wait = getattr(cfg_mod, "MAX_LTF_WAIT_CANDLES", 60)
    stop_mode = getattr(cfg_mod, "STOP_MODE", "window")
    night_start = getattr(cfg_mod, "NIGHT_START_HOUR", 0)
    night_end = getattr(cfg_mod, "NIGHT_END_HOUR", 0)
    blocked = list(getattr(cfg_mod, "BLOCKED_DAYS", []))
    skip_we = getattr(cfg_mod, "SKIP_WEEKENDS", True)
    rrr = getattr(cfg_mod, "RRR", 3.0)
    use_sweep = getattr(cfg_mod, "USE_SWEEP_FILTER", False)
    use_be = getattr(cfg_mod, "USE_BREAKEVEN", False)
    be_r = getattr(cfg_mod, "BREAKEVEN_R", 2.0) if use_be else 0.0
    use_liq = getattr(cfg_mod, "USE_LIQUIDITY_TP", True)
    use_partial = getattr(cfg_mod, "USE_PARTIAL_CLOSE", True)
    partial_trigger = getattr(cfg_mod, "PARTIAL_TRIGGER_PCT", 0.80)
    partial_close = getattr(cfg_mod, "PARTIAL_CLOSE_PCT", 0.80)
    max_risk_usd = getattr(cfg_mod, "MAX_RISK_USD", 0.0)

    m5 = data["frames"]["m5"]
    h1 = data["frames"]["h1"]
    point = data["point"]
    contract = data["contract"]

    h1_t = h1["time"].values
    m5_t = m5["time"].values
    h_idx = np.searchsorted(h1_t, m5_t - np.timedelta64(60, "m"), side="right") - 1
    o = m5["open"].to_numpy(float)
    hi = m5["high"].to_numpy(float)
    lo = m5["low"].to_numpy(float)
    cl = m5["close"].to_numpy(float)
    spread = m5["spread"].to_numpy(float) * point
    times = m5["time"]
    _weekdays = pd.to_datetime(m5["time"]).dt.weekday.to_numpy()
    _hours = pd.to_datetime(m5["time"]).dt.hour.to_numpy()

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

    for t in range(sb, len(m5) - 1):
        if pos is not None:
            p = pos
            vol = p["vol"]

            # Breakeven
            if be_r > 0 and not p.get("be"):
                td = p["risk_px"] * be_r
                if p["dir"] == "BUY" and hi[t] >= p["entry"] + td:
                    p["sl"] = p["entry"]
                    p["be"] = True
                elif p["dir"] == "SELL" and lo[t] + spread[t] <= p["entry"] - td:
                    p["sl"] = p["entry"]
                    p["be"] = True

            # Partial close
            if use_partial and not p.get("partial_done"):
                tp_dist = abs(p["tp"] - p["entry"])
                trigger_dist = tp_dist * partial_trigger
                if p["dir"] == "BUY":
                    triggered = hi[t] >= p["entry"] + trigger_dist
                else:
                    triggered = lo[t] + spread[t] <= p["entry"] - trigger_dist
                if triggered:
                    close_vol = vol * partial_close
                    remain_vol = vol - close_vol
                    partial_px = (p["entry"] + trigger_dist if p["dir"] == "BUY"
                                  else p["entry"] - trigger_dist)
                    partial_pnl = ((partial_px - p["entry"]) if p["dir"] == "BUY"
                                   else (p["entry"] - partial_px)) * close_vol * contract
                    p["partial_pnl"] = round(partial_pnl, 2)
                    p["partial_done"] = True
                    p["vol"] = remain_vol
                    p["sl"] = p["entry"]

            vol = p.get("vol", p.get("vol"))

            if p["dir"] == "BUY":
                hs = lo[t] <= p["sl"]
                ht = hi[t] >= p["tp"]
            else:
                hs = hi[t] + spread[t] >= p["sl"]
                ht = lo[t] + spread[t] <= p["tp"]

            if hs or ht:
                ep = p["sl"] if hs else p["tp"]
                remaining_pnl = ((ep - p["entry"]) if p["dir"] == "BUY"
                                 else (p["entry"] - ep)) * vol * contract
                total_pnl = remaining_pnl + p.get("partial_pnl", 0)
                p.update(
                    exit_time=str(times.iloc[t]),
                    pnl=round(total_pnl, 2),
                    exit_reason="SL" if hs else "TP",
                )
                trades.append(p)
                pos = None
                wk = None
                ab = False
            continue

        if skip_we and _weekdays[t] >= 5:
            continue
        if blocked and _weekdays[t] in blocked:
            continue
        h = _hours[t]
        if night_start > night_end:
            if h >= night_start or h < night_end:
                continue
        elif night_start != night_end and night_start <= h < night_end:
            continue

        if h_idx[t] != poi_h:
            poi_h = h_idx[t]
            if poi_h in htf_cache:
                trend, poi = htf_cache[poi_h]
            else:
                wh = h1.iloc[max(0, poi_h - HTF_CANDLES_LOOKBACK + 1): poi_h + 1]
                trend = SMCStrategy.get_htf_trend(wh, TREND_EMA_PERIOD, use_closed_candles=False)
                poi = SMCStrategy.detect_htf_poi(
                    wh, use_trend_filter=True,
                    ema_period=TREND_EMA_PERIOD, use_closed_candles=False)
                htf_cache[poi_h] = (trend, poi)
            if poi is None:
                wk = None
                ab = False

        if poi is None:
            continue
        mid = cl[t] + spread[t] / 2.0
        if not (poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom)):
            wk = None
            ab = False
            continue
        ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
        if not SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
            wk = None
            ab = False
            continue
        zk = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
        if zk != wk:
            wk = zk
            ws = t
            ab = False
        if ab:
            continue
        if (t - ws + 1) > max_wait:
            ab = True
            continue
        if SMCStrategy.is_consolidating(ltf, use_closed_candles=False):
            continue

        if use_sweep:
            sw = SMCStrategy.detect_liquidity_sweep(ltf, swing_strength=3, use_closed_candles=False)
            if sw is None:
                continue
            if poi.type == "BULLISH" and sw.direction != "BULLISH":
                continue
            if poi.type == "BEARISH" and sw.direction != "BEARISH":
                continue

        setup = SMCStrategy.check_ltf_confirmation(
            ltf, poi, rrr, use_closed_candles=False,
            stop_mode=stop_mode, buffer_atr=0.5)
        if setup is None:
            continue

        entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
        sl = setup["sl"]
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        tp = entry + risk * rrr if setup["direction"] == "BUY" else entry - risk * rrr

        # Dynamic lot sizing: 1% of current shared balance
        cur_balance = shared_balance_fn()
        risk_amount = cur_balance * RISK_PCT / 100.0
        pnl_per_lot = risk * contract
        if pnl_per_lot <= 0:
            continue
        vol = risk_amount / pnl_per_lot
        vol = max(0.01, int(vol * 100) / 100.0)

        # Hard cap
        risk_usd = risk * vol * contract
        if max_risk_usd > 0 and risk_usd > max_risk_usd:
            vol = max_risk_usd / pnl_per_lot
            vol = max(0.01, int(vol * 100) / 100.0)

        pos = {
            "dir": setup["direction"], "entry_time": str(times.iloc[t + 1]),
            "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "risk_px": round(risk, 2), "vol": vol, "entry_bar": t + 1,
            "symbol": sym,
        }

    return trades


# Run all symbols, interleave trades chronologically on shared balance
print("\nRunning backtests...")

symbol_trades = {}
for sym, cfg_name in symbols_cfg:
    if sym not in all_data:
        continue
    # First pass: collect trades with placeholder balance
    balance_ref = [BALANCE]
    symbol_trades[sym] = run_symbol(sym, all_data[sym], lambda: balance_ref[0])

# Merge all trades, sort by exit time
all_trades = []
for sym, trades in symbol_trades.items():
    for t in trades:
        t["symbol"] = sym
        all_trades.append(t)

all_trades.sort(key=lambda t: t.get("exit_time", ""))

# Replay on shared balance
balance = BALANCE
for t in all_trades:
    balance += t["pnl"]
    t["shared_balance"] = round(balance, 2)

# Monthly stats
monthly = defaultdict(lambda: defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}))
monthly_total = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})

for t in all_trades:
    month = t["exit_time"][:7]
    sym = t["symbol"]
    monthly[month][sym]["trades"] += 1
    monthly[month][sym]["pnl"] += t["pnl"]
    monthly_total[month]["trades"] += 1
    monthly_total[month]["pnl"] += t["pnl"]
    if t["pnl"] > 0:
        monthly[month][sym]["wins"] += 1
        monthly_total[month]["wins"] += 1
    else:
        monthly[month][sym]["losses"] += 1
        monthly_total[month]["losses"] += 1

syms = ["BTCUSD", "XAUUSD", "GBPUSD"]

print()
print("=" * 115)
print(f"  2-YEAR PORTFOLIO BACKTEST | $1,000 | 1% Risk Per Trade")
print(f"  BTCUSD (FVG) | XAUUSD (Sweep+FVG, BE 2R) | GBPUSD (FVG)")
print("=" * 115)
print()

header = f"{'Month':<10}"
for s in syms:
    header += f" | {s:>20}"
header += f" | {'TOTAL':>12} {'BAL':>10}"
print(header)
print("-" * len(header))

running = BALANCE
for month in sorted(monthly_total):
    row = f"{month:<10}"
    month_total = 0.0
    for s in syms:
        d = monthly[month][s]
        if d["trades"] > 0:
            sign = "+" if d["pnl"] >= 0 else ""
            row += f" | {sign}${d['pnl']:>8.2f} ({d['wins']}W/{d['losses']}L)"
        else:
            row += f" | {'---':>20}"
        month_total += d["pnl"]
    running += month_total
    sign = "+" if month_total >= 0 else ""
    row += f" | {sign}${month_total:>9.2f} ${running:>9.2f}"
    print(row)

print("-" * len(header))

# Yearly breakdown
print()
print("=== YEARLY SUMMARY ===")
yearly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "start_bal": 0, "end_bal": 0})
y_bal = BALANCE
for month in sorted(monthly_total):
    year = month[:4]
    if yearly[year]["start_bal"] == 0:
        yearly[year]["start_bal"] = round(y_bal, 2)
    yearly[year]["trades"] += monthly_total[month]["trades"]
    yearly[year]["wins"] += monthly_total[month]["wins"]
    yearly[year]["pnl"] += monthly_total[month]["pnl"]
    y_bal += monthly_total[month]["pnl"]
    yearly[year]["end_bal"] = round(y_bal, 2)

print(f"{'Year':<8} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'Net P&L':>12} {'Start':>10} {'End':>10} {'Return':>8}")
print("-" * 75)
for year in sorted(yearly):
    y = yearly[year]
    wr = round(y["wins"] / y["trades"] * 100, 1) if y["trades"] else 0
    ret = round((y["end_bal"] / y["start_bal"] - 1) * 100, 1) if y["start_bal"] > 0 else 0
    sign = "+" if y["pnl"] >= 0 else ""
    print(f"{year:<8} {y['trades']:>7} {y['wins']:>5} {wr:>5.1f}% "
          f"{sign}${y['pnl']:>10,.2f} ${y['start_bal']:>9,.2f} ${y['end_bal']:>9,.2f} {ret:>+7.1f}%")

# Per symbol summary
print()
print(f"{'Symbol':<10} {'Trades':>7} {'Wins':>5} {'Loss':>6} {'WR%':>7} {'Net P&L':>12} {'PF':>6}")
print("-" * 60)

sym_stats = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0, "gw": 0.0, "gl": 0.0})
for t in all_trades:
    s = sym_stats[t["symbol"]]
    s["t"] += 1
    s["pnl"] += t["pnl"]
    if t["pnl"] > 0:
        s["w"] += 1
        s["gw"] += t["pnl"]
    else:
        s["gl"] -= t["pnl"]

total_t = total_w = 0
total_pnl = 0.0
for sym in syms:
    s = sym_stats[sym]
    l = s["t"] - s["w"]
    wr = round(s["w"] / s["t"] * 100, 1) if s["t"] else 0
    pf = round(s["gw"] / s["gl"], 2) if s["gl"] > 0 else "-"
    sign = "+" if s["pnl"] >= 0 else ""
    print(f"{sym:<10} {s['t']:>7} {s['w']:>5} {l:>6} {wr:>6.1f}% {sign}${s['pnl']:>10,.2f} {str(pf):>6}")
    total_t += s["t"]
    total_w += s["w"]
    total_pnl += s["pnl"]

gw_all = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
gl_all = -sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)
pf_all = round(gw_all / gl_all, 2) if gl_all > 0 else "-"
wr_all = round(total_w / total_t * 100, 1) if total_t else 0
print("-" * 60)
sign = "+" if total_pnl >= 0 else ""
print(f"{'TOTAL':<10} {total_t:>7} {total_w:>5} {total_t - total_w:>6} "
      f"{wr_all:>6.1f}% {sign}${total_pnl:>10,.2f} {str(pf_all):>6}")

# Drawdown
peak = BALANCE
max_dd = 0
for t in all_trades:
    peak = max(peak, t["shared_balance"])
    dd = peak - t["shared_balance"]
    max_dd = max(max_dd, dd)
max_dd_pct = max_dd / peak * 100 if peak > 0 else 0

print(f"\n{'=' * 50}")
print(f"  ${BALANCE:,.0f} -> ${balance:,.2f} ({(balance/BALANCE-1)*100:+.1f}%)")
print(f"  MaxDD: ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
print(f"{'=' * 50}")
