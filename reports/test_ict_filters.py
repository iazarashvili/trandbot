"""Test all ICT filters on each symbol — find what helps."""
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

BALANCE = 1000.0
LOT = 0.01

symbols = [
    ("BTCUSD", "config_btcusd"),
    ("XAUUSD", "config_xauusd"),
    ("GBPUSD", "config_gbpusd"),
]

all_frames = {}
for sym, cfg_name in symbols:
    info = mt5.symbol_info(sym)
    frames = {}
    for key, tf, n in [("m5", mt5.TIMEFRAME_M5, 100000),
                        ("h1", mt5.TIMEFRAME_H1, 10000)]:
        rates = mt5.copy_rates_from_pos(sym, tf, 0, n)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        frames[key] = df
    all_frames[sym] = {
        "frames": frames,
        "contract": info.trade_contract_size,
        "point": info.point,
        "cfg_name": cfg_name,
    }
    print(f"{sym}: {len(frames['m5'])} bars, {frames['m5']['time'].iloc[0].date()} to {frames['m5']['time'].iloc[-1].date()}")

mt5.shutdown()


def run_with_filters(sym, data, filters):
    """Run backtest with specified ICT filters enabled."""
    cfg = importlib.import_module(data["cfg_name"])

    from config import HTF_CANDLES_LOOKBACK, LTF_CANDLES_LOOKBACK, TREND_EMA_PERIOD
    max_wait = getattr(cfg, "MAX_LTF_WAIT_CANDLES", 60)
    stop_mode = getattr(cfg, "STOP_MODE", "window")
    night_start = getattr(cfg, "NIGHT_START_HOUR", 0)
    night_end = getattr(cfg, "NIGHT_END_HOUR", 0)
    blocked = list(getattr(cfg, "BLOCKED_DAYS", []))
    skip_we = getattr(cfg, "SKIP_WEEKENDS", True)
    rrr = getattr(cfg, "RRR", 3.0)
    use_sweep = getattr(cfg, "USE_SWEEP_FILTER", False)
    be_r = getattr(cfg, "BREAKEVEN_R", 2.0) if getattr(cfg, "USE_BREAKEVEN", False) else 0.0

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
    balance = BALANCE
    rejections = defaultdict(int)

    for t in range(sb, len(m5) - 1):
        if pos is not None:
            p = pos
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
                       else (p["entry"] - ep)) * LOT * contract
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "balance": round(balance, 2)})
                pos = None; wk = None; ab = False
                if balance <= 0: break
            continue

        if skip_we and _weekdays[t] >= 5: continue
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
            if poi is None: wk = None; ab = False
        if poi is None: continue

        mid = cl[t] + spread[t] / 2.0
        if not (poi.bottom <= mid <= poi.top or (lo[t] <= poi.top and hi[t] >= poi.bottom)):
            wk = None; ab = False; continue
        ltf = m5.iloc[t - LTF_CANDLES_LOOKBACK + 1: t + 1]
        if not SMCStrategy.is_zone_in_play(poi, ltf, current_price=mid):
            wk = None; ab = False; continue
        zk = (poi.type, round(float(poi.top), 2), round(float(poi.bottom), 2))
        if zk != wk: wk = zk; ws = t; ab = False
        if ab: continue
        if (t - ws + 1) > max_wait: ab = True; continue
        if SMCStrategy.is_consolidating(ltf, use_closed_candles=False): continue

        # Sweep (existing)
        if use_sweep:
            sw = SMCStrategy.detect_liquidity_sweep(ltf, swing_strength=3, use_closed_candles=False)
            if sw is None: continue
            if poi.type == "BULLISH" and sw.direction != "BULLISH": continue
            if poi.type == "BEARISH" and sw.direction != "BEARISH": continue

        # FVG
        setup = SMCStrategy.check_ltf_confirmation(ltf, poi, rrr, use_closed_candles=False,
                    stop_mode=stop_mode, buffer_atr=0.5)
        if setup is None: continue

        # --- ICT FILTERS ---
        passed = True

        if "premium_discount" in filters:
            pd_zone = SMCStrategy.get_premium_discount(ltf, lookback=50, use_closed_candles=False)
            if not SMCStrategy.is_premium_discount_aligned(pd_zone, setup["direction"]):
                rejections["premium_discount"] += 1
                passed = False

        if passed and "structure_shift" in filters:
            mss = SMCStrategy.detect_structure_shift(ltf, swing_strength=3, lookback=50, use_closed_candles=False)
            if mss is None:
                rejections["no_mss"] += 1
                passed = False
            elif setup["direction"] == "BUY" and mss.direction != "BULLISH":
                rejections["mss_wrong_dir"] += 1
                passed = False
            elif setup["direction"] == "SELL" and mss.direction != "BEARISH":
                rejections["mss_wrong_dir"] += 1
                passed = False

        if passed and "asian_range" in filters:
            asian = SMCStrategy.get_asian_range(ltf, use_closed_candles=False)
            if asian is not None:
                if not SMCStrategy.is_asian_range_swept(ltf, asian, poi.type):
                    rejections["asian_not_swept"] += 1
                    passed = False

        if passed and "breaker" in filters:
            breaker = SMCStrategy.detect_breaker_block(ltf, lookback=50, use_closed_candles=False)
            if breaker is not None:
                if setup["direction"] == "BUY" and breaker.type == "BEARISH" and setup["entry"] < breaker.top:
                    rejections["breaker_block"] += 1
                    passed = False
                elif setup["direction"] == "SELL" and breaker.type == "BULLISH" and setup["entry"] > breaker.bottom:
                    rejections["breaker_block"] += 1
                    passed = False

        if passed and "po3" in filters:
            po3 = SMCStrategy.detect_po3(ltf, use_closed_candles=False)
            if po3 is None:
                rejections["no_po3"] += 1
                passed = False
            elif setup["direction"] == "BUY" and po3["direction"] != "BULLISH":
                rejections["po3_wrong_dir"] += 1
                passed = False
            elif setup["direction"] == "SELL" and po3["direction"] != "BEARISH":
                rejections["po3_wrong_dir"] += 1
                passed = False

        if passed and "ifvg" in filters:
            ifvg = SMCStrategy.detect_ifvg(ltf, lookback=50, use_closed_candles=False)
            if ifvg is not None:
                if setup["direction"] == "BUY" and ifvg["type"] == "BEARISH" and setup["entry"] < ifvg["top"]:
                    rejections["ifvg_block"] += 1
                    passed = False
                elif setup["direction"] == "SELL" and ifvg["type"] == "BULLISH" and setup["entry"] > ifvg["bottom"]:
                    rejections["ifvg_block"] += 1
                    passed = False

        if not passed:
            continue

        entry = o[t + 1] + spread[t + 1] if setup["direction"] == "BUY" else o[t + 1]
        sl = setup["sl"]; risk = abs(entry - sl)
        if risk <= 0: continue
        tp = entry + risk * rrr if setup["direction"] == "BUY" else entry - risk * rrr
        pos = {"dir": setup["direction"], "entry": round(entry, 2), "sl": round(sl, 2),
               "tp": round(tp, 2), "risk_px": round(risk, 2), "entry_bar": t + 1,
               "entry_time": str(times.iloc[t + 1])}

    wins = len([x for x in trades if x["pnl"] > 0])
    net = balance - BALANCE
    gw = sum(x["pnl"] for x in trades if x["pnl"] > 0)
    gl = -sum(x["pnl"] for x in trades if x["pnl"] <= 0)
    pf = round(gw / gl, 2) if gl > 0 else "-"
    wr = round(wins / len(trades) * 100, 1) if trades else 0
    peak = BALANCE; mdd = 0
    for tr in trades:
        peak = max(peak, tr["balance"]); dd = peak - tr["balance"]; mdd = max(mdd, dd)
    return {"t": len(trades), "w": wins, "wr": wr, "pf": pf,
            "net": round(net, 2), "mdd": round(mdd / peak * 100, 1) if peak else 0,
            "rej": dict(rejections)}


# Test each filter individually per symbol
filter_configs = [
    ("Baseline (current)",      []),
    ("+ Premium/Discount",      ["premium_discount"]),
    ("+ Structure Shift",       ["structure_shift"]),
    ("+ Asian Range",           ["asian_range"]),
    ("+ Breaker Block",         ["breaker"]),
    ("+ PO3",                   ["po3"]),
    ("+ IFVG",                  ["ifvg"]),
    ("+ PD + MSS",              ["premium_discount", "structure_shift"]),
    ("+ PD + Breaker",          ["premium_discount", "breaker"]),
    ("+ PD + IFVG",             ["premium_discount", "ifvg"]),
]

print()
for sym, cfg_name in symbols:
    d = all_frames[sym]
    # Reload config
    if cfg_name in sys.modules:
        del sys.modules[cfg_name]
    importlib.import_module(cfg_name)

    print(f"{'='*80}")
    print(f"  {sym}")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PF':>6} {'Net':>10} {'MaxDD':>6}")
    print("-" * 70)

    for label, flt in filter_configs:
        r = run_with_filters(sym, d, flt)
        sign = "+" if r["net"] >= 0 else ""
        print(f"{label:<25} {r['t']:>7} {r['w']:>5} {r['wr']:>5.1f}% {str(r['pf']):>6} "
              f"{sign}${r['net']:>8.2f} {r['mdd']:>5.1f}%")

    print()
