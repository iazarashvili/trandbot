"""Build HTML report from multi-strategy backtest."""
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
    all_data[sym] = {"frames": frames, "contract": info.trade_contract_size,
                     "point": info.point, "cfg_name": cfg_name}
mt5.shutdown()


def run_multi(sym, data):
    cfg = importlib.import_module(data["cfg_name"])
    from config import (HTF_CANDLES_LOOKBACK, LTF_CANDLES_LOOKBACK,
                        TREND_EMA_PERIOD, SKIP_WEEKENDS, SWING_STRENGTH,
                        MAX_LTF_WAIT_CANDLES)
    stop_mode = getattr(cfg, "STOP_MODE", "window")
    night_start = getattr(cfg, "NIGHT_START_HOUR", 0)
    night_end = getattr(cfg, "NIGHT_END_HOUR", 0)
    blocked = list(getattr(cfg, "BLOCKED_DAYS", []))
    rrr = getattr(cfg, "RRR", 3.0)
    be_r = getattr(cfg, "BREAKEVEN_R", 2.0) if getattr(cfg, "USE_BREAKEVEN", False) else 0.0

    m5, h1, m15 = data["frames"]["m5"], data["frames"]["h1"], data["frames"]["m15"]
    point, contract = data["point"], data["contract"]
    h1_t, m5_t, m15_t = h1["time"].values, m5["time"].values, m15["time"].values
    h_idx = np.searchsorted(h1_t, m5_t - np.timedelta64(60, "m"), side="right") - 1
    liq_idx = np.searchsorted(m15_t, m5_t - np.timedelta64(15, "m"), side="right") - 1
    o, hi, lo, cl = m5["open"].to_numpy(float), m5["high"].to_numpy(float), m5["low"].to_numpy(float), m5["close"].to_numpy(float)
    spread = m5["spread"].to_numpy(float) * point
    times = m5["time"]
    _wd = pd.to_datetime(times).dt.weekday.to_numpy()
    _hr = pd.to_datetime(times).dt.hour.to_numpy()
    _mn = pd.to_datetime(times).dt.minute.to_numpy()

    sb = LTF_CANDLES_LOOKBACK
    while sb < len(m5) and h_idx[sb] < HTF_CANDLES_LOOKBACK: sb += 1
    htf_cache = {}; poi = None; poi_h = -1; pos = None; trades = []; balance = BALANCE

    for t in range(sb, len(m5) - 1):
        if pos is not None:
            p = pos; vol = p["vol"]
            if be_r > 0 and not p.get("be"):
                td = p["risk_px"] * be_r
                if p["dir"]=="BUY" and hi[t]>=p["entry"]+td: p["sl"]=p["entry"]; p["be"]=True
                elif p["dir"]=="SELL" and lo[t]+spread[t]<=p["entry"]-td: p["sl"]=p["entry"]; p["be"]=True
            if p["dir"]=="BUY":
                p["mfe"]=max(p.get("mfe",0),hi[t]-p["entry"]); p["mae"]=max(p.get("mae",0),p["entry"]-lo[t])
                hs=lo[t]<=p["sl"]; ht=hi[t]>=p["tp"]
            else:
                p["mfe"]=max(p.get("mfe",0),p["entry"]-(lo[t]+spread[t])); p["mae"]=max(p.get("mae",0),(hi[t]+spread[t])-p["entry"])
                hs=hi[t]+spread[t]>=p["sl"]; ht=lo[t]+spread[t]<=p["tp"]
            if hs or ht:
                ep=p["sl"] if hs else p["tp"]
                pnl=((ep-p["entry"]) if p["dir"]=="BUY" else (p["entry"]-ep))*vol*contract
                balance+=pnl; ru=p["risk_px"]*vol*contract
                trades.append({"pnl":round(pnl,2),"balance":round(balance,2),"entry_time":p["entry_time"],
                    "exit_time":str(times.iloc[t]),"dir":p["dir"],"strategy":p["strategy"],
                    "exit_reason":"SL" if hs else "TP","r_multiple":round(pnl/ru,2) if ru>0 else 0,
                    "risk_usd":round(ru,2),"entry":p["entry"],"sl":p["sl"],"tp":p["tp"],
                    "risk_px":p["risk_px"],"vol":vol,"bars_held":t-p["entry_bar"],
                    "mfe_r":round(p.get("mfe",0)/p["risk_px"],2) if p["risk_px"]>0 else 0,
                    "entry_hour":p["eh"],"entry_weekday":p["ew"],"symbol":sym})
                pos=None
                if balance<=0: break
            continue
        if SKIP_WEEKENDS and _wd[t]>=5: continue
        if blocked and _wd[t] in blocked: continue
        h=_hr[t]
        if night_start>night_end:
            if h>=night_start or h<night_end: continue
        elif night_start!=night_end and night_start<=h<night_end: continue
        if h_idx[t]!=poi_h:
            poi_h=h_idx[t]
            if poi_h in htf_cache: trend,poi=htf_cache[poi_h]
            else:
                wh=h1.iloc[max(0,poi_h-HTF_CANDLES_LOOKBACK+1):poi_h+1]
                trend=SMCStrategy.get_htf_trend(wh,TREND_EMA_PERIOD,use_closed_candles=False)
                poi=SMCStrategy.detect_htf_poi(wh,use_trend_filter=True,ema_period=TREND_EMA_PERIOD,use_closed_candles=False)
                htf_cache[poi_h]=(trend,poi)
        ltf=m5.iloc[t-LTF_CANDLES_LOOKBACK+1:t+1]; mid=cl[t]+spread[t]/2.0
        setup=None; sn=None
        # AMD
        asian=SMCStrategy.get_asian_range(ltf,use_closed_candles=False)
        if asian:
            a=SMCStrategy.check_amd_setup(ltf,asian.high,asian.low,swing_strength=3,rrr_fallback=rrr,use_closed_candles=False)
            if a: setup=a; sn="AMD"
        # Silver Bullet
        if not setup:
            li=liq_idx[t]
            if li>=0:
                ls=m15.iloc[max(0,li-200):li+1]
                s=SMCStrategy.check_silver_bullet(ltf,ls,int(_hr[t]),int(_mn[t]),3,rrr,False)
                if s: setup=s; sn="SILVER_BULLET"
        # Sweep+FVG
        if not setup and poi:
            if (poi.bottom<=mid<=poi.top or (lo[t]<=poi.top and hi[t]>=poi.bottom)):
                if SMCStrategy.is_zone_in_play(poi,ltf,current_price=mid):
                    sw=SMCStrategy.detect_liquidity_sweep(ltf,swing_strength=3,use_closed_candles=False)
                    if sw and ((poi.type=="BULLISH" and sw.direction=="BULLISH") or (poi.type=="BEARISH" and sw.direction=="BEARISH")):
                        s=SMCStrategy.check_ltf_confirmation(ltf,poi,rrr,use_closed_candles=False,stop_mode=stop_mode,buffer_atr=0.5)
                        if s: setup=s; sn="SWEEP_FVG"
        if not setup: continue
        entry=o[t+1]+spread[t+1] if setup["direction"]=="BUY" else o[t+1]
        sl=setup["sl"]; risk=abs(entry-sl)
        if risk<=0: continue
        tp=setup["tp"]; ra=balance*RISK_PCT/100.0; ppl=risk*contract
        if ppl<=0: continue
        vol=max(0.01,int(ra/ppl*100)/100.0)
        pos={"dir":setup["direction"],"entry_time":str(times.iloc[t+1]),"entry":round(entry,2),
             "sl":round(sl,2),"tp":round(tp,2),"risk_px":round(risk,2),"vol":vol,"entry_bar":t+1,
             "strategy":sn,"eh":int(_hr[t+1]) if t+1<len(_hr) else int(_hr[t]),
             "ew":int(_wd[t+1]) if t+1<len(_wd) else int(_wd[t])}
    return trades


print("Running backtests...")
all_trades = []
for sym, cfg_name in symbols_cfg:
    if cfg_name in sys.modules: del sys.modules[cfg_name]
    all_trades.extend(run_multi(sym, all_data[sym]))
all_trades.sort(key=lambda t: t.get("exit_time", ""))
balance = BALANCE
equity = [{"time": "", "balance": BALANCE}]
for t in all_trades:
    balance += t["pnl"]
    t["shared_balance"] = round(balance, 2)
    equity.append({"time": t["exit_time"], "balance": round(balance, 2)})

# Build HTML
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
total_t = len(all_trades)
total_w = len([t for t in all_trades if t["pnl"] > 0])
total_pnl = sum(t["pnl"] for t in all_trades)
gw = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
gl = -sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)
pf = round(gw / gl, 2) if gl > 0 else 0
peak = BALANCE; max_dd = 0
for t in all_trades:
    peak = max(peak, t["shared_balance"])
    dd = peak - t["shared_balance"]; max_dd = max(max_dd, dd)
max_dd_pct = round(max_dd / peak * 100, 1) if peak else 0

# Monthly data
monthly = defaultdict(lambda: {"t": 0, "w": 0, "pnl": 0.0})
for t in all_trades:
    m = t["exit_time"][:7]; monthly[m]["t"] += 1; monthly[m]["pnl"] += t["pnl"]
    if t["pnl"] > 0: monthly[m]["w"] += 1

# Strategy data
strat_data = {}
for strat in ["AMD", "SILVER_BULLET", "SWEEP_FVG"]:
    st = [t for t in all_trades if t["strategy"] == strat]
    w = len([t for t in st if t["pnl"] > 0])
    g = sum(t["pnl"] for t in st if t["pnl"] > 0)
    l = -sum(t["pnl"] for t in st if t["pnl"] <= 0)
    strat_data[strat] = {"t": len(st), "w": w, "wr": round(w/len(st)*100,1) if st else 0,
                         "pf": round(g/l,2) if l>0 else 0, "net": round(sum(t["pnl"] for t in st),2)}

# Hour data
hour_data = {}
for h in range(24):
    ht = [t for t in all_trades if t["entry_hour"] == h]
    if not ht: continue
    w = len([t for t in ht if t["pnl"] > 0])
    hour_data[h] = {"t": len(ht), "w": w, "net": round(sum(t["pnl"] for t in ht), 2)}

# Day data
day_data = {}
for d in range(7):
    dt = [t for t in all_trades if t["entry_weekday"] == d]
    if not dt: continue
    w = len([t for t in dt if t["pnl"] > 0])
    day_data[d] = {"t": len(dt), "w": w, "net": round(sum(t["pnl"] for t in dt), 2)}

# Symbol data
sym_data = {}
for sym in ["BTCUSD", "XAUUSD", "GBPUSD"]:
    st = [t for t in all_trades if t["symbol"] == sym]
    w = len([t for t in st if t["pnl"] > 0])
    g = sum(t["pnl"] for t in st if t["pnl"] > 0)
    l = -sum(t["pnl"] for t in st if t["pnl"] <= 0)
    sym_data[sym] = {"t": len(st), "w": w, "wr": round(w/len(st)*100,1) if st else 0,
                     "pf": round(g/l,2) if l>0 else 0, "net": round(sum(t["pnl"] for t in st),2)}

# Equity curve JSON
eq_json = ",".join([f'["{e["time"][:16]}",{e["balance"]}]' for e in equity[::max(1,len(equity)//200)]])

# Monthly chart data
months_sorted = sorted(monthly.keys())
monthly_labels = ",".join([f'"{m}"' for m in months_sorted])
monthly_vals = ",".join([f'{monthly[m]["pnl"]:.0f}' for m in months_sorted])
monthly_colors = ",".join([f'"#089981"' if monthly[m]["pnl"] >= 0 else f'"#F23645"' for m in months_sorted])

# Worst/best days
daily_pnl = defaultdict(float)
for t in all_trades: daily_pnl[t["exit_time"][:10]] += t["pnl"]
worst10 = sorted(daily_pnl.items(), key=lambda x: x[1])[:10]
best10 = sorted(daily_pnl.items(), key=lambda x: x[1], reverse=True)[:10]

# Streak
streak = 0; mws = 0; mls = 0; mlsp = 0; clsp = 0
for t in all_trades:
    if t["pnl"] > 0: streak = streak+1 if streak>0 else 1; mws=max(mws,streak); clsp=0
    else: streak = streak-1 if streak<0 else -1; mls=max(mls,abs(streak)); clsp+=t["pnl"]; mlsp=min(mlsp,clsp)

# Symbol x Strategy matrix
matrix = {}
for sym in ["BTCUSD","XAUUSD","GBPUSD"]:
    for strat in ["AMD","SILVER_BULLET","SWEEP_FVG"]:
        st=[t for t in all_trades if t["symbol"]==sym and t["strategy"]==strat]
        if st:
            w=len([t for t in st if t["pnl"]>0])
            matrix[f"{sym}_{strat}"]={"t":len(st),"w":w,"wr":round(w/len(st)*100,1),"net":round(sum(t["pnl"] for t in st),2)}


html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMC Bot — Multi-Strategy Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0f1117; color:#d1d4dc; line-height:1.6; }}
.container {{ max-width:1400px; margin:0 auto; padding:20px; }}
h1 {{ color:#fff; font-size:28px; margin-bottom:5px; }}
h2 {{ color:#2962ff; font-size:20px; margin:30px 0 15px; border-bottom:1px solid #2a2e39; padding-bottom:8px; }}
.subtitle {{ color:#787b86; font-size:14px; margin-bottom:30px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:15px; margin-bottom:30px; }}
.card {{ background:#1e222d; border-radius:12px; padding:20px; text-align:center; }}
.card .value {{ font-size:28px; font-weight:bold; color:#fff; }}
.card .label {{ font-size:12px; color:#787b86; margin-top:5px; text-transform:uppercase; }}
.card.green .value {{ color:#089981; }}
.card.red .value {{ color:#F23645; }}
.card.blue .value {{ color:#2962ff; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:20px; background:#1e222d; border-radius:8px; overflow:hidden; }}
th {{ background:#2a2e39; color:#787b86; padding:10px 15px; text-align:left; font-size:12px; text-transform:uppercase; }}
td {{ padding:10px 15px; border-top:1px solid #2a2e39; font-size:13px; }}
tr:hover {{ background:#262a35; }}
.pos {{ color:#089981; }}
.neg {{ color:#F23645; }}
.chart-box {{ background:#1e222d; border-radius:12px; padding:20px; margin-bottom:20px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
@media(max-width:800px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:4px; font-size:11px; font-weight:bold; }}
.badge-amd {{ background:#2962ff33; color:#2962ff; }}
.badge-sb {{ background:#089981333; color:#089981; }}
.badge-sweep {{ background:#f5a62333; color:#f5a623; }}
</style>
</head>
<body>
<div class="container">

<h1>SMC Bot — Multi-Strategy Report</h1>
<p class="subtitle">$10,000 | 1% Risk | AMD + Silver Bullet + Sweep+FVG | BTCUSD + XAUUSD + GBPUSD</p>

<!-- KPI Cards -->
<div class="cards">
<div class="card green"><div class="value">${balance:,.0f}</div><div class="label">Final Balance</div></div>
<div class="card green"><div class="value">+{(balance/BALANCE-1)*100:.1f}%</div><div class="label">Total Return</div></div>
<div class="card blue"><div class="value">{total_t}</div><div class="label">Total Trades</div></div>
<div class="card"><div class="value">{round(total_w/total_t*100,1)}%</div><div class="label">Win Rate</div></div>
<div class="card blue"><div class="value">{pf}</div><div class="label">Profit Factor</div></div>
<div class="card red"><div class="value">{max_dd_pct}%</div><div class="label">Max Drawdown</div></div>
<div class="card"><div class="value">{mws}</div><div class="label">Max Win Streak</div></div>
<div class="card red"><div class="value">{mls}</div><div class="label">Max Loss Streak</div></div>
</div>

<!-- Equity Curve -->
<div class="chart-box">
<h2 style="margin-top:0">Equity Curve</h2>
<canvas id="equityChart" height="80"></canvas>
</div>

<!-- Monthly P&L -->
<div class="chart-box">
<h2 style="margin-top:0">Monthly P&L</h2>
<canvas id="monthlyChart" height="60"></canvas>
</div>

<!-- Strategy Breakdown -->
<h2>Strategy Performance</h2>
<table>
<tr><th>Strategy</th><th>Trades</th><th>Wins</th><th>Win Rate</th><th>PF</th><th>Net P&L</th></tr>
{"".join(f'<tr><td><span class="badge badge-{"amd" if s=="AMD" else "sb" if s=="SILVER_BULLET" else "sweep"}">{s}</span></td><td>{strat_data[s]["t"]}</td><td>{strat_data[s]["w"]}</td><td>{strat_data[s]["wr"]}%</td><td>{strat_data[s]["pf"]}</td><td class="{"pos" if strat_data[s]["net"]>=0 else "neg"}">${strat_data[s]["net"]:+,.2f}</td></tr>' for s in ["AMD","SILVER_BULLET","SWEEP_FVG"])}
</table>

<!-- Symbol Breakdown -->
<h2>Symbol Performance</h2>
<table>
<tr><th>Symbol</th><th>Trades</th><th>Wins</th><th>Win Rate</th><th>PF</th><th>Net P&L</th></tr>
{"".join(f'<tr><td><b>{s}</b></td><td>{sym_data[s]["t"]}</td><td>{sym_data[s]["w"]}</td><td>{sym_data[s]["wr"]}%</td><td>{sym_data[s]["pf"]}</td><td class="{"pos" if sym_data[s]["net"]>=0 else "neg"}">${sym_data[s]["net"]:+,.2f}</td></tr>' for s in ["BTCUSD","XAUUSD","GBPUSD"])}
</table>

<!-- Symbol x Strategy Matrix -->
<h2>Symbol x Strategy Matrix</h2>
<table>
<tr><th>Symbol</th><th>Strategy</th><th>Trades</th><th>Wins</th><th>WR%</th><th>Net P&L</th></tr>
{"".join(f'<tr><td>{k.split("_")[0]}</td><td><span class="badge badge-{"amd" if "AMD" in k else "sb" if "SILVER" in k else "sweep"}">{"_".join(k.split("_")[1:])}</span></td><td>{v["t"]}</td><td>{v["w"]}</td><td>{v["wr"]}%</td><td class="{"pos" if v["net"]>=0 else "neg"}">${v["net"]:+,.2f}</td></tr>' for k,v in sorted(matrix.items()))}
</table>

<div class="grid2">
<!-- Day of Week -->
<div>
<h2>Day of Week</h2>
<table>
<tr><th>Day</th><th>Trades</th><th>Wins</th><th>WR%</th><th>Net P&L</th></tr>
{"".join(f'<tr><td>{day_names[d]}</td><td>{day_data[d]["t"]}</td><td>{day_data[d]["w"]}</td><td>{round(day_data[d]["w"]/day_data[d]["t"]*100,1)}%</td><td class="{"pos" if day_data[d]["net"]>=0 else "neg"}">${day_data[d]["net"]:+,.2f}</td></tr>' for d in sorted(day_data))}
</table>
</div>

<!-- Direction -->
<div>
<h2>Direction Analysis</h2>
<table>
<tr><th>Dir</th><th>Trades</th><th>Wins</th><th>WR%</th><th>Net P&L</th></tr>
{"".join(f'<tr><td><b>{d}</b></td><td>{len([t for t in all_trades if t["dir"]==d])}</td><td>{len([t for t in all_trades if t["dir"]==d and t["pnl"]>0])}</td><td>{round(len([t for t in all_trades if t["dir"]==d and t["pnl"]>0])/len([t for t in all_trades if t["dir"]==d])*100,1)}%</td><td class="{"pos" if sum(t["pnl"] for t in all_trades if t["dir"]==d)>=0 else "neg"}">${sum(t["pnl"] for t in all_trades if t["dir"]==d):+,.2f}</td></tr>' for d in ["BUY","SELL"])}
</table>
</div>
</div>

<!-- Hour of Day -->
<h2>Hour of Day</h2>
<table>
<tr><th>Hour</th><th>Trades</th><th>Wins</th><th>WR%</th><th>Net P&L</th></tr>
{"".join(f'<tr><td>{h}:00</td><td>{hour_data[h]["t"]}</td><td>{hour_data[h]["w"]}</td><td>{round(hour_data[h]["w"]/hour_data[h]["t"]*100,1)}%</td><td class="{"pos" if hour_data[h]["net"]>=0 else "neg"}">${hour_data[h]["net"]:+,.2f}</td></tr>' for h in sorted(hour_data))}
</table>

<div class="grid2">
<!-- Worst Days -->
<div>
<h2>Worst 10 Days</h2>
<table>
<tr><th>Date</th><th>P&L</th></tr>
{"".join(f'<tr><td>{d}</td><td class="neg">${p:+,.2f}</td></tr>' for d,p in worst10)}
</table>
</div>
<!-- Best Days -->
<div>
<h2>Best 10 Days</h2>
<table>
<tr><th>Date</th><th>P&L</th></tr>
{"".join(f'<tr><td>{d}</td><td class="pos">${p:+,.2f}</td></tr>' for d,p in best10)}
</table>
</div>
</div>

<!-- Monthly Table -->
<h2>Monthly Breakdown</h2>
<table>
<tr><th>Month</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Net P&L</th><th>Balance</th></tr>
"""

running = BALANCE
for m in months_sorted:
    d = monthly[m]; running += d["pnl"]
    cls = "pos" if d["pnl"] >= 0 else "neg"
    html += f'<tr><td>{m}</td><td>{d["t"]}</td><td>{d["w"]}</td><td>{d["t"]-d["w"]}</td><td class="{cls}">${d["pnl"]:+,.2f}</td><td>${running:,.2f}</td></tr>\n'

html += f"""</table>

<!-- All Trades Table -->
<h2>All Trades ({total_t} total)</h2>
<div style="max-height:600px;overflow-y:auto;border-radius:8px;">
<table>
<tr><th>#</th><th>Date</th><th>Symbol</th><th>Strategy</th><th>Dir</th><th>Entry</th><th>SL</th><th>TP</th><th>Lots</th><th>Result</th><th>P&L</th><th>R</th><th>Balance</th></tr>
"""

for i, t in enumerate(all_trades):
    cls = "pos" if t["pnl"] > 0 else "neg"
    badge = "amd" if t["strategy"]=="AMD" else "sb" if t["strategy"]=="SILVER_BULLET" else "sweep"
    result_icon = "&#10004;" if t["pnl"] > 0 else "&#10008;" if t["pnl"] < 0 else "&#8212;"
    html += (f'<tr><td>{i+1}</td><td>{t["entry_time"][:16]}</td><td>{t["symbol"]}</td>'
             f'<td><span class="badge badge-{badge}">{t["strategy"]}</span></td>'
             f'<td>{t["dir"]}</td><td>{t["entry"]}</td><td>{t["sl"]}</td><td>{t["tp"]}</td>'
             f'<td>{t["vol"]:.2f}</td><td class="{cls}">{result_icon} {t["exit_reason"]}</td>'
             f'<td class="{cls}">${t["pnl"]:+,.2f}</td><td class="{cls}">{t["r_multiple"]:+.2f}R</td>'
             f'<td>${t["shared_balance"]:,.2f}</td></tr>\n')

html += """</table></div>

"""

html += f"""
<h2>Risk Stats</h2>
<div class="cards">
<div class="card"><div class="value">${np.mean([t["risk_usd"] for t in all_trades]):.0f}</div><div class="label">Avg Risk/Trade</div></div>
<div class="card"><div class="value">${max(t["pnl"] for t in all_trades):,.0f}</div><div class="label">Best Trade</div></div>
<div class="card red"><div class="value">${min(t["pnl"] for t in all_trades):,.0f}</div><div class="label">Worst Trade</div></div>
<div class="card"><div class="value">{round(np.mean([t["bars_held"] for t in all_trades]))}</div><div class="label">Avg Bars Held</div></div>
</div>

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
      fill: true,
      tension: 0.1,
      pointRadius: 0,
      borderWidth: 2
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
    datasets: [{{
      data: [{monthly_vals}],
      backgroundColor: [{monthly_colors}],
      borderRadius: 4
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#787b86' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#787b86', callback: v => '$'+v.toLocaleString() }}, grid: {{ color: '#2a2e39' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

out = Path(__file__).resolve().parent / "multi_strategy_report.html"
out.write_text(html, encoding="utf-8")
print(f"\nReport saved: {out}")
print(f"Open in browser: file:///{str(out).replace(chr(92), '/')}")
