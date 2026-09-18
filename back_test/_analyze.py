import json
from datetime import datetime
from collections import defaultdict
import statistics

with open(r"C:\git hub\trandbot\back_test\results_btcusd.json", "r") as f:
    data = json.load(f)

summary = data["summary"]
trades = data["trades"]

# Parse each trade
parsed = []
for t in trades:
    et = datetime.fromisoformat(t["entry_time"])
    xt = datetime.fromisoformat(t["exit_time"]) if t.get("exit_time") else None
    dow = et.weekday()
    h = et.hour
    parsed.append({
        **t,
        "entry_dt": et,
        "exit_dt": xt,
        "hour": h,
        "dow": dow,
        "dow_name": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow],
        "month": et.strftime("%Y-%m"),
        "session": (
            "Asian(0-6)"    if 0  <= h < 6  else
            "London(6-12)"  if 6  <= h < 12 else
            "NY_AM(12-17)"  if 12 <= h < 17 else
            "NY_PM(17-21)"  if 17 <= h < 21 else
            "Late(21-24)"
        ),
        "duration_cat": (
            "Short(<10)"    if t["bars_held"] < 10 else
            "Medium(10-30)" if t["bars_held"] <= 30 else
            "Long(30+)"
        ),
        "win": t["pnl"] > 0,
    })

def stats(group):
    if not group:
        return {"n":0,"wins":0,"losses":0,"wr":0,"total_pnl":0,"avg_pnl":0,"avg_win":0,"avg_loss":0,"pf":0,"total_r":0,"avg_r":0}
    wins   = [x for x in group if x["win"]]
    losses = [x for x in group if not x["win"]]
    n = len(group)
    w = len(wins)
    l = len(losses)
    tp  = sum(x["pnl"] for x in group)
    tw  = sum(x["pnl"] for x in wins)  if wins else 0
    tl  = sum(x["pnl"] for x in losses) if losses else 0
    tr  = sum(x["r_multiple"] for x in group)
    pf  = (tw / abs(tl)) if tl < 0 else (float('inf') if tw > 0 else 0)
    return {
        "n": n, "wins": w, "losses": l,
        "wr": round(w/n*100,1),
        "total_pnl": round(tp,2),
        "avg_pnl": round(tp/n,2),
        "avg_win":  round(tw/w,2)  if w else 0,
        "avg_loss": round(tl/l,2)  if l else 0,
        "pf": round(pf,2),
        "total_r": round(tr,2),
        "avg_r": round(tr/n,3),
    }

SEP = "=" * 72

# ============================================================
# SUMMARY
# ============================================================
print(SEP)
print("BTCUSD BACKTEST — DEEP ANALYSIS REPORT")
print(SEP)
print(f"Period  : {summary['period_from']} to {summary['period_to']} ({summary['days']:.1f} days)")
print(f"Balance : ${summary['start_balance']:,.2f} -> ${summary['end_balance']:,.2f}")
print(f"Net P&L : ${summary['net_pnl']:,.2f}  ({summary['return_pct']:.2f}%)")
print(f"Trades  : {summary['trades']}  |  Wins: {summary['wins']}  |  Losses: {summary['losses']}")
print(f"Win Rate: {summary['win_rate']:.1f}%  |  Profit Factor: {summary['profit_factor']:.2f}")
print(f"Avg Win : ${summary['avg_win']:,.2f}  |  Avg Loss: ${summary['avg_loss']:,.2f}")
print(f"Total R : {summary['total_r']:.1f}  |  Expectancy/trade: {summary['expectancy_r']:.3f}R")
print(f"Max DD  : ${summary['max_drawdown']:,.2f} ({summary['max_drawdown_pct']:.1f}%)")
print(f"Streaks : Best={summary['longest_win_streak']} wins  |  Worst={summary['longest_loss_streak']} losses")
print(f"Avg Hold: {summary['avg_bars_held']:.1f} bars (x5m = ~{summary['avg_bars_held']*5/60:.1f}h)")

# ============================================================
# 1. BY STRATEGY
# ============================================================
print()
print(SEP)
print("1. BY STRATEGY")
print(SEP)
strats = defaultdict(list)
for t in parsed:
    strats[t["strategy"]].append(t)

print(f"{'Strategy':<16} {'N':>4} {'W':>4} {'L':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgPnL':>8} {'AvgW':>8} {'AvgL':>8} {'PF':>5} {'TotalR':>7} {'AvgR':>7}")
print("-"*80)
for name in sorted(strats):
    s = stats(strats[name])
    print(f"{name:<16} {s['n']:>4} {s['wins']:>4} {s['losses']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_pnl']:>8.2f} {s['avg_win']:>8.2f} {s['avg_loss']:>8.2f} {s['pf']:>5.2f} {s['total_r']:>7.2f} {s['avg_r']:>7.3f}")

# ============================================================
# 2. BY HOUR OF DAY
# ============================================================
print()
print(SEP)
print("2. BY HOUR OF DAY (UTC)")
print(SEP)
hours = defaultdict(list)
for t in parsed:
    hours[t["hour"]].append(t)

print(f"{'Hour':>6} {'N':>4} {'W':>3} {'L':>3} {'WR%':>6} {'TotalPnL':>10} {'AvgPnL':>8} {'AvgR':>7}  bar")
print("-"*65)
for h in sorted(hours):
    s = stats(hours[h])
    bar = "##" * int(abs(s['total_pnl'])//20) if s['total_pnl'] != 0 else ""
    sign = "+" if s['total_pnl'] >= 0 else "-"
    print(f"  {h:02d}:xx {s['n']:>4} {s['wins']:>3} {s['losses']:>3} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_pnl']:>8.2f} {s['avg_r']:>7.3f}  {sign}{bar}")

# ============================================================
# 3. BY DAY OF WEEK
# ============================================================
print()
print(SEP)
print("3. BY DAY OF WEEK")
print(SEP)
days = defaultdict(list)
for t in parsed:
    days[(t["dow"], t["dow_name"])].append(t)

print(f"{'Day':<6} {'N':>4} {'W':>4} {'L':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgPnL':>8} {'AvgR':>7} {'PF':>5}")
print("-"*60)
for (di, dn) in sorted(days):
    s = stats(days[(di,dn)])
    flag = "  <-- LOSING" if s['total_pnl'] < 0 else ""
    print(f"{dn:<6} {s['n']:>4} {s['wins']:>4} {s['losses']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_pnl']:>8.2f} {s['avg_r']:>7.3f} {s['pf']:>5.2f}{flag}")

# ============================================================
# 4. BY SESSION
# ============================================================
print()
print(SEP)
print("4. BY SESSION")
print(SEP)
sessions = defaultdict(list)
for t in parsed:
    sessions[t["session"]].append(t)

print(f"{'Session':<16} {'N':>4} {'W':>4} {'L':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgPnL':>8} {'PF':>5} {'AvgR':>7}")
print("-"*72)
for name in ["Asian(0-6)","London(6-12)","NY_AM(12-17)","NY_PM(17-21)","Late(21-24)"]:
    if name in sessions:
        s = stats(sessions[name])
        flag = "  <-- BEST" if s['total_pnl'] == max(stats(sessions[n])['total_pnl'] for n in sessions) else ""
        print(f"{name:<16} {s['n']:>4} {s['wins']:>4} {s['losses']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_pnl']:>8.2f} {s['pf']:>5.2f} {s['avg_r']:>7.3f}{flag}")

# ============================================================
# 5. BY DIRECTION
# ============================================================
print()
print(SEP)
print("5. BY DIRECTION")
print(SEP)
dirs = defaultdict(list)
for t in parsed:
    dirs[t["dir"]].append(t)

print(f"{'Dir':<6} {'N':>4} {'W':>4} {'L':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgPnL':>8} {'AvgR':>7} {'PF':>5}")
print("-"*60)
for d in sorted(dirs):
    s = stats(dirs[d])
    print(f"{d:<6} {s['n']:>4} {s['wins']:>4} {s['losses']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_pnl']:>8.2f} {s['avg_r']:>7.3f} {s['pf']:>5.2f}")

# Also dir x strategy
print()
print("Direction x Strategy breakdown:")
dir_strat = defaultdict(list)
for t in parsed:
    dir_strat[(t["dir"], t["strategy"])].append(t)
print(f"{'Dir':<6} {'Strat':<16} {'N':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgR':>7}")
print("-"*55)
for (d,st), group in sorted(dir_strat.items()):
    s = stats(group)
    print(f"{d:<6} {st:<16} {s['n']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_r']:>7.3f}")

# ============================================================
# 6. BY TRADE DURATION
# ============================================================
print()
print(SEP)
print("6. BY TRADE DURATION")
print(SEP)
durs = defaultdict(list)
for t in parsed:
    durs[t["duration_cat"]].append(t)

print(f"{'Duration':<16} {'N':>4} {'W':>4} {'L':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgPnL':>8} {'AvgR':>7} {'PF':>5}")
print("-"*72)
for cat in ["Short(<10)","Medium(10-30)","Long(30+)"]:
    if cat in durs:
        s = stats(durs[cat])
        print(f"{cat:<16} {s['n']:>4} {s['wins']:>4} {s['losses']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_pnl']:>8.2f} {s['avg_r']:>7.3f} {s['pf']:>5.2f}")

# Also bars distribution
bars_list = [t["bars_held"] for t in parsed]
print(f"\nBars held stats: min={min(bars_list)}, max={max(bars_list)}, median={statistics.median(bars_list):.0f}, mean={statistics.mean(bars_list):.1f}")
wins_bars = [t["bars_held"] for t in parsed if t["win"]]
loss_bars = [t["bars_held"] for t in parsed if not t["win"]]
print(f"Winners: median {statistics.median(wins_bars):.0f} bars | Losers: median {statistics.median(loss_bars):.0f} bars")

# ============================================================
# 7. LOSING STREAKS
# ============================================================
print()
print(SEP)
print("7. LOSING STREAKS (consecutive losses)")
print(SEP)
streaks = []
cur_streak = []
for t in parsed:
    if not t["win"]:
        cur_streak.append(t)
    else:
        if len(cur_streak) >= 3:
            streaks.append(list(cur_streak))
        cur_streak = []
if len(cur_streak) >= 3:
    streaks.append(cur_streak)

streaks.sort(key=lambda x: -len(x))
print(f"Total streaks of 3+: {len(streaks)}")
print()
for i, streak in enumerate(streaks[:10]):
    loss_sum = sum(x["pnl"] for x in streak)
    strat_counts = defaultdict(int)
    for x in streak:
        strat_counts[x["strategy"]] += 1
    hours_in  = [x["hour"] for x in streak]
    days_in   = [x["dow_name"] for x in streak]
    print(f"Streak #{i+1}: {len(streak)} losses | Total P&L: ${loss_sum:.2f}")
    print(f"  Start: {streak[0]['entry_time']} ({streak[0]['dow_name']} {streak[0]['hour']:02d}:xx)")
    print(f"  End  : {streak[-1]['entry_time']} ({streak[-1]['dow_name']} {streak[-1]['hour']:02d}:xx)")
    print(f"  Strategies: {dict(strat_counts)}")
    print(f"  Hours     : {hours_in}")
    print(f"  Days      : {days_in}")
    print()

# ============================================================
# 8. TOP 10 BIGGEST LOSSES
# ============================================================
print()
print(SEP)
print("8. TOP 10 BIGGEST LOSSES")
print(SEP)
biggest_losses = sorted(parsed, key=lambda x: x["pnl"])[:10]
print(f"{'Rk':>2}  {'Date/Time':<20} {'Strategy':<16} {'Dir':<5} {'Hr':>3} {'Day':<4} {'PnL':>9} {'R':>6} {'Bars':>5} {'Exit':>6} {'Session'}")
print("-"*88)
for i, t in enumerate(biggest_losses, 1):
    print(f"{i:>2}  {t['entry_time']:<20} {t['strategy']:<16} {t['dir']:<5} {t['hour']:>3} {t['dow_name']:<4} {t['pnl']:>9.2f} {t['r_multiple']:>6.2f} {t['bars_held']:>5} {t['exit_reason']:>6} {t['session']}")

# ============================================================
# 9. EXIT REASON
# ============================================================
print()
print(SEP)
print("9. EXIT REASON BREAKDOWN")
print(SEP)
exits = defaultdict(list)
for t in parsed:
    exits[t["exit_reason"]].append(t)

print(f"{'ExitReason':<14} {'N':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgPnL':>8}")
print("-"*48)
for reason in sorted(exits):
    s = stats(exits[reason])
    print(f"{reason:<14} {s['n']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_pnl']:>8.2f}")

# Also: SL vs TP by strategy
print()
print("Exit by strategy:")
exit_strat = defaultdict(list)
for t in parsed:
    exit_strat[(t["strategy"], t["exit_reason"])].append(t)
print(f"{'Strategy':<16} {'Exit':<10} {'N':>4} {'TotalPnL':>10} {'AvgPnL':>8}")
print("-"*52)
for (st, ex), group in sorted(exit_strat.items()):
    s = stats(group)
    print(f"{st:<16} {ex:<10} {s['n']:>4} {s['total_pnl']:>10.2f} {s['avg_pnl']:>8.2f}")

# ============================================================
# 10. MONTHLY PERFORMANCE
# ============================================================
print()
print(SEP)
print("10. MONTHLY PERFORMANCE")
print(SEP)
months = defaultdict(list)
for t in parsed:
    months[t["month"]].append(t)

print(f"{'Month':<10} {'N':>4} {'W':>4} {'L':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgR':>7} {'PF':>5}  flag")
print("-"*65)
cumbal = summary["start_balance"]
for m in sorted(months):
    s = stats(months[m])
    cumbal += s['total_pnl']
    flag = " LOSING" if s['total_pnl'] < 0 else " PROFIT"
    print(f"{m:<10} {s['n']:>4} {s['wins']:>4} {s['losses']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_r']:>7.3f} {s['pf']:>5.2f}  {flag}")

# ============================================================
# BONUS A: STRATEGY x HOUR
# ============================================================
print()
print(SEP)
print("BONUS A: STRATEGY x HOUR (hours with >= 2 trades)")
print(SEP)
strat_hour = defaultdict(list)
for t in parsed:
    strat_hour[(t["strategy"], t["hour"])].append(t)

rows = []
for (st, h), group in strat_hour.items():
    s = stats(group)
    rows.append((st, h, s))
rows.sort(key=lambda x: (x[0], x[1]))

print(f"{'Strategy':<16} {'Hr':>4} {'N':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgR':>7}  note")
print("-"*65)
for st, h, s in rows:
    if s['n'] >= 2:
        note = ""
        if s['total_pnl'] < -50:
            note = "  *** DRAIN"
        elif s['total_pnl'] > 100:
            note = "  *** BEST"
        elif s['wr'] == 0:
            note = "  0% win rate"
        print(f"{st:<16} {h:>4}  {s['n']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_r']:>7.3f}{note}")

# ============================================================
# BONUS B: STRATEGY x DAY OF WEEK
# ============================================================
print()
print(SEP)
print("BONUS B: STRATEGY x DAY OF WEEK")
print(SEP)
strat_day = defaultdict(list)
for t in parsed:
    strat_day[(t["strategy"], t["dow"], t["dow_name"])].append(t)

rows2 = []
for (st, di, dn), group in strat_day.items():
    s = stats(group)
    rows2.append((st, di, dn, s))
rows2.sort(key=lambda x: (x[0], x[1]))

print(f"{'Strategy':<16} {'Day':<5} {'N':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgR':>7}  note")
print("-"*65)
for st, di, dn, s in rows2:
    note = ""
    if s['total_pnl'] < -30:
        note = "  *** LOSING"
    elif s['total_pnl'] > 80:
        note = "  *** BEST"
    print(f"{st:<16} {dn:<5} {s['n']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_r']:>7.3f}{note}")

# ============================================================
# BONUS C: STRATEGY x SESSION
# ============================================================
print()
print(SEP)
print("BONUS C: STRATEGY x SESSION")
print(SEP)
strat_sess = defaultdict(list)
for t in parsed:
    strat_sess[(t["strategy"], t["session"])].append(t)

rows3 = []
for (st, sess), group in strat_sess.items():
    s = stats(group)
    rows3.append((st, sess, s))
rows3.sort(key=lambda x: (x[0], x[1]))

print(f"{'Strategy':<16} {'Session':<16} {'N':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgR':>7} {'PF':>5}  note")
print("-"*80)
for st, sess, s in rows3:
    note = ""
    if s['total_pnl'] < -50:
        note = "  *** DRAIN"
    elif s['total_pnl'] > 100:
        note = "  *** GOLD"
    print(f"{st:<16} {sess:<16} {s['n']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_r']:>7.3f} {s['pf']:>5.2f}{note}")

# ============================================================
# BONUS D: PARTIAL CLOSE
# ============================================================
print()
print(SEP)
print("BONUS D: PARTIAL CLOSE ANALYSIS")
print(SEP)
with_p    = [t for t in parsed if t.get("partial_done")]
without_p = [t for t in parsed if not t.get("partial_done")]
sp  = stats(with_p)
sno = stats(without_p)
print(f"With partial   : N={sp['n']:>3},  WR={sp['wr']:>5}%, TotalPnL=${sp['total_pnl']:>9.2f}, AvgPnL=${sp['avg_pnl']:>7.2f}, PF={sp['pf']:.2f}")
print(f"Without partial: N={sno['n']:>3},  WR={sno['wr']:>5}%, TotalPnL=${sno['total_pnl']:>9.2f}, AvgPnL=${sno['avg_pnl']:>7.2f}, PF={sno['pf']:.2f}")

# ============================================================
# BONUS E: TREND ALIGNMENT
# ============================================================
print()
print(SEP)
print("BONUS E: TREND ALIGNMENT")
print(SEP)
trend_align = defaultdict(list)
for t in parsed:
    aligned = (
        (t["dir"]=="BUY"  and t["trend"]=="BULLISH") or
        (t["dir"]=="SELL" and t["trend"]=="BEARISH")
    )
    trend_align["ALIGNED" if aligned else "COUNTER"].append(t)

print(f"{'Alignment':<10} {'N':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgR':>7} {'PF':>5}")
print("-"*45)
for label, group in sorted(trend_align.items()):
    s = stats(group)
    print(f"{label:<10} {s['n']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_r']:>7.3f} {s['pf']:>5.2f}")

# ============================================================
# BONUS F: TP SOURCE
# ============================================================
print()
print(SEP)
print("BONUS F: TP SOURCE (fixed vs liquidity)")
print(SEP)
tp_src = defaultdict(list)
for t in parsed:
    tp_src[t.get("tp_source","unknown")].append(t)

print(f"{'TP Source':<16} {'N':>4} {'WR%':>6} {'TotalPnL':>10} {'AvgPnL':>8} {'AvgR':>7} {'PF':>5}")
print("-"*60)
for src in sorted(tp_src):
    s = stats(tp_src[src])
    print(f"{src:<16} {s['n']:>4} {s['wr']:>6}% {s['total_pnl']:>10.2f} {s['avg_pnl']:>8.2f} {s['avg_r']:>7.3f} {s['pf']:>5.2f}")

# ============================================================
# BONUS G: RISK USD distribution
# ============================================================
print()
print(SEP)
print("BONUS G: RISK USD & SPREAD STATS")
print(SEP)
risks = [t["risk_usd"] for t in parsed]
spreads = [t["spread_px"] for t in parsed]
print(f"Risk USD  : min={min(risks):.2f}, max={max(risks):.2f}, median={statistics.median(risks):.2f}, mean={statistics.mean(risks):.2f}")
print(f"Spread px : min={min(spreads):.1f}, max={max(spreads):.1f}, median={statistics.median(spreads):.1f}, mean={statistics.mean(spreads):.1f}")
print(f"Spread cost as % of avg_risk: {statistics.median(spreads)/statistics.mean(risks)*100:.1f}%")

# ============================================================
# KEY FINDINGS SUMMARY
# ============================================================
print()
print(SEP)
print("KEY FINDINGS SUMMARY")
print(SEP)

# Find worst hour
worst_hour = min(hours.items(), key=lambda kv: stats(kv[1])['total_pnl'])
best_hour  = max(hours.items(), key=lambda kv: stats(kv[1])['total_pnl'])
worst_day  = min(days.items(), key=lambda kv: stats(kv[1])['total_pnl'])
best_day   = max(days.items(), key=lambda kv: stats(kv[1])['total_pnl'])

wh = stats(worst_hour[1])
bh = stats(best_hour[1])
wd = stats(worst_day[1])
bd = stats(best_day[1])

print(f"Best hour  : {best_hour[0]:02d}:xx  N={bh['n']}, WR={bh['wr']}%, P&L=${bh['total_pnl']:.2f}, AvgR={bh['avg_r']:.3f}")
print(f"Worst hour : {worst_hour[0]:02d}:xx  N={wh['n']}, WR={wh['wr']}%, P&L=${wh['total_pnl']:.2f}, AvgR={wh['avg_r']:.3f}")
print(f"Best day   : {best_day[0][1]}  N={bd['n']}, WR={bd['wr']}%, P&L=${bd['total_pnl']:.2f}, AvgR={bd['avg_r']:.3f}")
print(f"Worst day  : {worst_day[0][1]}  N={wd['n']}, WR={wd['wr']}%, P&L=${wd['total_pnl']:.2f}, AvgR={wd['avg_r']:.3f}")

# SILVER_BULLET breakdown
sb = strats.get("SILVER_BULLET",[])
sb_hours = defaultdict(list)
for t in sb:
    sb_hours[t["hour"]].append(t)
print()
print("SILVER_BULLET by hour (all hours with trades):")
for h in sorted(sb_hours):
    s = stats(sb_hours[h])
    flag = " *** DRAIN" if s['total_pnl'] < -20 else (" *** OK" if s['total_pnl'] > 20 else "")
    print(f"  {h:02d}:xx  N={s['n']:>3}, WR={s['wr']:>5}%, P&L=${s['total_pnl']:>9.2f}{flag}")
