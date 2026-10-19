"""Detailed multi-symbol backtest report.

Runs the backtest on all specified symbols with per-symbol configs,
then generates a single HTML report with:
  - Summary stats per symbol and combined
  - Equity curve chart
  - Monthly P&L table with balance
  - Every trade listed with all details
  - Win/loss streaks, drawdown analysis

Usage:
    python reports/build_detailed_report.py
    python reports/build_detailed_report.py --months 12 --balance 50000
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reports.engine as bt
from reports.engine import load_data, run, _load_symbol_config

HERE = Path(__file__).resolve().parent
SYMBOLS = ["BTCUSD", "GBPUSD", "EURUSD"]


def run_all(symbols, months, start_balance):
    """Run backtest for each symbol, return combined results."""
    all_trades = []
    per_symbol = {}

    for sym in symbols:
        bt._SYMBOL_OVERRIDE = sym
        bt._HTF_CACHE.clear()

        sym_cfg = _load_symbol_config(sym)
        m5, h1, m15, contract, point = load_data(refresh=False)

        # Trim to last N months
        cutoff = m5["time"].iloc[-1] - pd.Timedelta(days=months * 30.44)
        m5_trim = m5[m5["time"] >= cutoff].reset_index(drop=True)
        h1_trim = h1[h1["time"] >= cutoff - pd.Timedelta(days=30)].reset_index(drop=True)
        m15_trim = m15[m15["time"] >= cutoff - pd.Timedelta(days=30)].reset_index(drop=True)

        print(f"{sym}: {len(m5_trim)} M5 bars, {m5_trim['time'].iloc[0].date()} .. {m5_trim['time'].iloc[-1].date()}")

        res = run(m5_trim, h1_trim, m15_trim, contract, point,
                  rrr=sym_cfg["rrr"], stop_mode=sym_cfg["stop_mode"],
                  sym_cfg=sym_cfg, start_balance=start_balance,
                  progress=True)

        for t in res["trades"]:
            t["symbol"] = sym
        all_trades.extend(res["trades"])

        per_symbol[sym] = res

    # Sort all trades by exit time
    all_trades.sort(key=lambda t: t.get("exit_time", ""))
    return all_trades, per_symbol


def build_html(all_trades, per_symbol, start_balance, months):
    """Generate the HTML report."""
    # Combined equity curve
    balance = start_balance
    equity_points = [{"time": "Start", "balance": balance}]
    for t in all_trades:
        balance += t["pnl"]
        equity_points.append({"time": t["exit_time"][:16], "balance": round(balance, 2)})

    final_balance = balance
    net_pnl = final_balance - start_balance
    total_trades = len(all_trades)
    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in losses))
    pf = round(gross_w / gross_l, 2) if gross_l > 0 else float("inf")
    avg_win = gross_w / len(wins) if wins else 0
    avg_loss = gross_l / len(losses) if losses else 0

    # Drawdown
    peak = start_balance
    max_dd = 0
    max_dd_pct = 0
    bal = start_balance
    for t in all_trades:
        bal += t["pnl"]
        peak = max(peak, bal)
        dd = peak - bal
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak * 100

    # Streaks
    streak = 0
    best_w_streak = 0
    best_l_streak = 0
    for t in all_trades:
        if t["pnl"] > 0:
            streak = streak + 1 if streak > 0 else 1
            best_w_streak = max(best_w_streak, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            best_l_streak = max(best_l_streak, abs(streak))

    # Monthly breakdown
    monthly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "balance": 0})
    running_bal = start_balance
    for t in all_trades:
        m = t["exit_time"][:7]
        monthly[m]["trades"] += 1
        monthly[m]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            monthly[m]["wins"] += 1
        running_bal += t["pnl"]
        monthly[m]["balance"] = round(running_bal, 2)
    month_keys = sorted(monthly.keys())

    # Per-symbol stats
    sym_stats = {}
    for sym, res in per_symbol.items():
        s = res["summary"]
        sym_stats[sym] = s

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Report — {months} Months</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 10px; }}
  h2 {{ color: #8b949e; margin: 30px 0 15px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
  .stat-card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; }}
  .stat-card .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 24px; font-weight: bold; margin-top: 4px; }}
  .positive {{ color: #3fb950; }}
  .negative {{ color: #f85149; }}
  .neutral {{ color: #8b949e; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }}
  th {{ background: #161b22; color: #8b949e; text-align: right; padding: 8px 12px; border-bottom: 2px solid #21262d; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #21262d; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  tr:hover {{ background: #161b22; }}
  .trade-win {{ background: rgba(63, 185, 80, 0.05); }}
  .trade-loss {{ background: rgba(248, 81, 73, 0.05); }}
  .chart-container {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px; margin: 20px 0; }}
  .sym-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
  .sym-BTCUSD {{ background: #f7931a20; color: #f7931a; }}
  .sym-GBPUSD {{ background: #2563eb20; color: #60a5fa; }}
  .sym-EURUSD {{ background: #7c3aed20; color: #a78bfa; }}
  .month-row-loss {{ background: rgba(248, 81, 73, 0.08); }}
  .summary-table {{ max-width: 700px; }}
  .summary-table td:last-child {{ font-weight: bold; }}
</style>
</head>
<body>

<h1>Backtest Report</h1>
<p style="color:#8b949e">{months} months | ${start_balance:,.0f} start | RRR 2.0 | Swing SL | {', '.join(per_symbol.keys())}</p>
<p style="color:#8b949e">Period: {all_trades[0]['entry_time'][:10] if all_trades else 'N/A'} — {all_trades[-1]['exit_time'][:10] if all_trades else 'N/A'}</p>

<h2>Overview</h2>
<div class="stats-grid">
  <div class="stat-card">
    <div class="label">Net P&L</div>
    <div class="value {'positive' if net_pnl >= 0 else 'negative'}">${net_pnl:+,.2f}</div>
  </div>
  <div class="stat-card">
    <div class="label">Return</div>
    <div class="value {'positive' if net_pnl >= 0 else 'negative'}">{net_pnl/start_balance*100:+.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">Final Balance</div>
    <div class="value">${final_balance:,.2f}</div>
  </div>
  <div class="stat-card">
    <div class="label">Total Trades</div>
    <div class="value">{total_trades}</div>
  </div>
  <div class="stat-card">
    <div class="label">Win Rate</div>
    <div class="value">{win_rate:.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">Profit Factor</div>
    <div class="value {'positive' if pf > 1 else 'negative'}">{pf}</div>
  </div>
  <div class="stat-card">
    <div class="label">Max Drawdown</div>
    <div class="value negative">${max_dd:,.2f} ({max_dd_pct:.1f}%)</div>
  </div>
  <div class="stat-card">
    <div class="label">Avg Win / Avg Loss</div>
    <div class="value">${avg_win:,.0f} / ${avg_loss:,.0f}</div>
  </div>
  <div class="stat-card">
    <div class="label">Best Win Streak</div>
    <div class="value positive">{best_w_streak}</div>
  </div>
  <div class="stat-card">
    <div class="label">Worst Loss Streak</div>
    <div class="value negative">{best_l_streak}</div>
  </div>
</div>

<h2>Per-Symbol Results</h2>
<table class="summary-table">
<tr><th>Symbol</th><th>Trades</th><th>Win%</th><th>PF</th><th>Net P&L</th><th>Max DD%</th><th>ExpR</th></tr>
"""
    for sym, s in sym_stats.items():
        pnl_class = "positive" if s["net_pnl"] >= 0 else "negative"
        html += f"""<tr>
  <td><span class="sym-badge sym-{sym}">{sym}</span></td>
  <td>{s['trades']}</td><td>{s['win_rate']:.1f}%</td><td>{s.get('profit_factor', 'N/A')}</td>
  <td class="{pnl_class}">${s['net_pnl']:+,.2f}</td><td>{s['max_drawdown_pct']:.1f}%</td>
  <td>{s['expectancy_r']:.3f}R</td>
</tr>"""
    html += "</table>"

    # Equity curve using inline SVG
    if equity_points:
        balances = [e["balance"] for e in equity_points]
        min_b, max_b = min(balances), max(balances)
        b_range = max_b - min_b or 1
        w, h_svg = 900, 300
        points_str = " ".join(
            f"{i/(len(balances)-1)*w:.1f},{h_svg - (b-min_b)/b_range*h_svg*0.9 - h_svg*0.05:.1f}"
            for i, b in enumerate(balances)
        )
        html += f"""
<h2>Equity Curve</h2>
<div class="chart-container">
  <svg width="{w}" height="{h_svg}" viewBox="0 0 {w} {h_svg}">
    <line x1="0" y1="{h_svg - (start_balance-min_b)/b_range*h_svg*0.9 - h_svg*0.05:.1f}" x2="{w}" y2="{h_svg - (start_balance-min_b)/b_range*h_svg*0.9 - h_svg*0.05:.1f}" stroke="#21262d" stroke-dasharray="4"/>
    <polyline points="{points_str}" fill="none" stroke="#58a6ff" stroke-width="2"/>
    <text x="5" y="15" fill="#8b949e" font-size="11">${max_b:,.0f}</text>
    <text x="5" y="{h_svg-5}" fill="#8b949e" font-size="11">${min_b:,.0f}</text>
  </svg>
</div>"""

    # Monthly P&L table
    html += """
<h2>Monthly Performance</h2>
<table>
<tr><th>Month</th><th>Trades</th><th>Wins</th><th>Win%</th><th>P&L</th><th>Balance</th></tr>
"""
    for mk in month_keys:
        m = monthly[mk]
        wr = m["wins"] / m["trades"] * 100 if m["trades"] else 0
        pnl_class = "positive" if m["pnl"] >= 0 else "negative"
        row_class = ' class="month-row-loss"' if m["pnl"] < 0 else ""
        html += f"""<tr{row_class}>
  <td>{mk}</td><td>{m['trades']}</td><td>{m['wins']}</td><td>{wr:.0f}%</td>
  <td class="{pnl_class}">${m['pnl']:+,.2f}</td><td>${m['balance']:,.2f}</td>
</tr>"""
    html += "</table>"

    # All trades table
    html += """
<h2>All Trades</h2>
<table>
<tr><th>#</th><th>Symbol</th><th>Dir</th><th>Entry Time</th><th>Exit Time</th>
<th>Entry</th><th>SL</th><th>TP</th><th>Vol</th><th>Risk$</th>
<th>P&L</th><th>R</th><th>Balance</th><th>Exit</th><th>Bars</th></tr>
"""
    running = start_balance
    for i, t in enumerate(all_trades, 1):
        running += t["pnl"]
        pnl_class = "trade-win" if t["pnl"] > 0 else "trade-loss"
        val_class = "positive" if t["pnl"] > 0 else "negative"
        r_mult = t.get("r_multiple", 0)
        html += f"""<tr class="{pnl_class}">
  <td>{i}</td>
  <td><span class="sym-badge sym-{t['symbol']}">{t['symbol']}</span></td>
  <td>{t['dir']}</td>
  <td>{t['entry_time'][:16]}</td><td>{t['exit_time'][:16]}</td>
  <td>{t['entry']:.5f}</td><td>{t['sl']:.5f}</td><td>{t['tp']:.5f}</td>
  <td>{t.get('vol', 1):.2f}</td><td>${t['risk_usd']:.0f}</td>
  <td class="{val_class}">${t['pnl']:+,.2f}</td>
  <td class="{val_class}">{r_mult:+.2f}R</td>
  <td>${running:,.2f}</td>
  <td>{t.get('exit_reason', '')}</td>
  <td>{t.get('bars_held', 0)}</td>
</tr>"""

    html += f"""</table>

<p style="color:#8b949e; margin-top:40px; font-size:12px">
Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC |
Start balance: ${start_balance:,.0f} |
Period: {months} months |
Strategy: Unified Trading Plan (HTF POI + DOL + Macro Time + 5m FVG + Swing SL)
</p>
</body></html>"""

    return html


def main():
    ap = argparse.ArgumentParser(description="Detailed multi-symbol backtest report")
    ap.add_argument("--months", type=int, default=12, help="lookback period in months")
    ap.add_argument("--balance", type=float, default=50000, help="starting balance")
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    args = ap.parse_args()

    print(f"Running backtest: {args.months} months, ${args.balance:,.0f}, {args.symbols}")
    print("=" * 60)

    all_trades, per_symbol = run_all(args.symbols, args.months, args.balance)

    print(f"\nTotal trades across all symbols: {len(all_trades)}")

    html = build_html(all_trades, per_symbol, args.balance, args.months)

    out = HERE / "detailed_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nReport saved: {out}")
    return str(out)


if __name__ == "__main__":
    path = main()
