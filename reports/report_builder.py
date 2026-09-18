r"""
Generate a single-page HTML backtest report from per-symbol JSON results.

Reads   : back_test/results_*.json
Writes  : reports/backtest_report.html

Run with:
    env\Scripts\python.exe -m reports.report_builder
"""

from __future__ import annotations

import json
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "back_test"
OUTPUT_PATH = ROOT / "reports" / "backtest_report.html"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results() -> list[dict[str, Any]]:
    """Load all back_test/results_*.json files, sorted by symbol name."""
    pattern = str(RESULTS_DIR / "results_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No result files found matching {pattern}")
        sys.exit(1)
    results = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val: float | int | str | None, kind: str = "num") -> str:
    """Format a value for display."""
    if val is None:
        return "&mdash;"
    if kind == "pct":
        return f"{val:+.1f}%" if isinstance(val, (int, float)) else str(val)
    if kind == "usd":
        v = float(val)
        cls = "profit" if v >= 0 else "loss"
        return f'<span class="{cls}">{v:+,.2f}</span>'
    if kind == "r":
        v = float(val)
        cls = "profit" if v >= 0 else "loss"
        return f'<span class="{cls}">{v:+.2f}R</span>'
    if kind == "int":
        return f"{int(val):,}"
    if kind == "price":
        return f"{float(val):,.2f}"
    if kind == "float2":
        return f"{float(val):.2f}"
    return str(val)


def _exit_cls(reason: str) -> str:
    if reason == "TP":
        return "profit"
    if reason == "SL":
        return "loss"
    return ""


def _dir_cls(d: str) -> str:
    return "buy-dir" if d == "BUY" else "sell-dir"


def _monthly_table(trades: list[dict]) -> list[dict]:
    """Group trades by YYYY-MM and compute stats per month."""
    months: dict[str, list[dict]] = {}
    for t in trades:
        key = t["entry_time"][:7]  # YYYY-MM
        months.setdefault(key, []).append(t)

    rows = []
    for month in sorted(months):
        group = months[month]
        wins = sum(1 for t in group if t["pnl"] > 0)
        losses = sum(1 for t in group if t["pnl"] <= 0)
        net = sum(t["pnl"] for t in group)
        last_bal = group[-1].get("balance", 0)
        rows.append({
            "month": month,
            "trades": len(group),
            "wins": wins,
            "losses": losses,
            "net_pnl": net,
            "balance": last_bal,
        })
    return rows


def _build_svg_equity(equity: list[dict], width: int = 760, height: int = 220) -> str:
    """Build an inline SVG polyline equity chart."""
    if not equity:
        return "<p>No equity data.</p>"

    balances = [e["balance"] for e in equity]
    times = list(range(len(balances)))

    min_b = min(balances)
    max_b = max(balances)
    b_range = max_b - min_b if max_b != min_b else 1.0

    pad_x, pad_y = 60, 30
    chart_w = width - pad_x - 20
    chart_h = height - pad_y - 20

    def tx(i: int) -> float:
        return pad_x + (i / max(len(times) - 1, 1)) * chart_w

    def ty(b: float) -> float:
        return pad_y + chart_h - ((b - min_b) / b_range) * chart_h

    pts = " ".join(f"{tx(i):.1f},{ty(b):.1f}" for i, b in enumerate(balances))

    # Y-axis labels (5 ticks)
    y_labels = ""
    for j in range(5):
        val = min_b + b_range * j / 4
        y = ty(val)
        y_labels += (
            f'<line x1="{pad_x - 4}" y1="{y:.1f}" x2="{pad_x}" y2="{y:.1f}" '
            f'stroke="#999" />'
            f'<text x="{pad_x - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#666" font-size="11">{val:,.0f}</text>'
        )

    # Gridlines
    grid = ""
    for j in range(5):
        val = min_b + b_range * j / 4
        y = ty(val)
        grid += (
            f'<line x1="{pad_x}" y1="{y:.1f}" x2="{width - 20}" y2="{y:.1f}" '
            f'stroke="#eee" />'
        )

    # Area fill under the curve
    area_pts = (
        f"{tx(0):.1f},{pad_y + chart_h:.1f} "
        + pts
        + f" {tx(len(balances) - 1):.1f},{pad_y + chart_h:.1f}"
    )

    # X-axis: first and last time labels
    x_labels = ""
    if equity:
        first_t = equity[0].get("time", "")[:10]
        last_t = equity[-1].get("time", "")[:10]
        x_labels = (
            f'<text x="{pad_x}" y="{height - 2}" fill="#666" font-size="11">'
            f'{first_t}</text>'
            f'<text x="{width - 20}" y="{height - 2}" text-anchor="end" '
            f'fill="#666" font-size="11">{last_t}</text>'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" class="equity-svg">
      {grid}
      {y_labels}
      <polygon points="{area_pts}" fill="rgba(46,139,87,0.12)" />
      <polyline points="{pts}" fill="none" stroke="#2e8b57" stroke-width="2" />
      <line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{pad_y + chart_h}"
            stroke="#ccc" />
      <line x1="{pad_x}" y1="{pad_y + chart_h}" x2="{width - 20}"
            y2="{pad_y + chart_h}" stroke="#ccc" />
      {x_labels}
    </svg>"""


# ---------------------------------------------------------------------------
# Portfolio totals
# ---------------------------------------------------------------------------

def _portfolio_totals(results: list[dict]) -> dict:
    total_trades = sum(r["summary"]["trades"] for r in results)
    total_wins = sum(r["summary"]["wins"] for r in results)
    total_losses = sum(r["summary"]["losses"] for r in results)
    total_pnl = sum(r["summary"]["net_pnl"] for r in results)
    total_r = sum(r["summary"].get("total_r", 0) for r in results)
    start_bal = sum(r["summary"]["start_balance"] for r in results)
    end_bal = sum(r["summary"]["end_balance"] for r in results)
    ret_pct = ((end_bal / start_bal - 1) * 100) if start_bal else 0
    max_dd = max((r["summary"].get("max_drawdown_pct", 0) for r in results), default=0)

    # Portfolio profit factor
    gross_profit = sum(
        r["summary"]["avg_win"] * r["summary"]["wins"]
        for r in results if r["summary"]["wins"] > 0
    )
    gross_loss = abs(sum(
        r["summary"]["avg_loss"] * r["summary"]["losses"]
        for r in results if r["summary"]["losses"] > 0
    ))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    exp_r = (total_r / total_trades) if total_trades else 0

    return {
        "start_balance": start_bal,
        "end_balance": end_bal,
        "net_pnl": total_pnl,
        "return_pct": ret_pct,
        "trades": total_trades,
        "wins": total_wins,
        "losses": total_losses,
        "win_rate": (total_wins / total_trades * 100) if total_trades else 0,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "expectancy_r": exp_r,
        "total_r": total_r,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #f7f8fa;
  --card: #ffffff;
  --hdr: #1a1e2e;
  --hdr-text: #e8eaf0;
  --accent: #3b82f6;
  --profit: #16a34a;
  --loss: #dc2626;
  --border: #e2e5ea;
  --text: #1e293b;
  --muted: #64748b;
  --row-alt: #f8fafc;
  --buy: #2563eb;
  --sell: #c026d3;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.5;
}
header {
  background: var(--hdr); color: var(--hdr-text);
  padding: 28px 32px; text-align: center;
}
header h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: .02em; }
header .sub { color: var(--muted); margin-top: 6px; font-size: .9rem; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px 16px; }
.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; margin-bottom: 24px; overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.card-title {
  font-size: 1.1rem; font-weight: 600; padding: 16px 20px;
  border-bottom: 1px solid var(--border); background: #fafbfc;
}

/* Stats grid */
.stats-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
  gap: 1px; background: var(--border);
}
.stat-cell {
  background: var(--card); padding: 14px 16px; text-align: center;
}
.stat-cell .label { font-size: .75rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .04em; margin-bottom: 4px; }
.stat-cell .value { font-size: 1.15rem; font-weight: 600; }

/* Tables */
.tbl-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
th { background: #f1f5f9; position: sticky; top: 0; z-index: 1;
  text-align: left; padding: 10px 12px; font-weight: 600;
  border-bottom: 2px solid var(--border); white-space: nowrap; cursor: pointer;
  user-select: none; }
th:hover { background: #e2e8f0; }
th::after { content: " \\2195"; color: #bbb; font-size: .7em; }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
tr:nth-child(even) td { background: var(--row-alt); }
tr:hover td { background: #eef2ff; }

.profit { color: var(--profit); font-weight: 600; }
.loss   { color: var(--loss);   font-weight: 600; }
.buy-dir  { color: var(--buy);  font-weight: 600; }
.sell-dir { color: var(--sell); font-weight: 600; }

/* Summary portfolio table */
.portfolio-tbl th { text-align: right; }
.portfolio-tbl th:first-child { text-align: left; }
.portfolio-tbl td { text-align: right; }
.portfolio-tbl td:first-child { text-align: left; font-weight: 600; }
.portfolio-tbl tr.total-row td {
  font-weight: 700; border-top: 2px solid var(--hdr);
  background: #f1f5f9;
}

/* Details / collapsible */
details { border-radius: 10px; }
details summary {
  cursor: pointer; font-size: 1.15rem; font-weight: 600;
  padding: 16px 20px; background: var(--card);
  border: 1px solid var(--border); border-radius: 10px;
  margin-bottom: 4px; list-style: none;
  display: flex; align-items: center; gap: 10px;
}
details summary::before { content: "\\25B6"; font-size: .75em; transition: transform .2s; }
details[open] summary::before { transform: rotate(90deg); }
details summary::-webkit-details-marker { display: none; }
details .inner { padding: 4px 0 20px; }

/* Monthly table */
.monthly-tbl td, .monthly-tbl th { text-align: right; }
.monthly-tbl td:first-child, .monthly-tbl th:first-child { text-align: left; }

/* Equity SVG */
.equity-svg { width: 100%; height: auto; display: block;
  font-family: inherit; margin: 16px 0; }

/* Print */
@media print {
  header { background: #333 !important; -webkit-print-color-adjust: exact; }
  details { break-inside: avoid; }
  details[open] summary ~ * { break-inside: avoid; }
  .tbl-wrap { overflow: visible; }
  body { background: #fff; }
}
@media (max-width: 640px) {
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  header { padding: 18px 16px; }
  header h1 { font-size: 1.3rem; }
}
"""

SORT_JS = """
document.addEventListener("DOMContentLoaded",function(){
  document.querySelectorAll("table.sortable").forEach(function(tbl){
    var heads=tbl.querySelectorAll("th");
    heads.forEach(function(th,ci){
      th.addEventListener("click",function(){
        var rows=Array.from(tbl.tBodies[0].rows);
        var asc=th.dataset.asc!=="1";
        th.dataset.asc=asc?"1":"0";
        rows.sort(function(a,b){
          var av=a.cells[ci].getAttribute("data-v")||a.cells[ci].textContent.trim();
          var bv=b.cells[ci].getAttribute("data-v")||b.cells[ci].textContent.trim();
          var an=parseFloat(av),bn=parseFloat(bv);
          if(!isNaN(an)&&!isNaN(bn)) return asc?an-bn:bn-an;
          return asc?av.localeCompare(bv):bv.localeCompare(av);
        });
        rows.forEach(function(r){tbl.tBodies[0].appendChild(r);});
      });
    });
  });
});
"""


def _render_portfolio_table(results: list[dict], totals: dict) -> str:
    rows = ""
    for r in results:
        s = r["summary"]
        rows += f"""<tr>
          <td>{s['symbol']}</td>
          <td>{_fmt(s['start_balance'], 'price')}</td>
          <td>{_fmt(s['end_balance'], 'price')}</td>
          <td>{_fmt(s['net_pnl'], 'usd')}</td>
          <td>{_fmt(s['return_pct'], 'pct')}</td>
          <td>{_fmt(s['trades'], 'int')}</td>
          <td>{_fmt(s['wins'], 'int')}</td>
          <td>{_fmt(s['losses'], 'int')}</td>
          <td>{s['win_rate']:.1f}%</td>
          <td>{s['profit_factor']:.2f}</td>
          <td>{s.get('max_drawdown_pct', 0):.1f}%</td>
          <td>{_fmt(s.get('expectancy_r', 0), 'r')}</td>
        </tr>"""

    t = totals
    rows += f"""<tr class="total-row">
      <td>TOTAL</td>
      <td>{_fmt(t['start_balance'], 'price')}</td>
      <td>{_fmt(t['end_balance'], 'price')}</td>
      <td>{_fmt(t['net_pnl'], 'usd')}</td>
      <td>{_fmt(t['return_pct'], 'pct')}</td>
      <td>{_fmt(t['trades'], 'int')}</td>
      <td>{_fmt(t['wins'], 'int')}</td>
      <td>{_fmt(t['losses'], 'int')}</td>
      <td>{t['win_rate']:.1f}%</td>
      <td>{t['profit_factor']:.2f}</td>
      <td>{t['max_drawdown_pct']:.1f}%</td>
      <td>{_fmt(t['expectancy_r'], 'r')}</td>
    </tr>"""

    return f"""
    <div class="card">
      <div class="card-title">Portfolio Summary</div>
      <div class="tbl-wrap">
        <table class="portfolio-tbl sortable">
          <thead><tr>
            <th>Symbol</th><th>Start Bal</th><th>End Bal</th><th>Net P&amp;L</th>
            <th>Return</th><th>Trades</th><th>Wins</th><th>Losses</th>
            <th>Win Rate</th><th>PF</th><th>Max DD</th><th>Exp R</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>"""


def _render_stats_card(s: dict) -> str:
    cells = [
        ("Start Balance", f"${s['start_balance']:,.2f}"),
        ("End Balance", f"${s['end_balance']:,.2f}"),
        ("Net P&L", _fmt(s["net_pnl"], "usd")),
        ("Return", _fmt(s["return_pct"], "pct")),
        ("Trades", f"{s['trades']}"),
        ("Win Rate", f"{s['win_rate']:.1f}%"),
        ("Profit Factor", f"{s['profit_factor']:.2f}"),
        ("Expectancy", _fmt(s.get("expectancy_r", 0), "r")),
        ("Total R", _fmt(s.get("total_r", 0), "r")),
        ("Max Drawdown", f"${s.get('max_drawdown', 0):,.2f} ({s.get('max_drawdown_pct', 0):.1f}%)"),
        ("Best Trade", _fmt(s.get("best_trade", 0), "usd")),
        ("Worst Trade", _fmt(s.get("worst_trade", 0), "usd")),
        ("Win Streak", f"{s.get('longest_win_streak', 0)}"),
        ("Loss Streak", f"{s.get('longest_loss_streak', 0)}"),
        ("Avg Bars Held", f"{s.get('avg_bars_held', 0):.1f}"),
        ("RRR / Stop", f"{s.get('rrr', '?')} / {s.get('stop_mode', '?')}"),
    ]
    html = ""
    for label, val in cells:
        html += f'<div class="stat-cell"><div class="label">{label}</div><div class="value">{val}</div></div>'
    return f'<div class="stats-grid">{html}</div>'


def _render_trade_table(trades: list[dict]) -> str:
    if not trades:
        return "<p style='padding:16px;color:var(--muted)'>No trades.</p>"

    rows = ""
    for t in trades:
        pnl_cls = "profit" if t["pnl"] >= 0 else "loss"
        r_cls = "profit" if t.get("r_multiple", 0) >= 0 else "loss"
        exit_cls = _exit_cls(t.get("exit_reason", ""))
        dir_cls = _dir_cls(t.get("dir", ""))
        entry_dt = t.get("entry_time", "")[:16]
        rows += f"""<tr>
          <td data-v="{t['n']}">{t['n']}</td>
          <td>{entry_dt}</td>
          <td>{t.get('strategy', '')}</td>
          <td><span class="{dir_cls}">{t.get('dir', '')}</span></td>
          <td data-v="{t['entry']}">{_fmt(t['entry'], 'price')}</td>
          <td data-v="{t['sl']}">{_fmt(t['sl'], 'price')}</td>
          <td data-v="{t['tp']}">{_fmt(t['tp'], 'price')}</td>
          <td data-v="{t['exit_price']}">{_fmt(t['exit_price'], 'price')}</td>
          <td><span class="{exit_cls}">{t.get('exit_reason', '')}</span></td>
          <td>{t.get('vol', '')}</td>
          <td data-v="{t.get('risk_usd', 0)}">{_fmt(t.get('risk_usd', 0), 'price')}</td>
          <td data-v="{t['pnl']}"><span class="{pnl_cls}">{t['pnl']:+,.2f}</span></td>
          <td data-v="{t.get('r_multiple', 0)}"><span class="{r_cls}">{t.get('r_multiple', 0):+.2f}R</span></td>
          <td data-v="{t.get('balance', 0)}">{_fmt(t.get('balance', 0), 'price')}</td>
        </tr>"""

    return f"""
    <div class="tbl-wrap">
      <table class="sortable">
        <thead><tr>
          <th>#</th><th>Date/Time</th><th>Strategy</th><th>Dir</th>
          <th>Entry</th><th>SL</th><th>TP</th><th>Exit</th>
          <th>Reason</th><th>Vol</th><th>Risk $</th><th>P&amp;L $</th>
          <th>R</th><th>Balance</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def _render_monthly_table(trades: list[dict]) -> str:
    months = _monthly_table(trades)
    if not months:
        return ""
    rows = ""
    for m in months:
        pnl_cls = "profit" if m["net_pnl"] >= 0 else "loss"
        rows += f"""<tr>
          <td>{m['month']}</td>
          <td>{m['trades']}</td>
          <td>{m['wins']}</td>
          <td>{m['losses']}</td>
          <td data-v="{m['net_pnl']}"><span class="{pnl_cls}">{m['net_pnl']:+,.2f}</span></td>
          <td data-v="{m['balance']}">{_fmt(m['balance'], 'price')}</td>
        </tr>"""

    return f"""
    <div class="card" style="margin-top:12px">
      <div class="card-title">Monthly P&amp;L</div>
      <div class="tbl-wrap">
        <table class="monthly-tbl sortable">
          <thead><tr>
            <th>Month</th><th>Trades</th><th>Wins</th><th>Losses</th>
            <th>Net P&amp;L</th><th>Balance</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>"""


def _render_symbol_section(result: dict) -> str:
    s = result["summary"]
    trades = result.get("trades", [])
    equity = result.get("equity", [])

    stats_html = _render_stats_card(s)
    trades_html = _render_trade_table(trades)
    monthly_html = _render_monthly_table(trades)
    equity_html = _build_svg_equity(equity)

    return f"""
    <details style="margin-bottom:20px">
      <summary>{s['symbol']} &mdash; {_fmt(s['net_pnl'], 'usd')} &nbsp;|&nbsp;
        {s['trades']} trades &nbsp;|&nbsp; WR {s['win_rate']:.1f}% &nbsp;|&nbsp;
        PF {s['profit_factor']:.2f}</summary>
      <div class="inner">
        <div class="card" style="margin-top:12px">
          <div class="card-title">{s['symbol']} &mdash; Key Metrics</div>
          {stats_html}
        </div>
        <div class="card" style="margin-top:12px">
          <div class="card-title">Equity Curve</div>
          {equity_html}
        </div>
        <div class="card" style="margin-top:12px">
          <div class="card-title">Trade Log ({len(trades)} trades)</div>
          {trades_html}
        </div>
        {monthly_html}
      </div>
    </details>"""


def build_html(results: list[dict]) -> str:
    """Build the full HTML string."""
    totals = _portfolio_totals(results)

    # Period
    periods = []
    for r in results:
        s = r["summary"]
        if s.get("period_from"):
            periods.append(s["period_from"])
        if s.get("period_to"):
            periods.append(s["period_to"])
    period_from = min(periods) if periods else "?"
    period_to = max(periods) if periods else "?"

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols_str = ", ".join(r["summary"]["symbol"] for r in results)

    portfolio_html = _render_portfolio_table(results, totals)
    symbol_sections = "\n".join(_render_symbol_section(r) for r in results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMC Bot Backtest Report</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>SMC Bot Backtest Report</h1>
  <div class="sub">{symbols_str} &nbsp;&bull;&nbsp; {period_from} &rarr; {period_to}
    &nbsp;&bull;&nbsp; Generated {gen_time}</div>
</header>
<div class="container">
  {portfolio_html}
  <h2 style="margin:28px 0 12px;font-size:1.2rem">Per-Symbol Results</h2>
  {symbol_sections}
  <footer style="text-align:center;padding:24px 0;color:var(--muted);font-size:.8rem">
    SMC Bot &mdash; Backtest Report &mdash; {gen_time}
  </footer>
</div>
<script>{SORT_JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    results = load_results()
    print(f"Loaded {len(results)} symbol result(s): "
          + ", ".join(r["summary"]["symbol"] for r in results))

    html = build_html(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
