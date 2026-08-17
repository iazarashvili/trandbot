"""Builds the backtest artifact from the JSON produced by engine.py."""

import json
from datetime import datetime
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text("utf-8"))


BASE = load("bt_sig_rrr3.json")     # the bot as it was
FINAL = load("final.json")          # the bot as it now ships
SWEEP = load("sweep_rrr.json")
ABL = load("ablation.json")
STOPS = load("stop_rules.json")

SA, SB = BASE["summary"], FINAL["summary"]
TB = FINAL["trades"]
START, VOL = SB["start_balance"], SB["volume"]
BE = 100 / (1 + SB["rrr"])


def p_at_least(k, n, p):
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


P_LUCK = p_at_least(SB["wins"], SB["trades"], 1 / (1 + SB["rrr"]))

# ------------------------------------------------------------- equity chart
PTS = [(datetime.fromisoformat(SB["period_from"]), START)]
for t in TB:
    PTS.append((datetime.fromisoformat(t["exit_time"]), t["balance"]))
Y_LO = min(v for _, v in PTS) - 12
Y_HI = max(v for _, v in PTS) + 12
T0, T1 = PTS[0][0], PTS[-1][0]
SPAN = (T1 - T0).total_seconds()

W, H, ML, MR, MT, MB = 1040, 280, 62, 24, 18, 34
PW, PH = W - ML - MR, H - MT - MB


def ex(d):
    return ML + (d - T0).total_seconds() / SPAN * PW


def ey(v):
    return MT + (Y_HI - v) / (Y_HI - Y_LO) * PH


base_y = ey(START)
line = " ".join(f"{'M' if i == 0 else 'L'}{ex(d):.1f},{ey(v):.1f}"
                for i, (d, v) in enumerate(PTS))
area = line + f" L{ex(T1):.1f},{base_y:.1f} L{ex(T0):.1f},{base_y:.1f} Z"

grid = ""
v = int(Y_LO // 25) * 25
while v <= Y_HI:
    if v >= Y_LO:
        grid += (f'<line class="grid" x1="{ML}" x2="{W-MR}" y1="{ey(v):.1f}" '
                 f'y2="{ey(v):.1f}"/><text class="tick" x="{ML-11}" '
                 f'y="{ey(v)+4:.1f}" text-anchor="end">${v}</text>')
    v += 25

xt = ""
for i, f in enumerate((0.0, 0.34, 0.67, 1.0)):
    d = T0 + (T1 - T0) * f
    anc = "start" if i == 0 else ("end" if i == 3 else "middle")
    xt += (f'<text class="tick" x="{ML+f*PW:.1f}" y="{H-MB+21}" '
           f'text-anchor="{anc}">{d.strftime("%d %b")}</text>')

dots = "".join(f'<circle class="pt" cx="{ex(d):.1f}" cy="{ey(v):.1f}" r="3"/>'
               for d, v in PTS)
hits = "".join(
    f'<rect class="hit" x="{ex(d)-8:.1f}" y="{MT}" width="16" height="{PH}" '
    f'data-label="{"Start" if i == 0 else f"After trade #{i}"}" '
    f'data-date="{d.strftime("%d %b, %H:%M")}" data-bal="{v:,.2f}"/>'
    for i, (d, v) in enumerate(PTS))

EQUITY = f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="Balance over 34 days, from $1000 to ${SB['end_balance']:,.2f}">
<defs><clipPath id="ab"><rect x="0" y="0" width="{W}" height="{base_y:.1f}"/></clipPath>
<clipPath id="be"><rect x="0" y="{base_y:.1f}" width="{W}" height="{H-base_y:.1f}"/></clipPath></defs>
{grid}<path class="fill-up" d="{area}" clip-path="url(#ab)"/>
<path class="fill-dn" d="{area}" clip-path="url(#be)"/>
<line class="baseline" x1="{ML}" x2="{W-MR}" y1="{base_y:.1f}" y2="{base_y:.1f}"/>
<text class="tick" x="{W-MR}" y="{base_y-8:.1f}" text-anchor="end">start ${START:,.0f}</text>
<path class="line" d="{line}"/>{dots}
<circle class="end-up" cx="{ex(T1):.1f}" cy="{ey(PTS[-1][1]):.1f}" r="5"/>
{xt}<line class="cross" x1="0" x2="0" y1="{MT}" y2="{MT+PH}" style="opacity:0"/>{hits}</svg>'''

# ------------------------------------------------------------- R bar chart
BW, BH, BML, BMR, BMT, BMB = 1040, 190, 38, 18, 14, 22
BPW, BPH = BW - BML - BMR, BH - BMT - BMB
R_HI, R_LO = SB["rrr"] + 0.4, -1.5
slot = BPW / len(TB)
bw = slot - 4
bars = ""
for r_ in (int(SB["rrr"]), 0, -1):
    yy = BMT + (R_HI - r_) / (R_HI - R_LO) * BPH
    bars += (f'<line class="{"baseline" if r_ == 0 else "grid"}" x1="{BML}" '
             f'x2="{BW-BMR}" y1="{yy:.1f}" y2="{yy:.1f}"/><text class="tick" '
             f'x="{BML-9}" y="{yy+4:.1f}" text-anchor="end">{r_:+d}R</text>')
for i, t in enumerate(TB):
    r = t["r_multiple"]
    x = BML + i * slot + 2
    y = BMT + (R_HI - max(r, 0)) / (R_HI - R_LO) * BPH
    h = abs(r) / (R_HI - R_LO) * BPH
    rad = min(4, bw / 2, h)
    if r > 0:
        d = (f"M{x:.1f},{y+h:.1f} L{x:.1f},{y+rad:.1f} Q{x:.1f},{y:.1f} {x+rad:.1f},{y:.1f} "
             f"L{x+bw-rad:.1f},{y:.1f} Q{x+bw:.1f},{y:.1f} {x+bw:.1f},{y+rad:.1f} "
             f"L{x+bw:.1f},{y+h:.1f} Z")
    else:
        d = (f"M{x:.1f},{y:.1f} L{x:.1f},{y+h-rad:.1f} Q{x:.1f},{y+h:.1f} {x+rad:.1f},{y+h:.1f} "
             f"L{x+bw-rad:.1f},{y+h:.1f} Q{x+bw:.1f},{y+h:.1f} {x+bw:.1f},{y+h-rad:.1f} "
             f"L{x+bw:.1f},{y:.1f} Z")
    bars += (f'<path class="bar {"up" if r > 0 else "dn"}" d="{d}" '
             f'data-tip="Trade #{t["n"]} · {t["dir"]}|{t["entry_time"][:16]}|'
             f'{t["exit_reason"]}  {r:+.2f}R   ${t["pnl"]:+.2f}"/>')
RBARS = (f'<svg viewBox="0 0 {BW} {BH}" role="img" aria-label="'
         f'{len(TB)} trade results in R">{bars}</svg>')

# ------------------------------------------------------- RRR sweep charts
def sweep_chart(mode):
    rows = [r for r in SWEEP if r["mode"] == mode]
    w, h, ml, mr, mt, mb = 1040, 150, 44, 16, 12, 26
    pw, ph = w - ml - mr, h - mt - mb
    lo, hi = -0.52, 0.42
    zero = mt + (hi / (hi - lo)) * ph
    s = pw / len(rows)
    bwid = min(64, s - 16)
    out = ""
    for gv in (0.25, 0.0, -0.25, -0.5):
        yy = mt + (hi - gv) / (hi - lo) * ph
        out += (f'<line class="{"baseline" if gv == 0 else "grid"}" x1="{ml}" '
                f'x2="{w-mr}" y1="{yy:.1f}" y2="{yy:.1f}"/><text class="tick" '
                f'x="{ml-9}" y="{yy+4:.1f}" text-anchor="end">{gv:+.2f}</text>')
    for i, r in enumerate(rows):
        e = r["expectancy_r"]
        cx = ml + i * s + s / 2
        x = cx - bwid / 2
        y = mt + (hi - max(e, 0)) / (hi - lo) * ph
        hh = abs(e) / (hi - lo) * ph
        rad = min(4, hh)
        if e > 0:
            d = (f"M{x:.1f},{y+hh:.1f} L{x:.1f},{y+rad:.1f} Q{x:.1f},{y:.1f} "
                 f"{x+rad:.1f},{y:.1f} L{x+bwid-rad:.1f},{y:.1f} Q{x+bwid:.1f},{y:.1f} "
                 f"{x+bwid:.1f},{y+rad:.1f} L{x+bwid:.1f},{y+hh:.1f} Z")
            lab_y = y - 6
        else:
            d = (f"M{x:.1f},{y:.1f} L{x:.1f},{y+hh-rad:.1f} Q{x:.1f},{y+hh:.1f} "
                 f"{x+rad:.1f},{y+hh:.1f} L{x+bwid-rad:.1f},{y+hh:.1f} "
                 f"Q{x+bwid:.1f},{y+hh:.1f} {x+bwid:.1f},{y+hh-rad:.1f} "
                 f"L{x+bwid:.1f},{y:.1f} Z")
            lab_y = y + hh + 14
        out += (f'<path class="bar {"up" if e > 0 else "dn"}" d="{d}" '
                f'data-tip="RRR 1:{r["rrr"]:g}|{r["trades"]} trades · '
                f'{r["win_rate"]}% won|{e:+.3f}R per trade   ${r["net_pnl"]:+.2f}"/>'
                f'<text class="vlab" x="{cx:.1f}" y="{lab_y:.1f}" '
                f'text-anchor="middle">{e:+.3f}</text>'
                f'<text class="tick" x="{cx:.1f}" y="{h-6}" '
                f'text-anchor="middle">1:{r["rrr"]:g}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Expectancy per '
            f'trade by risk-reward ratio, {mode.lower()}">{out}</svg>')


# --------------------------------------------------------------- tables
def ablation_rows():
    order, seen = [], set()
    for r in ABL:
        if r["layer"] not in seen:
            seen.add(r["layer"])
            order.append(r["layer"])
    out = ""
    for layer in order:
        sig = next(r for r in ABL if r["layer"] == layer and r["mode"] == "AS-SIGNALLED")
        inv = next(r for r in ABL if r["layer"] == layer and r["mode"] == "INVERTED")
        keep = layer == "full strategy"
        out += (f'<tr{" class=\'hi\'" if keep else ""}><th scope="row">{layer}</th>'
                f'<td class="num muted">{inv["trades"]}</td>'
                f'<td class="num sign {"up" if sig["expectancy_r"] > 0 else "dn"}">'
                f'{sig["expectancy_r"]:+.3f}</td>'
                f'<td class="num sign {"up" if inv["expectancy_r"] > 0 else "dn"}">'
                f'{inv["expectancy_r"]:+.3f}</td></tr>')
    return out


def stop_rows():
    out = ""
    for mode in ("window", "swing", "zone"):
        rows = [r for r in STOPS if r["stop_mode"] == mode]
        best = max(rows, key=lambda r: r["expectancy_r"])
        out += (f'<tr{" class=\'hi\'" if mode == "window" else ""}>'
                f'<th scope="row">{mode}</th>'
                f'<td class="num muted">${best["avg_risk_usd"]:.2f}</td>'
                + "".join(
                    f'<td class="num sign {"up" if r["expectancy_r"] > 0 else "dn"}">'
                    f'{r["expectancy_r"]:+.3f}</td>' for r in rows)
                + "</tr>")
    return out


VERDICTS = [
    ("Target multiple", "Fixed 1:3 may be throwing away winners", "kept",
     "Swept 1.0–4.0. Inverted expectancy is positive at every value — that is "
     "the finding. The peak at 2.5 is inside one standard error of every other "
     "value, so 2.5 is a pick, not an optimum."),
    ("15m zone layer", "Zones are not filtered for quality", "kept",
     "Removing them entirely drops inverted expectancy from +0.370 to −0.169 "
     "over 278 trades. The zones are carrying the whole result."),
    ("Trend filter", "100 EMA is a weak bias filter", "kept",
     "Removing it costs +0.370 → +0.235 expectancy. It earns its place."),
    ("Structural stop", "Stop should sit where the setup dies, not N bars back", "killed",
     "Measured worse at every ratio: swing −0.045, zone +0.050, against +0.370 "
     "for the original window rule. Fading a signal means a tighter stop on the "
     "fade side simply gets hit more."),
    ("One shot per zone", "A zone that already failed should not re-fire", "killed",
     "+0.370 → +0.273. No evidence it helps; on 23 trades the difference is one "
     "trade anyway."),
]
verdict_cards = "".join(
    f'<div class="v-card {v[2]}"><div class="v-head"><h3>{v[0]}</h3>'
    f'<span class="pill {v[2]}">{"kept" if v[2] == "kept" else "rejected"}</span></div>'
    f'<p class="v-claim">{v[1]}</p><p>{v[3]}</p></div>' for v in VERDICTS)

ledger = ""
for t in TB:
    win = t["pnl"] > 0
    ledger += (f'<tr class="{"w" if win else "l"}"><td class="n">{t["n"]}</td>'
               f'<td><span class="dir {t["dir"].lower()}">{t["dir"]}</span></td>'
               f'<td class="d">{t["entry_time"][:16]}</td>'
               f'<td class="d">{t["exit_time"][:16]}</td>'
               f'<td class="num">{t["entry"]:,.2f}</td>'
               f'<td class="num">{t["sl"]:,.2f}</td>'
               f'<td class="num">{t["tp"]:,.2f}</td>'
               f'<td class="num muted">{t["risk_usd"]:.2f}</td>'
               f'<td><span class="tag {"tp" if win else "sl"}">{t["exit_reason"]}</span></td>'
               f'<td class="num sign {"up" if win else "dn"}">{t["pnl"]:+.2f}</td>'
               f'<td class="num sign {"up" if win else "dn"}">{t["r_multiple"]:+.2f}</td>'
               f'<td class="num muted">{t["bars_held"]}m</td>'
               f'<td class="num bal">{t["balance"]:,.2f}</td></tr>')

html = f'''<title>SMC Backtest Ledger</title>
<style>
:root {{
  color-scheme: light;
  --plane:#eaeff4; --surface:#fbfcfd; --ink:#0e161e; --ink-2:#4c5a67; --muted:#7c8a97;
  --grid:#dde4ea; --rule:#d2dbe3; --ring:rgba(14,22,30,.10);
  --up:#2a78d6; --dn:#d03b3b;
  --up-soft:rgba(42,120,214,.13); --dn-soft:rgba(208,59,59,.13);
  --row-w:rgba(42,120,214,.05); --row-l:rgba(208,59,59,.045);
  --hi:rgba(42,120,214,.08);
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --plane:#0b1015; --surface:#151c23; --ink:#eef4f9; --ink-2:#a9b6c2; --muted:#7c8a97;
    --grid:#222d37; --rule:#2a3641; --ring:rgba(255,255,255,.10);
    --up:#3987e5; --dn:#e05a5a;
    --up-soft:rgba(57,135,229,.16); --dn-soft:rgba(224,90,90,.16);
    --row-w:rgba(57,135,229,.07); --row-l:rgba(224,90,90,.06);
    --hi:rgba(57,135,229,.10);
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --plane:#0b1015; --surface:#151c23; --ink:#eef4f9; --ink-2:#a9b6c2; --muted:#7c8a97;
  --grid:#222d37; --rule:#2a3641; --ring:rgba(255,255,255,.10);
  --up:#3987e5; --dn:#e05a5a;
  --up-soft:rgba(57,135,229,.16); --dn-soft:rgba(224,90,90,.16);
  --row-w:rgba(57,135,229,.07); --row-l:rgba(224,90,90,.06);
  --hi:rgba(57,135,229,.10);
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1120px; margin:0 auto; padding:46px 22px 76px;
  display:flex; flex-direction:column; gap:34px; }}
.eyebrow {{ font-family:var(--mono); font-size:11px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); }}
header {{ display:flex; flex-direction:column; gap:16px; }}
h1 {{ font-size:clamp(27px,4vw,39px); line-height:1.1; margin:0;
  letter-spacing:-.022em; font-weight:680; text-wrap:balance; }}
.lede {{ margin:0; max-width:66ch; color:var(--ink-2); font-size:16px; }}
.duo {{ display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
  border-radius:3px; overflow:hidden;
  grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); }}
.pane {{ background:var(--surface); padding:22px 24px;
  display:flex; flex-direction:column; gap:3px; }}
.k-lab {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--muted); }}
.fig {{ font-family:var(--mono); font-size:clamp(28px,4.4vw,42px); line-height:1.05;
  letter-spacing:-.03em; font-weight:600; font-variant-numeric:tabular-nums; }}
.fig.up {{ color:var(--up); }} .fig.dn {{ color:var(--dn); }}
.pane .note {{ font-size:13px; color:var(--ink-2); }}
section {{ display:flex; flex-direction:column; gap:13px; }}
h2 {{ font-size:19px; margin:0; font-weight:640; letter-spacing:-.012em; }}
h3.run {{ font-family:var(--mono); font-size:11px; letter-spacing:.13em; margin:0;
  text-transform:uppercase; color:var(--muted); font-weight:500; }}
.sub {{ margin:0; color:var(--ink-2); font-size:14px; max-width:72ch; }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:3px;
  padding:16px 18px 10px; position:relative; }}
.scroll {{ overflow-x:auto; }}
.scroll svg {{ display:block; width:100%; min-width:640px; height:auto; }}
.stack {{ display:flex; flex-direction:column; gap:9px; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.baseline {{ stroke:var(--rule); stroke-width:1.5; stroke-dasharray:4 4; }}
.tick {{ fill:var(--muted); font-family:var(--mono); font-size:11px; }}
.vlab {{ fill:var(--ink-2); font-family:var(--mono); font-size:11px;
  font-variant-numeric:tabular-nums; }}
.line {{ fill:none; stroke:var(--ink); stroke-width:2; stroke-linejoin:round; }}
.fill-up {{ fill:var(--up-soft); }} .fill-dn {{ fill:var(--dn-soft); }}
.pt {{ fill:var(--surface); stroke:var(--ink); stroke-width:1.4; }}
.end-up {{ fill:var(--up); stroke:var(--surface); stroke-width:2; }}
.cross {{ stroke:var(--muted); stroke-width:1; stroke-dasharray:3 3;
  pointer-events:none; transition:opacity .12s; }}
.hit {{ fill:transparent; cursor:crosshair; }}
.bar {{ stroke:var(--surface); stroke-width:2; cursor:pointer; }}
.bar.up {{ fill:var(--up); }} .bar.dn {{ fill:var(--dn); }}
.bar:hover {{ opacity:.72; }}
.tip {{ position:absolute; pointer-events:none; opacity:0; transform:translate(-50%,-100%);
  background:var(--ink); color:var(--plane); font-family:var(--mono); font-size:11.5px;
  line-height:1.5; padding:7px 10px; border-radius:3px; white-space:nowrap;
  transition:opacity .1s; z-index:5; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
thead th {{ font-family:var(--mono); font-size:10px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--muted); font-weight:500; text-align:right;
  padding:0 10px 9px; border-bottom:1px solid var(--rule); white-space:nowrap; }}
thead th:first-child {{ text-align:left; }}
table.led thead th:nth-child(-n+4) {{ text-align:left; }}
tbody td {{ padding:7px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
tbody tr.w {{ background:var(--row-w); }} tbody tr.l {{ background:var(--row-l); }}
tbody tr.hi {{ background:var(--hi); }}
td.num, td.n {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
td.n {{ color:var(--muted); font-size:12px; }}
td.d {{ font-family:var(--mono); font-size:11.5px; color:var(--ink-2); }}
td.muted {{ color:var(--muted); }} td.bal {{ font-weight:600; }}
td.sign.up {{ color:var(--up); }} td.sign.dn {{ color:var(--dn); }}
th[scope=row] {{ text-align:left; font-family:var(--sans); font-size:13.5px;
  text-transform:none; letter-spacing:0; color:var(--ink); font-weight:500;
  padding:7px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
.dir {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.07em;
  padding:2px 6px; border-radius:2px; border:1px solid var(--ring); }}
.dir.buy {{ color:var(--up); }} .dir.sell {{ color:var(--dn); }}
.tag {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.07em;
  padding:2px 6px; border-radius:2px; font-weight:600; }}
.tag.tp {{ color:var(--up); background:var(--up-soft); }}
.tag.sl {{ color:var(--dn); background:var(--dn-soft); }}
.verdicts {{ display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
  border-radius:3px; overflow:hidden;
  grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); }}
.v-card {{ background:var(--surface); padding:16px 18px;
  display:flex; flex-direction:column; gap:6px; }}
.v-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
.v-card h3 {{ margin:0; font-size:14px; font-weight:640; }}
.v-card p {{ margin:0; font-size:13px; color:var(--ink-2); }}
.v-claim {{ font-style:italic; }}
.pill {{ font-family:var(--mono); font-size:9.5px; letter-spacing:.1em;
  text-transform:uppercase; padding:2px 7px; border-radius:2px; font-weight:600;
  white-space:nowrap; }}
.pill.kept {{ color:var(--up); background:var(--up-soft); }}
.pill.killed {{ color:var(--dn); background:var(--dn-soft); }}
.flag {{ background:var(--surface); border:1px solid var(--ring);
  border-left:3px solid var(--dn); border-radius:3px; padding:18px 22px;
  display:flex; flex-direction:column; gap:8px; }}
.flag h2 {{ font-size:17px; }}
.flag p {{ margin:0; font-size:14.5px; color:var(--ink-2); max-width:74ch; }}
.mono {{ font-family:var(--mono); }}
footer {{ border-top:1px solid var(--rule); padding-top:18px; font-size:12.5px;
  color:var(--muted); display:flex; flex-direction:column; gap:6px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="wrap">

<header>
  <div class="eyebrow">Backtest · {SB['symbol']} · {SB['period_from'][:10]} → {SB['period_to'][:10]} · {SB['days']} days</div>
  <h1>Five ideas for making the bot smarter. The data killed three of them.</h1>
  <p class="lede">Every M1 bar of real Exness history, replayed through the same
  <span class="mono" style="font-size:.92em">SMCStrategy</span> code the live bot runs.
  ${START:,.0f} to start, {VOL} lots. The bot went from losing {abs(SA['return_pct'])}% to
  making {SB['return_pct']}% — but almost none of that came from the changes that
  sounded most sensible.</p>
</header>

<div class="duo">
  <div class="pane">
    <div class="k-lab">Before</div>
    <div class="fig dn">−${abs(SA['net_pnl']):,.2f}</div>
    <div class="note">1:3, signals as generated · {SA['trades']} trades · {SA['win_rate']}% won</div>
  </div>
  <div class="pane">
    <div class="k-lab">Now</div>
    <div class="fig up">+${SB['net_pnl']:,.2f}</div>
    <div class="note">1:{SB['rrr']:g}, signals inverted · {SB['trades']} trades · {SB['win_rate']}% won</div>
  </div>
  <div class="pane">
    <div class="k-lab">Per trade</div>
    <div class="fig up">+{SB['expectancy_r']:.2f}R</div>
    <div class="note">profit factor {SB['profit_factor']} · max drawdown {SB['max_drawdown_pct']}%</div>
  </div>
</div>

<section>
  <h2>What survived contact with the data</h2>
  <p class="sub">Each change was measured on its own before being kept. Expectancy
  per trade in R is the yardstick — dollar totals mislead when the stop rule
  changes how much each trade risks.</p>
  <div class="verdicts">{verdict_cards}</div>
</section>

<section>
  <h2>The signal is wrong at every target</h2>
  <p class="sub">Expectancy per trade, sweeping the risk-reward ratio. Taking the
  signals as generated loses at all six settings; taking the opposite side wins at
  all six. That consistency is worth more than any single result — but note these
  are six views of one dataset, not six independent tests.</p>
  <div class="stack">
    <h3 class="run">Signals as generated</h3>
    <div class="card"><div class="scroll">{sweep_chart("AS-SIGNALLED")}</div><div class="tip"></div></div>
    <h3 class="run">Signals inverted</h3>
    <div class="card"><div class="scroll">{sweep_chart("INVERTED")}</div><div class="tip"></div></div>
  </div>
</section>

<section>
  <h2>Where the edge actually lives</h2>
  <p class="sub">Strip out one layer at a time and re-run. The 15m point-of-interest
  zones are not decoration — without them the inverted bot takes 278 trades and loses
  $315. The 1m trigger on its own is worth nothing.</p>
  <div class="card scroll">
    <table>
      <thead><tr><th scope="col">Layer removed</th><th scope="col">Trades</th>
      <th scope="col">expR as signalled</th><th scope="col">expR inverted</th></tr></thead>
      <tbody>{ablation_rows()}</tbody>
    </table>
  </div>
</section>

<section>
  <h2>The structural stop was a good idea that lost</h2>
  <p class="sub">Placing the stop where the setup is invalidated — behind the swing
  the shift broke, or behind the zone — is the textbook answer and it is measurably
  worse here. Fading a signal puts the stop on the side price is travelling toward,
  so tightening it just feeds the loss column. Expectancy in R, inverted mode.</p>
  <div class="card scroll">
    <table>
      <thead><tr><th scope="col">Stop rule</th><th scope="col">Avg risk</th>
      <th scope="col">1:1.5</th><th scope="col">1:2</th><th scope="col">1:2.5</th>
      <th scope="col">1:3</th></tr></thead>
      <tbody>{stop_rows()}</tbody>
    </table>
  </div>
</section>

<div class="flag">
  <h2>The honest size of this result</h2>
  <p>{SB['trades']} trades. If every trade were a coin with the {BE:.0f}% win chance that a
  1:{SB['rrr']:g} target needs to break even, you would see {SB['wins']} wins or better
  <b>{P_LUCK*100:.0f}% of the time</b> by luck alone. The one statistically solid number in
  this whole report is the 389-trade ablation run — and it says the 1m trigger by itself
  is worth −0.08R.</p>
  <p>The choice of 1:{SB['rrr']:g} in particular is not supported: the standard error on
  expectancy here is about ±0.35R, so every ratio in the sweep sits inside one error bar
  of every other. Treat the configuration as a hypothesis to test forward, not a result.</p>
</div>

<section>
  <h2>Equity path</h2>
  <div class="card"><div class="scroll">{EQUITY}</div><div class="tip"></div></div>
</section>

<section>
  <h2>Every trade, in R</h2>
  <div class="card"><div class="scroll">{RBARS}</div><div class="tip"></div></div>
</section>

<section>
  <h2>The ledger</h2>
  <p class="sub">All {SB['trades']} trades at the shipped configuration. Entry is the fill
  price including spread; risk is the dollar distance to the stop at {VOL} lots.</p>
  <div class="card scroll"><table class="led">
    <thead><tr><th>#</th><th>Side</th><th>Entered</th><th>Exited</th><th>Entry</th>
    <th>Stop</th><th>Target</th><th>Risk&nbsp;$</th><th>Exit</th><th>P&amp;L&nbsp;$</th>
    <th>R</th><th>Held</th><th>Balance</th></tr></thead>
    <tbody>{ledger}</tbody>
  </table></div>
</section>

<footer>
  <div><b>Shipped configuration.</b>
  <span class="mono">INVERT_SIGNALS = True</span> ·
  <span class="mono">RRR = {SB['rrr']:g}</span> ·
  <span class="mono">STOP_MODE = "window"</span> ·
  <span class="mono">USE_TREND_FILTER = True</span>. Each is one line in
  <span class="mono">config.py</span>; reverting any of them changes nothing else.</div>
  <div><b>Method.</b> Decisions on each closed M1 bar, executed at the next bar's open —
  no lookahead. MT5 candles are bid: a long fills at ask and exits on bid, a short fills at
  bid and exits on ask. A bar touching both stop and target is scored as the loss.
  Not modelled: commission, swap, slippage beyond the spread, intrabar tick order.
  Spread is the terminal's recorded M1 value, a flat {SB['spread_points']} points.</div>
  <div>{SB['m1_bars']:,} M1 bars · {SB['days']} days · every run reproducible with
  <span class="mono">python reports/engine.py</span>.</div>
</footer>

</div>

<script>
(function () {{
  document.querySelectorAll(".card").forEach(function (card) {{
    var tip = card.querySelector(".tip");
    if (!tip) return;
    var cross = card.querySelector(".cross");
    function hide() {{ tip.style.opacity = 0; if (cross) cross.style.opacity = 0; }}
    card.querySelectorAll(".hit, .bar").forEach(function (el) {{
      el.addEventListener("mouseenter", function () {{
        var cr = card.getBoundingClientRect(), r = el.getBoundingClientRect();
        if (el.classList.contains("hit")) {{
          tip.innerHTML = "<b>" + el.dataset.label + "</b><br>" + el.dataset.date +
                          "<br>balance $" + el.dataset.bal;
          if (cross) {{
            var x = parseFloat(el.getAttribute("x")) + parseFloat(el.getAttribute("width")) / 2;
            cross.setAttribute("x1", x); cross.setAttribute("x2", x);
            cross.style.opacity = .8;
          }}
        }} else {{
          var p = el.dataset.tip.split("|");
          tip.innerHTML = "<b>" + p[0] + "</b><br>" + p[1] + "<br>" + p[2];
        }}
        tip.style.left = (r.left - cr.left + r.width / 2) + "px";
        tip.style.top = (r.top - cr.top - 8 +
          (el.classList.contains("hit") ? r.height / 2 : 0)) + "px";
        tip.style.opacity = 1;
      }});
      el.addEventListener("mouseleave", hide);
    }});
  }});
}})();
</script>
'''

out = HERE / "backtest_report.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out.name} ({len(html):,} bytes)")
print(f"before {SA['net_pnl']:+.2f} -> now {SB['net_pnl']:+.2f} "
      f"({SB['expectancy_r']:+.3f}R/trade, p_luck={P_LUCK:.2f})")
