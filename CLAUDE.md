# tradingBot — MT5 SMC bot

Automated Smart-Money-Concepts bot for MetaTrader 5. Finds a **1H** point of
interest (FVG overlapping an order block), waits for price to return to it,
then takes a **5m** displacement FVG as the trigger. Runs on BTCUSD, XAUUSD,
GBPUSD and EURUSD, each with its own measured configuration.

Last worked on: **2026-09-09**.

---

## Setup

```powershell
# credentials live in .env (git-ignored) — never in config.py
copy .env.example .env      # then fill in MT5_LOGIN / MT5_PASSWORD / MT5_SERVER
                            # optional: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID

env\Scripts\python.exe -m pip install -r requirements.txt
```

- Virtualenv is `env/` (Python 3.14). Always invoke it explicitly:
  `E:\trading\tradingBot\env\Scripts\python.exe`.
- Live account is an **Exness demo/trial** (`Exness-MT5Trial15`).
- ⚠️ The MT5 password was hard-coded in `config.py` before 2026-08-17 and is
  still in use. It should be rotated.

## Running

`core/engine.py` is the only live runner.

```powershell
env\Scripts\python.exe -m core.engine                    # all symbols
env\Scripts\python.exe -m core.engine --symbols BTCUSD   # a subset

env\Scripts\python.exe test_conn.py                      # connection smoke test
env\Scripts\python.exe -m unittest discover -s tests -t . # 30 tests, no MT5 needed
```

**MT5 must be running with the "Algo Trading" button ON (Ctrl+E).** Without it
every order is rejected with retcode 10027 and nothing else looks wrong.
`MT5Connector.trading_enabled()` checks this at startup and before each order.

## Layout

| File | Role |
|---|---|
| `core/engine.py` | **Current live engine.** One MT5 connection, one loop, all symbols. Tries several strategies per scan; first valid signal wins. |
| `core/symbol_context.py` | `SymbolConfig` (loads `config_<symbol>.py`), `SymbolSpec` (broker facts), `SymbolState` (per-symbol POI/risk/rejection state). |
| `risk/risk_manager.py` | Lot sizing, dollar cap, portfolio risk, spread check — `validate_entry()` gates every entry in the engine. |
| `telegram_bot.py` | Telegram dashboard: trade notifications plus `/status`, `/close <SYM>`, `/closeall`. Silently inert without the env vars. |
| `strategy.py` | Pure functions over DataFrames — no I/O. All the trading logic. |
| `mt5_connector.py` | Every broker-specific concern: digits, volume steps, stop levels, filling modes, retcodes. |
| `config.py` | Shared defaults. Credentials read from `.env`. |
| `config_<symbol>.py` | Per-symbol overrides (`from config import *` then override). |
| `tests/test_strategy.py` | 30 unit tests on the pure strategy functions. |
| `reports/engine.py` | Backtest harness. Drives the *same* strategy functions — never reimplements them. |
| `reports/build_full_report.py` | Generates `reports/multi_strategy_report.html` — all strategies on all symbols. The one committed report. |
| `reports/build_report.py` | Generates `backtest_report.html` from the engine's JSON. Needs a fresh engine run first (see below). |
| `core/persistence.py` | SQLite state store — **written but not imported anywhere.** State does not survive a restart today. |

---

## Where configuration comes from

`core/engine.py` reads every per-symbol setting through
`SymbolConfig.load()` (`core/symbol_context.py:91`), which imports
`config_<symbol>.py` and falls back to `config.py`. There is no second path —
`main.py` and `run_both.py`, which read only a handful of keys and traded
XAUUSD/GBPUSD/EURUSD on settings measured and rejected for them, were deleted
on 2026-09-09 (recoverable from git if ever needed).

`SymbolConfig` loads four flags that **no executing code reads**:
`use_structure_shift`, `use_breaker_blocks`, `use_po3` and `use_ifvg`.
`SMCStrategy.detect_ifvg`, `detect_breaker_block`, `detect_po3` and
`detect_structure_shift` exist in `strategy.py` and are exercised only by
`reports/test_ict_filters.py`. Setting `USE_IFVG = True` in a symbol config
therefore does nothing live — the measured PF numbers in those config comments
come from the research script, not from a wired filter.

Config keys read by neither the engine nor the backtester:
`TIME_STOP_BARS`, `TRAILING_STOP_TRIGGER_PCT`, `INVERT_SIGNALS` (loaded into
`SymbolConfig.invert_signals`, never applied).

---

## How a trade happens (`core/engine.py:203 _scan_symbol`)

1. **Manage first.** If the symbol has an open position, manage it and return
   — one position per symbol, always.
2. **Gate.** `SymbolContext.is_trading_allowed()`: weekend, blocked weekday,
   night window, daily-loss limit, consecutive losses, daily trade count,
   manual pause. Blocked reasons are counted in `state.rejections`.
3. **Trend.** 1H close vs EMA(100) → `BULLISH` / `BEARISH` / `NEUTRAL`.
4. **1H POI** (`detect_htf_poi`): newest unmitigated 3-candle FVG whose
   preceding opposite-colour order block overlaps it, in the trend direction.
   OB candles with range ≥ 2×ATR are skipped (LuxAlgo high-volatility filter).
   A fully filled gap is dead; a partially filled one is still valid.
5. **Strategies, first match wins:**
   - **AMD** — Asian range → sweep → MSS → FVG.
   - **Silver Bullet** — strict time window → draw on liquidity → MSS → FVG.
   - **Sweep+FVG** — price inside the POI *and* a same-direction liquidity
     sweep, then the 5m FVG confirmation.
   - P/D+FVG and baseline POI→FVG are **disabled** (`core/engine.py:275`):
     both lost money in the 2026-09-01 backtest. `_try_pd_fvg` and
     `_try_base_fvg` are still there, just not called.
6. **Zone patience** (`_check_zone`): once price enters a POI, the setup has
   `MAX_LTF_WAIT_CANDLES` (15) 5m candles to appear. After that the zone is
   abandoned until price leaves and returns.
7. **Trigger** (`check_ltf_confirmation`): a 5m FVG in the POI direction, with
   a "disrespect" veto — if the middle candle closed through the gap, no trade.
   Stop from `STOP_MODE`; entry is the live market price, not the candle close.
8. **TP**: nearest 15m swing (liquidity) if it is at least
   `MIN_RRR_LIQUIDITY` away, else fixed `RRR`.
9. **Risk** (`validate_entry` → `calculate_lot_size`): % of balance, dollar
   cap, portfolio exposure, spread.
10. **Manage** (`_manage_positions`): at `PARTIAL_TRIGGER_PCT` (80%) of the TP
    distance, close `PARTIAL_CLOSE_PCT` (80%), move the stop to entry, and push
    the runner's TP to the next liquidity level. If the position is at minimum
    lot and cannot be split, just move the stop to break-even.

`INVERT_SIGNALS` is **False** — signals are traded as generated. See
[historical findings](#historical-findings-m15m1-era) for when it was not.

---

## Per-symbol configuration

Each `config_<symbol>.py` starts from `config.py` and overrides. Every entry
below carries a measurement and a date in the file — keep that convention.

| | BTCUSD | XAUUSD | GBPUSD | EURUSD |
|---|---|---|---|---|
| magic | 100200 | 100201 | 100202 | 100203 |
| `STOP_MODE` | `window` | `ob` | `ob` | `ob` |
| `RRR` | 3.0 | 3.5 | 3.0 | 3.0 |
| session (UTC) | no night filter | none | 13:00–21:00 | 13:00–21:00 |
| blocked days | — | Friday | Mon, Tue | — |
| risk | 1% | 1%, $40 cap | 1% | 1% |
| extras | `USE_IFVG` (inert) | sweep filter, BE @2R, no liquidity TP, no partial | P/D filter, `USE_IFVG` (inert), trailing | trailing |

**Judge every change on expectancy in R, never on net dollars** — a config that
risks more per trade will show a bigger dollar total while being worse.

---

## Historical findings (M15/M1 era)

Measured on 34 days of M1 history on **2026-08-17**, when the bot ran M15 zones
with an M1 trigger and `INVERT_SIGNALS = True`. The timeframes changed to
H1/M5 on 2026-08-20 (the user's manual approach, and the bot had not fired in
two days), and inversion was turned off afterwards. These numbers describe the
old configuration — they are kept because the *reasoning* still applies, not
because they still hold.

| Idea | Result then |
|---|---|
| **Structural stop** (`swing`, `zone`) | −0.045R / +0.050R vs `window` +0.370R |
| **One shot per zone** | +0.370R → +0.273R |
| **Removing the trend filter** | +0.370R → +0.235R |
| **Removing the 15m POI layer** | +0.370R → **−0.169R** over 278 trades |

The 1m trigger alone was worth −0.08R over 389 trades: **the HTF zones carried
the entire result.** That is the one finding with real sample size behind it,
and the reason the POI layer should not be "simplified" away.

The headline +0.370R came from **23 trades**. Under the null that each is a
coin with the 28.6% win chance a 1:2.5 target needs, 9+ wins happens 18% of the
time by luck. Treat any configuration here as a hypothesis to test forward.

---

## Backtesting

History is cached to `reports/_history.pkl` on first run, so MT5 is only needed
once. A single run is ~75s over 50,000 M1 bars (34 days — the terminal's M1
limit).

```powershell
env\Scripts\python.exe reports\engine.py                  # baseline
env\Scripts\python.exe reports\engine.py --rrr 2.5 --invert --out final.json
env\Scripts\python.exe reports\engine.py --sweep          # RRR 1.0-4.0, both modes
env\Scripts\python.exe reports\engine.py --ablate         # what each layer contributes
env\Scripts\python.exe reports\engine.py --stops          # stop placement rules
env\Scripts\python.exe reports\engine.py --refresh        # re-pull history from MT5
env\Scripts\python.exe reports\build_report.py            # rebuild the HTML report
```

`reports/` also holds one-off research scripts (`test_ict_filters.py`,
`run_6m_all.py`, `test_xau_*.py`, …). They are where the per-symbol config
numbers come from. **A result from a research script is not a wired feature** —
check that the executing path actually reads the flag before believing it.

Fidelity rules the engine holds to — preserve these in any change:

- Decisions on the close of bar `t`, execution at the open of `t+1`. No lookahead.
- MT5 candles are **bid**. A long fills at ask and exits on bid; a short fills
  at bid and exits on ask. The spread asymmetry is real and carried through.
- One position at a time, exactly like the live engine.
- A bar touching both stop and target is scored as the **loss**.
- Not modelled: commission, swap, slippage beyond spread, intrabar tick order.

Only `reports/multi_strategy_report.html` is committed. Every other generated
output — `backtest_report.html` and all the `*.json` intermediates — was deleted
on 2026-09-09 and is now git-ignored; the scripts that produce them are all
still there.

`build_report.py` reads five JSON files that no longer exist in the tree, so it
needs the engine run first:

```powershell
env\Scripts\python.exe reports\engine.py --out bt_sig_rrr3.json
env\Scripts\python.exe reports\engine.py --rrr 2.5 --invert --out final.json
env\Scripts\python.exe reports\engine.py --sweep      # writes sweep_rrr.json
env\Scripts\python.exe reports\engine.py --stops      # writes stop_rules.json
env\Scripts\python.exe reports\engine.py --ablate     # writes ablation.json
env\Scripts\python.exe reports\build_report.py
```

The old single-symbol report is published at
<https://claude.ai/code/artifact/e2076137-0a1c-4f90-8825-9daabe54f835>
(republish with that URL to keep the link stable).

---

## Broker facts (Exness BTCUSD, measured 2026-08-17)

Worth knowing before diagnosing an order problem:

- `digits=2`, `point=0.01`, `contract_size=1.0` → **P/L in USD = price move × volume**
- `volume_min/max/step` = 0.01 / 200 / 0.01
- `trade_stops_level = 0` — no minimum stop distance enforced on this symbol
- `filling_mode = 3` — both FOK and IOC allowed
- `trade_exemode = 2` (market execution) — the server ignores the `price` field
  for rejection, and it **does** accept SL/TP on the opening deal
- Spread ≈ **700 points ($7.00)**, about 8% of a median trade's risk

A live 0.01-lot test order was placed on 2026-08-17 (ticket **#2174448408**,
BUY @ 64356.46) to verify the execution path end to end — zero slippage, stops
attached on the opening deal. The user closes their own positions; the bot does
not close on shutdown.

---

## Conventions

- `strategy.py` stays pure — DataFrames in, dicts out, no MT5 calls. That's what
  makes it unit-testable and backtestable.
- All broker pedantry belongs in `mt5_connector.py`, not in the loop.
- `order_send` can return `None`. Always check before touching `.retcode`.
- Market orders re-quote from the live tick on every attempt; never send a
  candle close as the price.
- Config changes that affect trading behaviour get a comment recording **what
  was measured** and the number, like the existing ones.
- A new `SymbolConfig` field is only half the work — something in
  `core/engine.py` has to read it, or it joins the dead-flag list above.

## Open items

1. **Wire or delete the dead flags.** `use_ifvg`, `use_structure_shift`,
   `use_breaker_blocks`, `use_po3` are loaded and ignored. `USE_IFVG` in
   particular is set in two symbol configs with a measured comment, which reads
   as if it were live. Wiring it changes live behaviour on BTCUSD and GBPUSD —
   it is a decision, not a cleanup.
2. **`core/persistence.py` is not imported.** Consecutive-loss counts, daily
   P/L and trade counts reset on every restart, so the risk limits are weaker
   than they look.
3. **Take profit at a structural level** instead of a fixed R multiple, and
   skip the trade when the level is closer than ~1.5R. Partly implemented as
   the liquidity TP; the skip rule is not.
4. **Zone quality filter** — require real displacement in the FVG (gap size vs
   ATR), cap zone width, reject stale zones. `detect_htf_poi` has the 2×ATR
   volatility filter but no other quality test.

Not enabled, available if wanted: `MAX_SPREAD_POINTS` (read by
`risk/risk_manager.py:140`). `TIME_STOP_BARS` and `TRAILING_STOP_TRIGGER_PCT`
have config entries and notes but no implementation anywhere — enabling them
does nothing.
