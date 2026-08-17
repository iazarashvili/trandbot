# tradingBot — MT5 SMC bot

Automated Smart-Money-Concepts bot for MetaTrader 5. Finds a 15m point of
interest (FVG overlapping an order block), waits for price to return to it,
then takes a 1m market-structure-shift + FVG as the trigger.

**It currently trades its own signals backwards.** That is deliberate and
measured — see [Configuration decisions](#configuration-decisions).

Last worked on: **2026-08-17**.

---

## Setup

```powershell
# credentials live in .env (git-ignored) — never in config.py
copy .env.example .env      # then fill in MT5_LOGIN / MT5_PASSWORD / MT5_SERVER

env\Scripts\python.exe -m pip install -r requirements.txt
```

- Virtualenv is `env/` (Python 3.14). Always invoke it explicitly:
  `E:\trading\tradingBot\env\Scripts\python.exe`.
- Live account is an **Exness demo/trial** (`Exness-MT5Trial15`), balance ~$128.
- ⚠️ The MT5 password was hard-coded in `config.py` before 2026-08-17 and is
  still in use. It should be rotated.

## Running

```powershell
env\Scripts\python.exe main.py                    # the bot
env\Scripts\python.exe test_conn.py               # connection smoke test
env\Scripts\python.exe -m unittest discover -s tests -t .   # 30 tests, no MT5 needed
```

**MT5 must be running with the "Algo Trading" button ON (Ctrl+E).** Without it
every order is rejected with retcode 10027 and nothing else looks wrong.
`MT5Connector.trading_enabled()` checks this at startup and before each order.

## Layout

| File | Role |
|---|---|
| `main.py` | Poll loop, position management, order placement. 4-space indent. |
| `strategy.py` | Pure functions over DataFrames — no I/O. All the trading logic. |
| `mt5_connector.py` | Every broker-specific concern: digits, volume steps, stop levels, filling modes, retcodes. |
| `config.py` | All tunables. Credentials read from `.env`. |
| `tests/test_strategy.py` | 30 unit tests on the pure strategy functions. |
| `reports/engine.py` | Backtest harness. Drives the *same* strategy functions — never reimplements them. |
| `reports/build_report.py` | Generates `backtest_report.html` from the engine's JSON. |

---

## Configuration decisions

These were measured, not guessed. **Don't re-propose the rejected ones** —
they've been tested on 34 days of real M1 history.

| Setting | Value | Why |
|---|---|---|
| `INVERT_SIGNALS` | `True` | Signals as generated lose at **every** RRR from 1:1 to 1:4; inverted wins at every one. Reverting is this one flag. |
| `RRR` | `2.5` | Peak of the sweep — but the standard error on expectancy is ±0.35R, so this is a pick, not an optimum. |
| `STOP_MODE` | `"window"` | The structural alternatives measured **worse**. See below. |
| `USE_TREND_FILTER` | `True` | Removing it costs +0.370R → +0.235R per trade. |
| `USE_CLOSED_CANDLES_ONLY` | `True` | The forming candle repaints; signals appear and vanish within a minute. |

### Measured and rejected

| Idea | Result |
|---|---|
| **Structural stop** (behind the swing the MSS broke, or behind the POI) | `swing` −0.045R, `zone` +0.050R vs `window` **+0.370R**. Implemented as `STOP_MODE` options; both lose. The bot fades its own signal, so the stop sits where price is heading — tightening it onto structure just feeds the loss column. |
| **One shot per zone** (a zone that produced a loss doesn't re-fire) | +0.370R → +0.273R. No evidence it helps. |

`zone` mode shows a *higher dollar total* purely because it risks 2.3× as much
per trade. **Judge every change on expectancy in R, never on net dollars.**

### Where the edge lives

Ablation (`python reports/engine.py --ablate`):

| Layer removed | Trades | expR inverted |
|---|---|---|
| nothing (full strategy) | 23 | **+0.370** |
| trend filter | 34 | +0.235 |
| **15m POI zones** | 278 | **−0.169** |

The 1m trigger on its own is worth −0.08R over 389 trades. **The 15m zones
carry the entire result.** Do not "simplify" that layer away.

### Honest size of the result

23 trades. Under the null that each trade is a coin with the 28.6% win chance
a 1:2.5 target needs, you'd see 9+ wins **18% of the time** by luck. The only
statistically solid number in the whole study is the 389-trade ablation run.
Treat the configuration as a hypothesis to test forward.

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

Fidelity rules the engine holds to — preserve these in any change:

- Decisions on the close of bar `t`, execution at the open of `t+1`. No lookahead.
- MT5 candles are **bid**. A long fills at ask and exits on bid; a short fills
  at bid and exits on ask. The spread asymmetry is real and carried through.
- One position at a time, exactly like `main.py`.
- A bar touching both stop and target is scored as the **loss**.
- Not modelled: commission, swap, slippage beyond spread, intrabar tick order.

Report: `reports/backtest_report.html`, also published at
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

## Open items

Two ideas not yet tested. Both change *logic* rather than a parameter, so the
overfitting risk is lower than another parameter sweep:

1. **Take profit at a structural level** instead of a fixed R multiple — the
   nearest opposing swing / liquidity pool, skipping the trade when that level
   is closer than ~1.5R. The 19% win rate at 1:3 suggests price systematically
   fails to reach an arithmetic target.
2. **Zone quality filter** — require real displacement in the FVG (gap size vs
   ATR), cap zone width, and reject stale zones. `detect_htf_poi` currently
   takes the most recent unmitigated zone with no quality test at all.

Not enabled, available if wanted: `USE_RISK_BASED_LOT`, `MAX_SPREAD_POINTS`,
`USE_BREAKEVEN` — all default-off in `config.py` with notes on when they'd help.
