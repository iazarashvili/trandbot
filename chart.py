"""Visual chart module — shows what the bot sees.

Generates interactive HTML charts with:
  - Candlesticks (1H and 5m)
  - EMA trend line
  - HTF POI zones (FVG + OB)
  - DOL target levels
  - 5m FVG execution zones
  - Entry / SL / TP lines
  - Liquidity sweep markers

Usage from engine:
    from chart import BotChart
    chart = BotChart()
    chart.plot_analysis(df_htf, df_ltf, poi, dol, setup, ctx)

Standalone:
    python chart.py --symbol BTCUSD
"""

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger("smc_bot")

CHART_DIR = Path(__file__).resolve().parent / "charts"


class BotChart:
    """Generates analysis charts showing what the bot sees."""

    def __init__(self, output_dir: Path = CHART_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def plot_analysis(
        self,
        symbol: str,
        df_htf: pd.DataFrame,
        df_ltf: pd.DataFrame,
        poi=None,
        dol_target: Optional[float] = None,
        setup: Optional[dict] = None,
        trend: str = "NEUTRAL",
        ema_period: int = 100,
        save: bool = True,
    ) -> str:
        """Generate a two-panel chart: 1H overview + 5m execution.

        Returns the path to the saved HTML file.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=(
                f"{symbol} — 1H Narrative (HTF)",
                f"{symbol} — 5m Execution (LTF)",
            ),
            row_heights=[0.55, 0.45],
        )

        # ============================================================
        # TOP PANEL: 1H candlesticks + EMA + POI + DOL
        # ============================================================
        self._add_candlesticks(fig, df_htf, row=1, col=1)
        self._add_ema(fig, df_htf, ema_period, row=1, col=1)

        if poi is not None:
            self._add_poi_zone(fig, df_htf, poi, row=1, col=1)

        if dol_target is not None:
            self._add_dol_line(fig, df_htf, dol_target, row=1, col=1)

        # ============================================================
        # BOTTOM PANEL: 5m candlesticks + FVG + Entry/SL/TP
        # ============================================================
        # Show last 60 candles for clarity
        ltf_view = df_ltf.iloc[-60:] if len(df_ltf) > 60 else df_ltf
        self._add_candlesticks(fig, ltf_view, row=2, col=1)

        # Mark the 5m FVG if we have a setup
        if setup is not None:
            self._add_fvg_zone(fig, ltf_view, setup, row=2, col=1)
            self._add_trade_levels(fig, ltf_view, setup, row=2, col=1)

        # If POI is visible on LTF, shade it there too
        if poi is not None:
            self._add_poi_zone(fig, ltf_view, poi, row=2, col=1, opacity=0.08)

        # ============================================================
        # Layout
        # ============================================================
        trend_emoji = {"BULLISH": "BULL", "BEARISH": "BEAR"}.get(trend, "NEUTRAL")
        title = f"{symbol} | {trend_emoji} | {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        if setup:
            title += f" | SETUP: {setup['direction']}"

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            template="plotly_dark",
            height=900,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=2, col=1)

        # Save
        if save:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{ts}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            logger.info("Chart saved: %s", filepath)

            # Also save a "latest" file that always has the most recent
            latest = self.output_dir / f"{symbol}_latest.html"
            fig.write_html(str(latest))

            return str(filepath)

        return ""

    def plot_htf_only(
        self,
        symbol: str,
        df_htf: pd.DataFrame,
        poi=None,
        dol_target: Optional[float] = None,
        trend: str = "NEUTRAL",
        ema_period: int = 100,
    ) -> str:
        """Quick 1H-only chart for status checks."""
        fig = go.Figure()

        self._add_candlesticks(fig, df_htf, row=None, col=None, use_subplots=False)
        self._add_ema(fig, df_htf, ema_period, row=None, col=None, use_subplots=False)

        if poi is not None:
            self._add_poi_zone(fig, df_htf, poi, row=None, col=None, use_subplots=False)

        if dol_target is not None:
            self._add_dol_line(fig, df_htf, dol_target, row=None, col=None, use_subplots=False)

        fig.update_layout(
            title=f"{symbol} 1H | {trend} | {datetime.datetime.utcnow().strftime('%H:%M')} UTC",
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False,
        )

        latest = self.output_dir / f"{symbol}_htf.html"
        fig.write_html(str(latest))
        return str(latest)

    # ================================================================
    # Building blocks
    # ================================================================

    def _add_candlesticks(self, fig, df, row=1, col=1, use_subplots=True):
        times = pd.to_datetime(df["time"], unit="s") if df["time"].dtype != "datetime64[ns]" else df["time"]

        candle = go.Candlestick(
            x=times,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        )

        if use_subplots and row is not None:
            fig.add_trace(candle, row=row, col=col)
        else:
            fig.add_trace(candle)

    def _add_ema(self, fig, df, period, row=1, col=1, use_subplots=True):
        if len(df) < period:
            return

        times = pd.to_datetime(df["time"], unit="s") if df["time"].dtype != "datetime64[ns]" else df["time"]
        ema = df["close"].ewm(span=period, adjust=False).mean()

        trace = go.Scatter(
            x=times,
            y=ema,
            name=f"EMA({period})",
            line=dict(color="#ffd54f", width=1.5, dash="dot"),
            showlegend=True,
        )

        if use_subplots and row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)

    def _add_poi_zone(self, fig, df, poi, row=1, col=1, opacity=0.15, use_subplots=True):
        """Shade the POI (FVG+OB) zone."""
        times = pd.to_datetime(df["time"], unit="s") if df["time"].dtype != "datetime64[ns]" else df["time"]

        color = "rgba(38, 166, 154, {a})" if poi.type == "BULLISH" else "rgba(239, 83, 80, {a})"
        fill_color = color.format(a=opacity)
        line_color = color.format(a=0.6)

        # Draw from POI creation to end of chart
        x_start = times.iloc[max(0, poi.index)] if poi.index >= 0 and poi.index < len(times) else times.iloc[0]
        x_end = times.iloc[-1]

        shape = dict(
            type="rect",
            x0=x_start, x1=x_end,
            y0=poi.bottom, y1=poi.top,
            fillcolor=fill_color,
            line=dict(color=line_color, width=1),
        )

        label_text = f"{'BUY' if poi.type == 'BULLISH' else 'SELL'} POI"

        if use_subplots and row is not None:
            shape["xref"] = f"x{row}" if row > 1 else "x"
            shape["yref"] = f"y{row}" if row > 1 else "y"
            fig.add_shape(shape)
            fig.add_annotation(
                x=x_end, y=poi.top,
                text=label_text,
                showarrow=False,
                font=dict(size=10, color=line_color.replace(str(opacity), "1")),
                xref=shape["xref"], yref=shape["yref"],
                xanchor="right",
            )
        else:
            fig.add_shape(shape)
            fig.add_annotation(
                x=x_end, y=poi.top, text=label_text,
                showarrow=False, font=dict(size=10),
                xanchor="right",
            )

    def _add_dol_line(self, fig, df, dol_target, row=1, col=1, use_subplots=True):
        """Draw the Draw on Liquidity target as a dashed line."""
        times = pd.to_datetime(df["time"], unit="s") if df["time"].dtype != "datetime64[ns]" else df["time"]

        xref = f"x{row}" if use_subplots and row and row > 1 else "x"
        yref = f"y{row}" if use_subplots and row and row > 1 else "y"

        fig.add_shape(
            type="line",
            x0=times.iloc[0], x1=times.iloc[-1],
            y0=dol_target, y1=dol_target,
            line=dict(color="#ab47bc", width=2, dash="dash"),
            xref=xref, yref=yref,
        )
        fig.add_annotation(
            x=times.iloc[-1], y=dol_target,
            text=f"DOL {dol_target:.2f}",
            showarrow=True, arrowhead=2,
            font=dict(size=10, color="#ab47bc"),
            xref=xref, yref=yref,
            ax=-50, ay=-15,
        )

    def _add_fvg_zone(self, fig, df, setup, row=2, col=1):
        """Mark the 5m FVG zone that triggered the entry."""
        times = pd.to_datetime(df["time"], unit="s") if df["time"].dtype != "datetime64[ns]" else df["time"]

        if len(df) < 3:
            return

        # The FVG is formed by the last 3 closed candles
        c1_time = times.iloc[-3]
        c3_time = times.iloc[-1]

        if setup["direction"] == "BUY":
            fvg_bottom = float(df.iloc[-3]["high"])
            fvg_top = float(df.iloc[-1]["low"])
        else:
            fvg_top = float(df.iloc[-3]["low"])
            fvg_bottom = float(df.iloc[-1]["high"])

        if fvg_top <= fvg_bottom:
            return  # no valid FVG to draw

        xref = f"x{row}" if row > 1 else "x"
        yref = f"y{row}" if row > 1 else "y"

        color = "rgba(100, 181, 246, 0.25)" if setup["direction"] == "BUY" else "rgba(255, 138, 101, 0.25)"
        border = "#64b5f6" if setup["direction"] == "BUY" else "#ff8a65"

        fig.add_shape(
            type="rect",
            x0=c1_time, x1=c3_time,
            y0=fvg_bottom, y1=fvg_top,
            fillcolor=color,
            line=dict(color=border, width=2),
            xref=xref, yref=yref,
        )
        fig.add_annotation(
            x=c3_time, y=(fvg_top + fvg_bottom) / 2,
            text="FVG",
            showarrow=False,
            font=dict(size=11, color=border, family="Arial Black"),
            xref=xref, yref=yref,
            xanchor="left",
        )

    def _add_trade_levels(self, fig, df, setup, row=2, col=1):
        """Draw entry, SL, and TP lines."""
        times = pd.to_datetime(df["time"], unit="s") if df["time"].dtype != "datetime64[ns]" else df["time"]

        xref = f"x{row}" if row > 1 else "x"
        yref = f"y{row}" if row > 1 else "y"

        x_start = times.iloc[-10] if len(times) > 10 else times.iloc[0]
        x_end = times.iloc[-1]

        levels = [
            ("entry", setup.get("entry"), "#ffd54f", "solid", "Entry"),
            ("sl", setup.get("sl"), "#ef5350", "dash", "SL"),
            ("tp", setup.get("tp"), "#26a69a", "dash", "TP"),
        ]

        for key, price, color, dash, label in levels:
            if price is None:
                continue
            fig.add_shape(
                type="line",
                x0=x_start, x1=x_end,
                y0=price, y1=price,
                line=dict(color=color, width=2, dash=dash),
                xref=xref, yref=yref,
            )
            fig.add_annotation(
                x=x_end, y=price,
                text=f"{label} {price:.2f}",
                showarrow=False,
                font=dict(size=10, color=color),
                xref=xref, yref=yref,
                xanchor="left",
                bgcolor="rgba(0,0,0,0.5)",
            )

        # Risk/Reward box annotation
        entry = setup.get("entry")
        sl = setup.get("sl")
        tp = setup.get("tp")
        if entry and sl and tp:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0
            fig.add_annotation(
                x=x_start, y=entry,
                text=f"R:R = 1:{rr:.1f}",
                showarrow=False,
                font=dict(size=12, color="white"),
                xref=xref, yref=yref,
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="#ffd54f",
                borderwidth=1,
                borderpad=4,
            )


def generate_chart_standalone(symbol: str = "BTCUSD"):
    """Generate a chart from live MT5 data (standalone usage)."""
    import MetaTrader5 as mt5
    from config import (
        HTF, HTF_CANDLES_LOOKBACK, LTF, LTF_CANDLES_LOOKBACK,
        LIQUIDITY_TF, LIQUIDITY_CANDLES, SWING_STRENGTH,
        USE_CLOSED_CANDLES_ONLY, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    )
    from strategy import SMCStrategy
    from core.symbol_context import SymbolConfig

    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 login failed: {mt5.last_error()}")
        return

    cfg = SymbolConfig.load(symbol)

    # Fetch data
    rates_htf = mt5.copy_rates_from_pos(symbol, HTF, 0, HTF_CANDLES_LOOKBACK)
    rates_ltf = mt5.copy_rates_from_pos(symbol, LTF, 0, LTF_CANDLES_LOOKBACK)
    rates_liq = mt5.copy_rates_from_pos(symbol, LIQUIDITY_TF, 0, LIQUIDITY_CANDLES)

    if rates_htf is None or rates_ltf is None:
        print("Failed to fetch data")
        mt5.shutdown()
        return

    df_htf = pd.DataFrame(rates_htf)
    df_ltf = pd.DataFrame(rates_ltf)
    df_liq = pd.DataFrame(rates_liq) if rates_liq is not None else None

    # Analyze
    trend = SMCStrategy.get_htf_trend(
        df_htf, cfg.trend_ema_period, use_closed_candles=USE_CLOSED_CANDLES_ONLY)

    poi = SMCStrategy.detect_htf_poi(
        df_htf=df_htf,
        use_trend_filter=cfg.use_trend_filter,
        ema_period=cfg.trend_ema_period,
        use_closed_candles=USE_CLOSED_CANDLES_ONLY,
    )

    # DOL
    tick = mt5.symbol_info_tick(symbol)
    current_price = (tick.bid + tick.ask) / 2 if tick else 0
    dol_target = None
    if df_liq is not None and poi is not None:
        direction = "BUY" if poi.type == "BULLISH" else "SELL"
        dol_target = SMCStrategy.find_nearest_liquidity(
            df_liq, direction, current_price,
            strength=SWING_STRENGTH, use_closed_candles=True)

    # LTF setup (if available)
    setup = None
    if poi is not None:
        setup = SMCStrategy.check_ltf_confirmation(
            df_ltf, poi, cfg.rrr,
            use_closed_candles=USE_CLOSED_CANDLES_ONLY,
            stop_mode=cfg.stop_mode, buffer_atr=0.5)

    # Generate chart
    chart = BotChart()
    path = chart.plot_analysis(
        symbol=symbol,
        df_htf=df_htf,
        df_ltf=df_ltf,
        poi=poi,
        dol_target=dol_target,
        setup=setup,
        trend=trend,
        ema_period=cfg.trend_ema_period,
    )

    mt5.shutdown()
    print(f"Chart saved: {path}")
    print(f"Trend: {trend}")
    print(f"POI: {poi}")
    print(f"DOL: {dol_target}")
    print(f"Setup: {setup}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate bot analysis chart")
    ap.add_argument("--symbol", default="BTCUSD", help="Symbol to chart")
    args = ap.parse_args()
    generate_chart_standalone(args.symbol)
