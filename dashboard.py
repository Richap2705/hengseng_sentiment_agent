"""
dashboard.py
------------
Interactive Plotly Dash dashboard for HengSeng Sentiment Agent.

Features
--------
• Live KPI cards (Price, Predicted n+1, Direction, Position, Directional Accuracy)
• Exchange selector tabs (HKEX / KRX)
• Sentiment Heatmap — tickers × features
• Social Mentions + Composite Sentiment dual-axis chart
• Price + Support/Resistance + n+1 Forecast candlestick-style chart
• OOS Walk-Forward Validation curve with error fill
• Full ranked ticker table with colour-coded direction badges
• Dark-glass design with professional typography
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ── Colour constants ──────────────────────────────────────────────────────────
COL = {
    "BULLISH": "#00e5a0",
    "BEARISH": "#ff4d6d",
    "NEUTRAL": "#8c9ec0",
    "bg": "#0b0e1a",
    "card": "#131929",
    "card2": "#1a2035",
    "border": "#2a3050",
    "text": "#e2e8f0",
    "subtext": "#94a3b8",
    "accent": "#3b82f6",
    "accent2": "#f59e0b",
    "green": "#22d3ee",
    "red": "#f43f5e",
}

DIRECTION_BADGE = {
    "BULLISH": {"color": COL["BULLISH"], "icon": "▲"},
    "BEARISH": {"color": COL["BEARISH"], "icon": "▼"},
    "NEUTRAL": {"color": COL["NEUTRAL"], "icon": "◆"},
}

POSITION_BADGE = {
    "LONG": COL["BULLISH"],
    "SHORT": COL["BEARISH"],
    "FLAT": COL["NEUTRAL"],
}

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="HengSeng Sentiment Intelligence",
    suppress_callback_exceptions=True,
)
app.server.secret_key = "hssent2024"

# ── Global data store (populated by launch_dashboard) ─────────────────────────
_SUMMARIES: dict[str, pd.DataFrame] = {}
_OOS: dict[str, dict] = {}


# ═════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ═════════════════════════════════════════════════════════════════════════════

def _card(children, style: dict | None = None, className: str = "") -> dbc.Card:
    base = {
        "background": COL["card"],
        "border": f"1px solid {COL['border']}",
        "borderRadius": "12px",
        "padding": "16px 20px",
        "marginBottom": "16px",
        "boxShadow": "0 4px 24px rgba(0,0,0,0.5)",
    }
    if style:
        base.update(style)
    return dbc.Card(children, style=base, className=className)


def _kpi_card(label: str, value: str, sub: str = "", color: str = COL["accent"]) -> dbc.Col:
    return dbc.Col(
        _card([
            html.P(label, style={"color": COL["subtext"], "fontSize": "11px",
                                  "margin": "0", "letterSpacing": "0.08em",
                                  "textTransform": "uppercase"}),
            html.H4(value, style={"color": color, "margin": "4px 0 2px",
                                   "fontWeight": "700", "fontSize": "22px"}),
            html.P(sub, style={"color": COL["subtext"], "fontSize": "11px", "margin": "0"}),
        ], style={"minHeight": "88px"}),
        xs=12, sm=6, md=4, lg=2,
    )


def _section_title(text: str) -> html.Div:
    return html.Div([
        html.Span(text, style={"fontSize": "13px", "fontWeight": "600",
                                "color": COL["text"], "letterSpacing": "0.05em"}),
        html.Hr(style={"borderColor": COL["border"], "margin": "6px 0 14px"}),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Main layout
# ═════════════════════════════════════════════════════════════════════════════

def _build_layout(exchanges: list[str]) -> html.Div:
    tabs = dbc.Tabs(
        [dbc.Tab(label=ex, tab_id=ex) for ex in exchanges],
        id="exchange-tabs",
        active_tab=exchanges[0] if exchanges else "HKEX",
        style={"marginBottom": "20px"},
    )

    return html.Div([
        # ── Header ────────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1(
                    "HengSeng Sentiment Intelligence",
                    style={"color": COL["text"], "fontSize": "22px",
                           "fontWeight": "800", "margin": "0", "letterSpacing": "-0.02em"},
                ),
                html.P(
                    "AI-powered social sentiment · Exa news · n+1 price prediction · Walk-forward OOS validation",
                    style={"color": COL["subtext"], "fontSize": "12px", "margin": "2px 0 0"},
                ),
            ]),
            html.Div([
                html.Span("● LIVE", style={"color": COL["BULLISH"], "fontSize": "12px",
                                           "fontWeight": "700", "marginRight": "16px"}),
                html.Span(id="last-updated", style={"color": COL["subtext"], "fontSize": "11px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "padding": "18px 28px 14px",
            "borderBottom": f"1px solid {COL['border']}",
            "background": "linear-gradient(90deg, #0b0e1a 0%, #111827 100%)",
            "marginBottom": "24px",
        }),

        # ── Body ──────────────────────────────────────────────────────────────
        html.Div([
            tabs,

            # KPI row
            dbc.Row(id="kpi-row", style={"marginBottom": "8px"}),

            # Row 1: Social mentions + Sentiment
            dbc.Row([
                dbc.Col(_card([
                    _section_title("Social Mentions × Sentiment Direction"),
                    dcc.Graph(id="mentions-chart", config={"displayModeBar": False},
                              style={"height": "280px"}),
                ]), md=7),
                dbc.Col(_card([
                    _section_title("Composite Sentiment Gauge"),
                    dcc.Graph(id="sentiment-chart", config={"displayModeBar": False},
                              style={"height": "280px"}),
                ]), md=5),
            ]),

            # Row 2: Price forecast + OOS
            dbc.Row([
                dbc.Col(_card([
                    _section_title("Price · Support/Resistance · n+1 Forecast"),
                    dcc.Graph(id="price-chart", config={"displayModeBar": False},
                              style={"height": "300px"}),
                ]), md=6),
                dbc.Col(_card([
                    _section_title("Out-of-Sample Walk-Forward Validation"),
                    dcc.Dropdown(
                        id="oos-ticker-select",
                        clearable=False,
                        style={
                            "background": COL["card2"],
                            "color": COL["text"],
                            "borderColor": COL["border"],
                            "fontSize": "12px",
                            "marginBottom": "8px",
                        },
                    ),
                    dcc.Graph(id="oos-chart", config={"displayModeBar": False},
                              style={"height": "260px"}),
                ]), md=6),
            ]),

            # Row 3: Heatmap
            dbc.Row([
                dbc.Col(_card([
                    _section_title("Sentiment Signal Heatmap — All Tickers"),
                    dcc.Graph(id="heatmap-chart", config={"displayModeBar": False},
                              style={"height": "220px"}),
                ]),),
            ]),

            # Row 4: Ranked table
            dbc.Row([
                dbc.Col(_card([
                    _section_title("Ranked Ticker Summary — Sorted by Social Mentions & Sentiment"),
                    html.Div(id="summary-table"),
                ]),),
            ]),
        ], style={"padding": "0 24px 28px"}),

        # Interval refresh
        dcc.Interval(id="clock-interval", interval=30_000, n_intervals=0),
    ], style={"background": COL["bg"], "minHeight": "100vh", "fontFamily": "'Inter', 'Segoe UI', sans-serif"})


# ═════════════════════════════════════════════════════════════════════════════
# Chart builders
# ═════════════════════════════════════════════════════════════════════════════

def _base_fig(**kwargs) -> go.Figure:
    fig = go.Figure(**kwargs)
    fig.update_layout(
        paper_bgcolor=COL["card"],
        plot_bgcolor=COL["card"],
        font=dict(color=COL["text"], size=11),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(gridcolor=COL["border"], zeroline=False),
        yaxis=dict(gridcolor=COL["border"], zeroline=False),
    )
    return fig


def build_mentions_chart(df: pd.DataFrame) -> go.Figure:
    colors = [COL[d] for d in df["direction"]]
    fig = _base_fig()
    fig.add_trace(go.Bar(
        x=df["ticker"], y=df["social_mentions"],
        marker=dict(color=colors, opacity=0.9,
                    line=dict(color=COL["border"], width=1)),
        name="Social Mentions",
        hovertemplate="<b>%{x}</b><br>Mentions: %{y}<extra></extra>",
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=df["ticker"], y=df["composite_sentiment"],
        mode="lines+markers",
        name="Composite Sentiment",
        line=dict(color=COL["accent2"], width=2),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.4f}<extra></extra>",
        yaxis="y2",
    ))
    fig.update_layout(
        yaxis=dict(title="Mention Count", gridcolor=COL["border"]),
        yaxis2=dict(title="Sentiment", overlaying="y", side="right",
                    gridcolor=COL["border"], zeroline=True,
                    zerolinecolor=COL["subtext"], zerolinewidth=0.8,
                    range=[-1.1, 1.1]),
        barmode="overlay",
    )
    return fig


def build_sentiment_chart(df: pd.DataFrame) -> go.Figure:
    colors = [COL["BULLISH"] if s > 0.05 else COL["BEARISH"] if s < -0.05 else COL["NEUTRAL"]
              for s in df["composite_sentiment"]]
    fig = _base_fig()
    fig.add_trace(go.Bar(
        x=df["ticker"],
        y=df["composite_sentiment"],
        marker=dict(color=colors, opacity=0.88,
                    line=dict(color="rgba(0,0,0,0.3)", width=1)),
        hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=COL["subtext"], line_width=0.8, line_dash="dot")
    fig.add_hrect(y0=0.05, y1=1.1, fillcolor=COL["BULLISH"], opacity=0.04)
    fig.add_hrect(y0=-1.1, y1=-0.05, fillcolor=COL["BEARISH"], opacity=0.04)
    fig.update_yaxes(title="Composite Sentiment", range=[-1.1, 1.1])
    return fig


def build_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = _base_fig()
    x = df["ticker"].tolist()

    # S/R range bars
    for i, row in df.iterrows():
        fig.add_shape(type="line", x0=i - 0.15, x1=i + 0.15,
                      y0=row["support"], y1=row["support"],
                      line=dict(color="#ffd32a", width=2))
        fig.add_shape(type="line", x0=i - 0.15, x1=i + 0.15,
                      y0=row["resistance"], y1=row["resistance"],
                      line=dict(color="#ff4d6d", width=2))
        fig.add_shape(type="line", x0=i, x1=i,
                      y0=row["support"], y1=row["resistance"],
                      line=dict(color=COL["border"], width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=x, y=df["current_price"].tolist(),
        mode="markers", name="Current Price",
        marker=dict(color=COL["accent"], size=12, symbol="circle",
                    line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x}</b><br>Current: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["predicted_price_n_plus_1"].tolist(),
        mode="markers", name="Predicted n+1",
        marker=dict(color=COL["accent2"], size=11, symbol="triangle-up",
                    line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x}</b><br>n+1 Pred: %{y:.2f}<br>Return: %{customdata:.3f}%",
        customdata=df["predicted_return_pct"].tolist(),
    ))

    # Invisible legend entries for S/R
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="#ffd32a"), name="Support"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="#ff4d6d"), name="Resistance"))

    fig.update_xaxes(ticktext=x, tickvals=list(range(len(x))))
    fig.update_yaxes(title="Price (Local Currency)")
    return fig


def build_oos_chart(hist: dict, ticker: str) -> go.Figure:
    if not hist or not hist.get("oos_actuals"):
        fig = _base_fig()
        fig.add_annotation(text="No OOS data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=COL["subtext"], size=13))
        return fig

    actuals = hist["oos_actuals"]
    preds = hist["oos_preds"]
    idx = list(range(len(actuals)))
    dir_acc = hist.get("directional_accuracy_pct", 0)
    oos_mae = hist.get("oos_mae", 0)

    fig = _base_fig()
    fig.add_trace(go.Scatter(
        x=idx, y=actuals,
        mode="lines", name="Actual Price",
        line=dict(color=COL["BULLISH"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=idx, y=preds,
        mode="lines", name="OOS Predicted",
        line=dict(color=COL["accent2"], width=2, dash="dash"),
    ))
    # Error fill
    fig.add_trace(go.Scatter(
        x=idx + idx[::-1],
        y=actuals + preds[::-1],
        fill="toself",
        fillcolor="rgba(59,130,246,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="Prediction Band",
    ))
    fig.add_annotation(
        text=f"Dir Acc: {dir_acc}%  |  OOS MAE: {oos_mae:.3f}",
        xref="paper", yref="paper", x=0.01, y=0.97,
        showarrow=False, bgcolor=COL["card2"],
        bordercolor=COL["border"], borderwidth=1,
        font=dict(color=COL["text"], size=10), align="left",
    )
    fig.update_xaxes(title="Walk-Forward Bar Index")
    fig.update_yaxes(title="Price")
    return fig


def build_heatmap(df: pd.DataFrame) -> go.Figure:
    features = ["social_mentions", "composite_sentiment", "predicted_return_pct",
                 "oos_mae", "directional_accuracy_pct"]
    labels = ["Social Mentions", "Composite Sentiment", "Pred Return %",
              "OOS MAE", "Dir Accuracy %"]

    z = []
    for feat in features:
        col = df[feat].values.astype(float)
        mn, mx = col.min(), col.max()
        rng = mx - mn if mx != mn else 1.0
        z.append(((col - mn) / rng).tolist())

    fig = _base_fig()
    fig.add_trace(go.Heatmap(
        z=z, x=df["ticker"].tolist(), y=labels,
        colorscale=[
            [0, COL["BEARISH"]], [0.5, COL["card2"]], [1, COL["BULLISH"]],
        ],
        showscale=True,
        hovertemplate="<b>%{x}</b><br>%{y}: %{z:.3f}<extra></extra>",
        colorbar=dict(len=0.9, tickfont=dict(color=COL["text"], size=9),
                      bgcolor=COL["card"], bordercolor=COL["border"]),
    ))
    fig.update_layout(margin=dict(l=130, r=20, t=10, b=40))
    return fig


def build_summary_table(df: pd.DataFrame) -> dash_table.DataTable:
    display = df[[
        "ticker", "asset_name", "current_price", "predicted_price_n_plus_1",
        "predicted_return_pct", "social_mentions", "composite_sentiment",
        "direction", "position", "support", "resistance", "directional_accuracy_pct",
    ]].copy()
    display.columns = [
        "Ticker", "Name", "Current Price", "n+1 Pred Price",
        "Pred Return %", "Mentions", "Sentiment",
        "Direction", "Position", "Support", "Resistance", "Dir Acc %",
    ]

    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display.columns],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COL["card2"],
            "color": COL["text"],
            "fontWeight": "600",
            "fontSize": "11px",
            "borderBottom": f"1px solid {COL['border']}",
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
        },
        style_cell={
            "backgroundColor": COL["card"],
            "color": COL["text"],
            "fontSize": "12px",
            "padding": "9px 14px",
            "border": f"1px solid {COL['border']}",
            "fontFamily": "'Inter', 'Segoe UI', monospace",
        },
        style_data_conditional=[
            # Direction colouring
            {
                "if": {"filter_query": "{Direction} = BULLISH"},
                "color": COL["BULLISH"],
                "fontWeight": "600",
            },
            {
                "if": {"filter_query": "{Direction} = BEARISH"},
                "color": COL["BEARISH"],
                "fontWeight": "600",
            },
            {
                "if": {"filter_query": "{Direction} = NEUTRAL"},
                "color": COL["NEUTRAL"],
            },
            # Position colouring
            {
                "if": {"filter_query": "{Position} = LONG"},
                "color": COL["BULLISH"],
            },
            {
                "if": {"filter_query": "{Position} = SHORT"},
                "color": COL["BEARISH"],
            },
            # Alternating row
            {
                "if": {"row_index": "odd"},
                "backgroundColor": COL["card2"],
            },
        ],
        sort_action="native",
        filter_action="native",
        page_size=10,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Callbacks
# ═════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("last-updated", "children"),
    Input("clock-interval", "n_intervals"),
)
def update_clock(_):
    from datetime import datetime, timezone
    return f"Last refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M UTC')}"


@app.callback(
    Output("kpi-row", "children"),
    Output("mentions-chart", "figure"),
    Output("sentiment-chart", "figure"),
    Output("price-chart", "figure"),
    Output("oos-ticker-select", "options"),
    Output("oos-ticker-select", "value"),
    Output("heatmap-chart", "figure"),
    Output("summary-table", "children"),
    Input("exchange-tabs", "active_tab"),
)
def update_exchange(exchange: str):
    df = _SUMMARIES.get(exchange, pd.DataFrame())
    oos = _OOS.get(exchange, {})

    if df.empty:
        empty_fig = _base_fig()
        empty_fig.add_annotation(text="No data", xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False,
                                  font=dict(color=COL["subtext"]))
        return [], empty_fig, empty_fig, empty_fig, [], None, empty_fig, html.P("No data")

    top = df.iloc[0]
    kpi_cards = [
        _kpi_card("Top Ticker", top["ticker"], top["asset_name"], COL["accent"]),
        _kpi_card("Current Price", f"{top['current_price']:,.2f}",
                  "Local currency", COL["text"]),
        _kpi_card("n+1 Prediction", f"{top['predicted_price_n_plus_1']:,.2f}",
                  f"Return: {top['predicted_return_pct']:+.3f}%",
                  COL["BULLISH"] if top["predicted_return_pct"] > 0 else COL["BEARISH"]),
        _kpi_card("Direction", f"{DIRECTION_BADGE[top['direction']]['icon']} {top['direction']}",
                  f"Position: {top['position']}",
                  DIRECTION_BADGE[top["direction"]]["color"]),
        _kpi_card("Dir. Accuracy", f"{top['directional_accuracy_pct']:.1f}%",
                  f"OOS MAE: {top['oos_mae']:.4f}", COL["accent2"]),
        _kpi_card("Social Mentions", f"{int(top['social_mentions']):,}",
                  f"10-Day Window", COL["green"]),
    ]

    ticker_opts = [{"label": f"{r['ticker']} — {r['asset_name']}", "value": r["ticker"]}
                   for _, r in df.iterrows()]
    top_ticker = df.iloc[0]["ticker"]

    return (
        kpi_cards,
        build_mentions_chart(df),
        build_sentiment_chart(df),
        build_price_chart(df),
        ticker_opts,
        top_ticker,
        build_heatmap(df),
        build_summary_table(df),
    )


@app.callback(
    Output("oos-chart", "figure"),
    Input("oos-ticker-select", "value"),
    Input("exchange-tabs", "active_tab"),
)
def update_oos(ticker: str, exchange: str):
    oos = _OOS.get(exchange, {})
    hist = oos.get(ticker, {})
    return build_oos_chart(hist, ticker or "")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def launch_dashboard(summaries: dict[str, pd.DataFrame], oos: dict[str, dict],
                     host: str = "127.0.0.1", port: int = 8050):
    """Inject data and start the Dash server."""
    global _SUMMARIES, _OOS
    _SUMMARIES = summaries
    _OOS = oos

    exchanges = list(summaries.keys()) or ["HKEX"]
    app.layout = _build_layout(exchanges)

    print(f"\n{'='*60}")
    print(f"  Dashboard  →  http://{host}:{port}/")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    app.run(host=host, port=port, debug=False)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone: load from saved reports/
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summaries: dict[str, pd.DataFrame] = {}
    oos: dict[str, dict] = {}

    reports_dir = "reports"
    for fname in os.listdir(reports_dir) if os.path.isdir(reports_dir) else []:
        if fname.endswith("_summary.json"):
            exchange = fname.replace("_summary.json", "")
            with open(os.path.join(reports_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            summaries[exchange] = pd.DataFrame(data)

    if not summaries:
        print("No saved reports found. Run main.py first, or run:  python main.py --demo --dashboard")
    else:
        launch_dashboard(summaries, oos)
