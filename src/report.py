"""
report.py
---------
Static matplotlib report generator (PNG + CSV + JSON).
Produces a 2×2 panel chart per exchange:
  Panel 1: Social Mentions ranked bar
  Panel 2: Composite Sentiment heatmap-style bar
  Panel 3: Price + Support/Resistance + n+1 Forecast scatter
  Panel 4: Walk-Forward OOS Validation curve (top ranked ticker)
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import numpy as np


# ── Colour palette ─────────────────────────────────────────────────────────────
_C = {
    "BULLISH": "#00c896",
    "BEARISH": "#ff4757",
    "NEUTRAL": "#747d8c",
    "blue": "#1e90ff",
    "orange": "#ffa502",
    "bg": "#0f1117",
    "panel": "#1a1f2e",
    "text": "#e0e0e0",
    "grid": "#2a2f3e",
}


def _style_ax(ax, title: str):
    """Apply dark-panel styling to an axes object."""
    ax.set_facecolor(_C["panel"])
    ax.set_title(title, color=_C["text"], fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=_C["text"], labelsize=8)
    ax.xaxis.label.set_color(_C["text"])
    ax.yaxis.label.set_color(_C["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_C["grid"])
    ax.grid(axis="y", color=_C["grid"], linewidth=0.6, linestyle="--")


def generate_visual_report(
    exchange: str,
    summary_df: pd.DataFrame,
    oos_history: dict,
    output_dir: str = "reports",
) -> str:
    """
    Generate a dark-themed multi-panel PNG report and save CSV/JSON.

    Panels:
      1. Total Social Mentions (10-day) sorted by mention rank
      2. Engagement-Weighted Composite Sentiment
      3. Current Price ± Support/Resistance + n+1 Predicted Price
      4. Out-of-Sample Walk-Forward Validation (top-ranked ticker)

    Returns path to the saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    ranked = summary_df.sort_values(
        by=["social_mentions", "composite_sentiment"], ascending=[False, False]
    ).reset_index(drop=True)

    plt.rcParams.update({
        "figure.facecolor": _C["bg"],
        "axes.facecolor": _C["panel"],
        "text.color": _C["text"],
        "font.family": "DejaVu Sans",
    })

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.patch.set_facecolor(_C["bg"])
    fig.suptitle(
        f"{exchange}  ·  Sentiment & n+1 Price Prediction Report",
        fontsize=15,
        fontweight="bold",
        color=_C["text"],
        y=0.98,
    )

    # ── Panel 1: Social Mentions Bar ──────────────────────────────────────────
    ax1 = axes[0, 0]
    colors_1 = [_C[d] for d in ranked["direction"]]
    bars1 = ax1.bar(ranked["ticker"], ranked["social_mentions"], color=colors_1, width=0.55, zorder=2)
    ax1.set_xlabel("Ticker")
    ax1.set_ylabel("Mention Count")
    for bar, val in zip(bars1, ranked["social_mentions"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(int(val)), ha="center", va="bottom", color=_C["text"], fontsize=8)
    patches = [mpatches.Patch(color=_C[d], label=d) for d in ["BULLISH", "BEARISH", "NEUTRAL"]]
    ax1.legend(handles=patches, fontsize=7, facecolor=_C["panel"], labelcolor=_C["text"])
    _style_ax(ax1, "① Social Mentions Ranking — 10-Day Window")

    # ── Panel 2: Composite Sentiment ─────────────────────────────────────────
    ax2 = axes[0, 1]
    sent_colors = [_C["BULLISH"] if s > 0.05 else _C["BEARISH"] if s < -0.05 else _C["NEUTRAL"]
                   for s in ranked["composite_sentiment"]]
    bars2 = ax2.bar(ranked["ticker"], ranked["composite_sentiment"], color=sent_colors, width=0.55, zorder=2)
    ax2.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Ticker")
    ax2.set_ylabel("Composite Sentiment (−1 to +1)")
    for bar, val in zip(bars2, ranked["composite_sentiment"]):
        ypos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.015
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{val:.3f}", ha="center", va="bottom", color=_C["text"], fontsize=7)
    _style_ax(ax2, "② Engagement-Weighted Composite Sentiment Score")

    # ── Panel 3: Price + Support/Resistance + n+1 Forecast ───────────────────
    ax3 = axes[1, 0]
    x_idx = np.arange(len(ranked))
    ax3.vlines(x_idx, ranked["support"], ranked["resistance"],
               color=_C["blue"], linewidth=2, alpha=0.4, label="S/R Range")
    ax3.scatter(x_idx, ranked["current_price"], color=_C["blue"], s=70, zorder=5, label="Current Price", marker="o")
    ax3.scatter(x_idx, ranked["predicted_price_n_plus_1"],
                color=_C["orange"], s=90, zorder=6, label="Predicted n+1 Price", marker="^")
    ax3.scatter(x_idx, ranked["support"], color="#ffdd59", s=40, zorder=4, marker="_", linewidths=2)
    ax3.scatter(x_idx, ranked["resistance"], color="#ff6b81", s=40, zorder=4, marker="_", linewidths=2)
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(ranked["ticker"], color=_C["text"], fontsize=8)
    ax3.set_ylabel("Price (Local Currency)")
    ax3.legend(fontsize=7, facecolor=_C["panel"], labelcolor=_C["text"])
    _style_ax(ax3, "③ Current Price, Support/Resistance & n+1 Forecast")

    # ── Panel 4: OOS Walk-Forward Validation ──────────────────────────────────
    ax4 = axes[1, 1]
    top_ticker = ranked.loc[0, "ticker"]
    hist = oos_history.get(top_ticker, {})
    if hist and hist.get("oos_actuals"):
        ax4.plot(hist["oos_actuals"], label="Actual Price", color=_C["BULLISH"], linewidth=1.8)
        ax4.plot(hist["oos_preds"], label="OOS Predicted", color=_C["orange"],
                 linestyle="--", linewidth=1.8, alpha=0.85)
        ax4.fill_between(range(len(hist["oos_actuals"])),
                         hist["oos_actuals"], hist["oos_preds"],
                         alpha=0.08, color=_C["blue"])
        ax4.set_xlabel("Walk-Forward Bar Index")
        ax4.set_ylabel("Price")
        ax4.legend(fontsize=7, facecolor=_C["panel"], labelcolor=_C["text"])
        _style_ax(
            ax4,
            f"④ OOS Validation — {top_ticker}  |  Dir Acc: {hist['directional_accuracy_pct']}%"
            f"  |  MAE: {hist['oos_mae']:.2f}  |  Naive MAE: {hist['naive_persistence_mae']:.2f}",
        )
    else:
        ax4.text(0.5, 0.5, "Insufficient OOS Bars\n(need ≥ 20 training bars)",
                 ha="center", va="center", color=_C["text"], fontsize=10)
        _style_ax(ax4, f"④ OOS Validation — {top_ticker}")

    fig.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.08, hspace=0.32, wspace=0.25)
    chart_path = os.path.join(output_dir, f"{exchange}_sentiment_report.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=_C["bg"])
    plt.close(fig)

    csv_path = os.path.join(output_dir, f"{exchange}_ranked_report.csv")
    ranked.to_csv(csv_path, index=False)
    return chart_path
