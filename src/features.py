"""
features.py
-----------
Feature engineering layer:
  - Hybrid VADER + financial-lexicon sentiment scoring (English + Chinese + Korean)
  - Engagement-weighted aggregation aligned to price bars (no look-ahead)
  - Dynamic support / resistance via pivot-point method
  - Rolling sentiment persistence features
"""

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# ── Domain financial lexicon (English + Cantonese/Mandarin + Korean) ──────────
BULLISH_CUES = [
    # English
    "buy", "bull", "bullish", "surge", "rally", "breakout", "beat",
    "upgrade", "long", "accumulate", "outperform", "rebound", "gap up",
    # Chinese (HK)
    "利好", "買入", "升", "突破", "看好", "反彈", "強勢", "超買",
    # Korean (KRX)
    "매수", "상승", "돌파", "강세", "급등", "반등", "목표가상향",
]
BEARISH_CUES = [
    # English
    "sell", "bear", "bearish", "drop", "crash", "miss", "downgrade",
    "short", "resistance", "gap down", "overvalued", "dump", "liquidate",
    # Chinese (HK)
    "利空", "沽出", "跌", "看淡", "暴跌", "破位", "弱勢",
    # Korean (KRX)
    "매도", "하락", "약세", "급락", "목표가하향", "손절",
]


def score_text_sentiment(text: str) -> float:
    """
    Hybrid sentiment score in [-1.0, 1.0].
    Combines VADER compound score with a domain financial-lexicon adjustment.
    """
    if not text or not isinstance(text, str):
        return 0.0
    base = analyzer.polarity_scores(text)["compound"]
    lowered = text.lower()
    bull_hits = sum(1 for w in BULLISH_CUES if w in lowered)
    bear_hits = sum(1 for w in BEARISH_CUES if w in lowered)
    lexicon_adj = np.clip((bull_hits - bear_hits) * 0.15, -0.45, 0.45)
    return float(np.clip(base + lexicon_adj, -1.0, 1.0))


def compute_support_resistance(price_df: pd.DataFrame, window: int = 24) -> tuple[float, float]:
    """
    Dynamic Support & Resistance levels using rolling swing extrema + pivot points.

    Method:
      Pivot  = (SwingHigh + SwingLow + LastClose) / 3
      S1     = 2 * Pivot − SwingHigh
      R1     = 2 * Pivot − SwingLow
    Returns (support, resistance) as the more conservative of S1/SwingLow and R1/SwingHigh.
    """
    recent = price_df.tail(window)
    swing_low = float(recent["Low"].min())
    swing_high = float(recent["High"].max())
    last_close = float(recent["Close"].iloc[-1])

    pivot = (swing_high + swing_low + last_close) / 3.0
    support_1 = round(2 * pivot - swing_high, 2)
    resistance_1 = round(2 * pivot - swing_low, 2)
    return min(support_1, round(swing_low, 2)), max(resistance_1, round(swing_high, 2))


def align_events_to_bars(
    price_df: pd.DataFrame,
    social_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Strictly align social posts & news to the **next** available price bar
    (bar timestamp >= event timestamp) to prevent look-ahead leakage.

    Feature columns added:
      social_mentions, social_sentiment, social_raw_sentiment, total_engagement,
      news_count, news_sentiment,
      rolling_social_sent_3, rolling_news_sent_3,
      return_1, volatility_6, momentum_6
    """
    bars = price_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    bars["return_1"] = bars["Close"].pct_change().fillna(0.0)
    bars["volatility_6"] = bars["return_1"].rolling(6, min_periods=1).std().fillna(0.0)
    bars["momentum_6"] = bars["Close"].pct_change(6).fillna(0.0)

    # ── Social aggregation ────────────────────────────────────────────────────
    if not social_df.empty:
        s_df = social_df.copy()
        s_df["raw_sent"] = s_df["text"].apply(score_text_sentiment)
        # Engagement weight: likes + reposts (high signal) + views + follower reach
        s_df["weight"] = (
            1.0
            + np.log1p(s_df["likes"].clip(lower=0))
            + 1.5 * np.log1p(s_df["reposts"].clip(lower=0))
            + 0.25 * np.log1p(s_df["views"].clip(lower=0))
            + 0.35 * np.log1p(s_df["followers"].clip(lower=0))
        )
        s_df["weighted_sent"] = s_df["raw_sent"] * s_df["weight"]
        s_df = s_df.sort_values("timestamp_utc")

        merged_s = pd.merge_asof(
            s_df,
            bars[["timestamp_utc"]].assign(bar_time=bars["timestamp_utc"]),
            on="timestamp_utc",
            direction="forward",
        )
        agg_social = (
            merged_s.groupby("bar_time")
            .agg(
                social_mentions=("text", "count"),
                social_sentiment=("weighted_sent", "mean"),
                social_raw_sentiment=("raw_sent", "mean"),
                total_engagement=("weight", "sum"),
            )
            .reset_index()
            .rename(columns={"bar_time": "timestamp_utc"})
        )
        bars = bars.merge(agg_social, on="timestamp_utc", how="left")
    else:
        bars["social_mentions"] = 0
        bars["social_sentiment"] = 0.0
        bars["social_raw_sentiment"] = 0.0
        bars["total_engagement"] = 0.0

    # ── News aggregation ──────────────────────────────────────────────────────
    if not news_df.empty:
        n_df = news_df.copy()
        n_df["news_sent"] = n_df["text"].apply(score_text_sentiment)
        n_df = n_df.sort_values("timestamp_utc")

        merged_n = pd.merge_asof(
            n_df,
            bars[["timestamp_utc"]].assign(bar_time=bars["timestamp_utc"]),
            on="timestamp_utc",
            direction="forward",
        )
        agg_news = (
            merged_n.groupby("bar_time")
            .agg(
                news_count=("text", "count"),
                news_sentiment=("news_sent", "mean"),
            )
            .reset_index()
            .rename(columns={"bar_time": "timestamp_utc"})
        )
        bars = bars.merge(agg_news, on="timestamp_utc", how="left")
    else:
        bars["news_count"] = 0
        bars["news_sentiment"] = 0.0

    fill_cols = [
        "social_mentions", "social_sentiment", "social_raw_sentiment",
        "total_engagement", "news_count", "news_sentiment",
    ]
    bars[fill_cols] = bars[fill_cols].fillna(0.0)

    # Rolling 3-bar sentiment persistence
    bars["rolling_social_sent_3"] = bars["social_sentiment"].rolling(3, min_periods=1).mean()
    bars["rolling_news_sent_3"] = bars["news_sentiment"].rolling(3, min_periods=1).mean()
    return bars
