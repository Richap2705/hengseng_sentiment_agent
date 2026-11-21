"""
main.py
-------
HengSeng Sentiment Price Prediction Agent — Entry Point

Usage:
  python main.py --demo          # Offline synthetic data (no API tokens needed)
  python main.py                 # Live mode (requires APIFY_TOKEN + EXA_API_KEY)
  python main.py --demo --dashboard   # Launch interactive Dash dashboard after pipeline
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.config import EXCHANGES, LOOKBACK_DAYS, BAR_INTERVAL
from src.scrapers import fetch_price_bars, scrape_social_apify, fetch_news_exa
from src.features import align_events_to_bars, compute_support_resistance
from src.model import walk_forward_predict_n_plus_1
from src.report import generate_visual_report


# ─────────────────────────────────────────────────────────────────────────────
# Demo / offline data generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_demo_social_and_news(
    ticker: str, name: str, exchange: str, price_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthetic data strictly for offline pipeline testing.
    Realistic engagement distributions; two social sources per exchange.
    """
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    timestamps = price_df["timestamp_utc"].tolist()

    # Exchange-specific source labels (mirrors the live Apify sources)
    if exchange == "HKEX":
        sources = ["LIHKG_Forum", "Telegram_HK"]
        sample_posts = [
            f"{name} ({ticker}) 量能放大，突破關鍵阻力 bullish breakout incoming",
            f"Watching {ticker} closely — volume surge confirms accumulation phase",
            f"Taking profit on {name}, resistance holding at current levels",
            f"{ticker} earnings outlook solid 利好, accumulating on dips 買入",
            f"Short-term pullback on {ticker} 看淡 — waiting for support confirmation",
            f"{name} 反彈 rebound today, 買入機會",
            f"{ticker} 跌穿支撑位 drop below support, possible 暴跌",
        ]
    else:  # KRX
        sources = ["Naver_Finance", "Telegram_KR"]
        sample_posts = [
            f"{name} ({ticker}) 거래량 급증 매수 신호 strong volume breakout",
            f"{ticker} 상승추세 유지 — 목표가 상향 upgrade signal",
            f"{name} 차익실현 구간 — 단기 저항선 도달",
            f"{ticker} 실적 호조 예상 하이닉스 반등 rebound expected",
            f"{ticker} 하락 돌파 매도 pressure, 지지선 테스트",
            f"{name} 외국인 순매수 지속 bullish institutional flow",
            f"{ticker} 약세 패턴 bearish divergence 주의",
        ]

    social_rows = []
    for _ in range(55):
        ts = rng.choice(timestamps) - timedelta(minutes=int(rng.integers(5, 55)))
        social_rows.append({
            "ticker": ticker,
            "source": str(rng.choice(sources)),
            "timestamp_utc": pd.to_datetime(ts, utc=True),
            "text": str(rng.choice(sample_posts)),
            "likes": int(rng.integers(0, 200)),
            "reposts": int(rng.integers(0, 50)),
            "views": int(rng.integers(50, 8000)),
            "followers": int(rng.integers(20, 50000)),
        })

    news_rows = []
    for i in range(8):
        ts = timestamps[min(len(timestamps) - 1, (i + 1) * max(1, len(timestamps) // 9))]
        news_rows.append({
            "ticker": ticker,
            "source": "Exa_News",
            "timestamp_utc": pd.to_datetime(ts, utc=True),
            "title": f"{name} Market Update #{i + 1}",
            "text": (
                f"{name} ({ticker}) sees active institutional flow amid index volatility. "
                f"Analysts {'upgrade' if i % 2 == 0 else 'maintain'} target on improving fundamentals."
            ),
            "url": f"https://exa.ai/results/{ticker}/{i + 1}",
        })

    return pd.DataFrame(social_rows), pd.DataFrame(news_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Signal derivation
# ─────────────────────────────────────────────────────────────────────────────
def derive_signal_and_position(
    composite_sentiment: float, pred_return_pct: float
) -> tuple[str, str]:
    """Map composite sentiment + predicted return to a direction and position."""
    if composite_sentiment > 0.05 and pred_return_pct > 0.15:
        return "BULLISH", "LONG"
    elif composite_sentiment < -0.05 and pred_return_pct < -0.15:
        return "BEARISH", "SHORT"
    elif pred_return_pct > 0.25:
        return "BULLISH", "LONG"
    elif pred_return_pct < -0.25:
        return "BEARISH", "SHORT"
    return "NEUTRAL", "FLAT"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(demo_mode: bool = False) -> dict[str, pd.DataFrame]:
    """
    Main pipeline — iterates both exchanges, scrapes / generates data,
    builds features, runs walk-forward model, writes reports.

    Returns dict[exchange_name → summary_df] for use by the dashboard.
    """
    os.makedirs("reports", exist_ok=True)
    all_summaries: dict[str, pd.DataFrame] = {}
    all_oos: dict[str, dict] = {}

    for exchange, assets in EXCHANGES.items():
        summary_rows = []
        oos_history = {}
        print(f"\n{'='*60}")
        print(f"  Exchange: {exchange}  |  Mode: {'DEMO' if demo_mode else 'LIVE'}")
        print(f"{'='*60}")

        for asset in assets:
            ticker = asset["ticker"]
            name = asset["name"]
            print(f"  [{exchange}] -> {ticker:>12}  {name}")

            try:
                price_df = fetch_price_bars(ticker, days=LOOKBACK_DAYS, interval=BAR_INTERVAL)
            except Exception as exc:
                print(f"    [ERROR] Price fetch failed: {exc}")
                continue

            try:
                if demo_mode:
                    social_df, news_df = generate_demo_social_and_news(ticker, name, exchange, price_df)
                else:
                    social_df = scrape_social_apify(exchange, asset, days=LOOKBACK_DAYS)
                    news_df = fetch_news_exa(asset, days=LOOKBACK_DAYS)
            except Exception as exc:
                print(f"    [ERROR] Data ingestion failed: {exc}")
                social_df = pd.DataFrame()
                news_df = pd.DataFrame()

            aligned_df = align_events_to_bars(price_df, social_df, news_df)
            support, resistance = compute_support_resistance(price_df)
            pred_stats = walk_forward_predict_n_plus_1(aligned_df)

            total_mentions = int(aligned_df["social_mentions"].sum())
            avg_social_sent = float(aligned_df["social_sentiment"].mean())
            avg_news_sent = float(aligned_df["news_sentiment"].mean())
            composite_sent = round(0.7 * avg_social_sent + 0.3 * avg_news_sent, 4)

            direction, position = derive_signal_and_position(
                composite_sent, pred_stats["predicted_return_pct"]
            )

            oos_history[ticker] = pred_stats
            summary_rows.append({
                "exchange": exchange,
                "ticker": ticker,
                "asset_name": name,
                "current_price": pred_stats["current_price"],
                "social_mentions": total_mentions,
                "composite_sentiment": composite_sent,
                "direction": direction,
                "position": position,
                "support": support,
                "resistance": resistance,
                "predicted_price_n_plus_1": pred_stats["predicted_price_n_plus_1"],
                "predicted_return_pct": pred_stats["predicted_return_pct"],
                "oos_mae": pred_stats["oos_mae"],
                "naive_mae": pred_stats["naive_persistence_mae"],
                "directional_accuracy_pct": pred_stats["directional_accuracy_pct"],
            })
            print(
                f"    → Sentiment: {composite_sent:+.4f}  |  Direction: {direction:<8}"
                f"  |  Position: {position:<6}  |  n+1 Price: {pred_stats['predicted_price_n_plus_1']}"
                f"  |  Dir Acc: {pred_stats['directional_accuracy_pct']}%"
            )

        if not summary_rows:
            print(f"  [WARN] No assets processed for {exchange}")
            continue

        summary_df = pd.DataFrame(summary_rows).sort_values(
            by=["social_mentions", "composite_sentiment"], ascending=[False, False]
        ).reset_index(drop=True)

        # Static visual report (PNG)
        chart_path = generate_visual_report(exchange, summary_df, oos_history, output_dir="reports")

        # JSON summary
        json_path = f"reports/{exchange}_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_df.to_dict(orient="records"), f, indent=2)

        print(f"\n  Saved chart  → {chart_path}")
        print(f"  Saved JSON   → {json_path}")
        print(f"\n  {'─'*55}")
        print(f"  {exchange} Ranked Summary")
        print(f"  {'─'*55}")
        print(
            summary_df[
                ["ticker", "asset_name", "social_mentions", "composite_sentiment",
                 "direction", "position", "predicted_price_n_plus_1", "directional_accuracy_pct"]
            ].to_string(index=False)
        )

        all_summaries[exchange] = summary_df
        all_oos[exchange] = oos_history

    return all_summaries, all_oos


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HengSeng Sentiment Price Prediction Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                  Run offline with synthetic data
  python main.py                         Run live (requires .env tokens)
  python main.py --demo --dashboard      Run offline then launch Dash dashboard
  python main.py --dashboard             Run live then launch Dash dashboard
        """,
    )
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data (no API tokens required)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch the interactive Dash dashboard after pipeline")
    args = parser.parse_args()

    summaries, oos_all = run_pipeline(demo_mode=args.demo)

    if args.dashboard:
        from dashboard import launch_dashboard
        launch_dashboard(summaries, oos_all)
