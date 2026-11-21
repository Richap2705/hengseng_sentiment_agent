# HengSeng Sentiment Price Prediction Agent

A production-grade Python pipeline that predicts equity prices at **n+1** by fusing:
- **Social sentiment** (Apify-scraped from ≥2 sources per exchange — LIHKG/Telegram for HKEX, Naver Finance/Telegram for KRX)
- **News intelligence** (Exa.com neural search)
- **Walk-forward OOS validation** (Ridge Regression, no look-ahead)

Covers **two exchanges**: HKEX (Hong Kong) and KRX (South Korea), 5 assets each.

---

## Project Structure

```
hengseng_sentiment_agent/
├── main.py               ← Pipeline entry point
├── dashboard.py          ← Interactive Plotly Dash dashboard
├── src/
│   ├── config.py         ← Exchange assets, API keys, constants
│   ├── scrapers.py       ← Price (yfinance) + Social (Apify) + News (Exa)
│   ├── features.py       ← Sentiment scoring, feature engineering
│   ├── model.py          ← Walk-forward OOS prediction (Ridge)
│   └── report.py         ← Static PNG/CSV/JSON report generator
├── reports/              ← Auto-created output directory
├── requirements.txt
└── .env.example
```

---

## Exchanges & Assets

### HKEX (Hong Kong)
| Ticker   | Name               | Social Sources              |
|----------|--------------------|-----------------------------|
| 0700.HK  | Tencent Holdings   | LIHKG Forum + Telegram_HK   |
| 9988.HK  | Alibaba Group      | LIHKG Forum + Telegram_HK   |
| 3690.HK  | Meituan            | LIHKG Forum + Telegram_HK   |
| 1810.HK  | Xiaomi Corporation | LIHKG Forum + Telegram_HK   |
| 1299.HK  | AIA Group          | LIHKG Forum + Telegram_HK   |

### KRX (South Korea)
| Ticker      | Name               | Social Sources                      |
|-------------|--------------------|------------------------------------- |
| 005930.KS   | Samsung Electronics| Naver Finance Discussion + Telegram_KR|
| 000660.KS   | SK Hynix           | Naver Finance Discussion + Telegram_KR|
| 035420.KS   | NAVER Corp         | Naver Finance Discussion + Telegram_KR|
| 051910.KS   | LG Chem            | Naver Finance Discussion + Telegram_KR|
| 006400.KS   | Samsung SDI        | Naver Finance Discussion + Telegram_KR|

---

## Setup

```bash
# 1. Clone / navigate to project
cd hengseng_sentiment_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env — fill in APIFY_TOKEN, EXA_API_KEY
```

---

## Usage

### Offline Demo (no API keys needed)
```bash
python main.py --demo
```

### Offline Demo + Interactive Dashboard
```bash
python main.py --demo --dashboard
# Then open: http://127.0.0.1:8050/
```

### Live Mode (requires tokens in .env)
```bash
python main.py
```

### Live Mode + Dashboard
```bash
python main.py --dashboard
```

### Standalone Dashboard (from saved reports)
```bash
python dashboard.py
```

---

## Pipeline Steps

1. **Price Ingestion** — `fetch_price_bars()` pulls 10 days × 1h OHLCV bars per ticker via yfinance.
2. **Social Scraping** — `scrape_social_apify()` runs 2 Apify actors per exchange:
   - Actor 1 (`APIFY_ACTOR_SOURCE_1`): Forum scraping (LIHKG / Naver Finance)
   - Actor 2 (`APIFY_ACTOR_SOURCE_2`): Telegram channel scraping
   - Extracts: UTC timestamp, text, likes, reposts, views, follower count
3. **News Enrichment** — `fetch_news_exa()` fetches 15 recent articles per ticker from Exa.com.
4. **Feature Engineering** — `align_events_to_bars()`:
   - Scores text sentiment (VADER + multilingual financial lexicon)
   - Engagement-weighted aggregation, forward-mapped to price bars (no leakage)
   - Computes: return_1, volatility_6, momentum_6, rolling sentiment features
5. **Support / Resistance** — `compute_support_resistance()`: pivot-point method over 24-bar rolling window.
6. **Model** — `walk_forward_predict_n_plus_1()`: expanding-window OOS Ridge regression.
7. **Signal** — Composite sentiment (70% social + 30% news) → BULLISH/BEARISH/NEUTRAL + LONG/SHORT/FLAT.
8. **Reports** — Static PNG + CSV + JSON in `reports/`; interactive Dash dashboard.

---

## Outputs per Exchange

| File | Description |
|------|-------------|
| `reports/{EXCHANGE}_sentiment_report.png` | 4-panel dark-themed chart |
| `reports/{EXCHANGE}_ranked_report.csv`    | Ranked ticker CSV |
| `reports/{EXCHANGE}_summary.json`         | Machine-readable summary |

---

## Submission Checklist (CrowdWisdomTrading)

- [x] Python-based sentiment pipeline predicting n+1 price
- [x] Min. 2 social sources per exchange (LIHKG/Telegram for HKEX; Naver/Telegram for KRX)
- [x] Apify-based social ingestion with UTC timestamps, text, engagement, follower counts
- [x] Exa.com news sentiment enrichment
- [x] 5 assets per exchange (10 total across HKEX + KRX)
- [x] Per-ticker extraction: ticker, asset name, current price, sentiment, direction, position, support/resistance
- [x] Visual report sorted by social mentions / sentiment score / direction
- [x] n+1 price prediction with rigorous OOS walk-forward validation
- [x] Interactive Dash dashboard (advanced, non-generic look)

---

## Technical Notes

- **Sentiment scoring**: Hybrid VADER + financial domain lexicon (English + 中文 + 한국어)
- **Engagement weighting**: `1 + log(likes) + 1.5·log(reposts) + 0.25·log(views) + 0.35·log(followers)`
- **No look-ahead**: social/news events are forward-mapped to the next bar's close
- **OOS validation**: expanding window, `min_train_bars=20`, Ridge α=5.0

---

*Contact: gilad@crowdwisdomtrading.com*
