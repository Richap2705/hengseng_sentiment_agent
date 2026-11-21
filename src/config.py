import os
from dotenv import load_dotenv

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
APIFY_ACTOR_SOURCE_1 = os.getenv("APIFY_ACTOR_SOURCE_1", "apify/web-scraper")
APIFY_ACTOR_SOURCE_2 = os.getenv("APIFY_ACTOR_SOURCE_2", "apify/telegram-scraper")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

# ──────────────────────────────────────────────
# EXCHANGE 1 — HKEX (Hang Seng / Hong Kong)
#   Social sources used: LIHKG Forum + Telegram
# ──────────────────────────────────────────────
EXCHANGES = {
    "HKEX": [
        {
            "ticker": "0700.HK",
            "name": "Tencent Holdings",
            "keywords": ["0700.HK", "Tencent", "騰訊", "$0700"],
            "hashtags": ["#Tencent", "#0700HK", "#騰訊"],
        },
        {
            "ticker": "9988.HK",
            "name": "Alibaba Group",
            "keywords": ["9988.HK", "Alibaba", "阿里巴巴", "$9988"],
            "hashtags": ["#Alibaba", "#9988HK", "#阿里巴巴"],
        },
        {
            "ticker": "3690.HK",
            "name": "Meituan",
            "keywords": ["3690.HK", "Meituan", "美團", "$3690"],
            "hashtags": ["#Meituan", "#3690HK", "#美團"],
        },
        {
            "ticker": "1810.HK",
            "name": "Xiaomi Corporation",
            "keywords": ["1810.HK", "Xiaomi", "小米", "$1810"],
            "hashtags": ["#Xiaomi", "#1810HK", "#小米"],
        },
        {
            "ticker": "1299.HK",
            "name": "AIA Group",
            "keywords": ["1299.HK", "AIA", "友邦保險", "$1299"],
            "hashtags": ["#AIA", "#1299HK", "#友邦"],
        },
    ],
    # ──────────────────────────────────────────────
    # EXCHANGE 2 — KRX (Korea Stock Exchange)
    #   Social sources used: Naver Finance Discussion + KakaoTalk/Telegram
    # ──────────────────────────────────────────────
    "KRX": [
        {
            "ticker": "005930.KS",
            "name": "Samsung Electronics",
            "keywords": ["005930", "Samsung", "삼성전자", "삼전"],
            "hashtags": ["#삼성전자", "#005930", "#Samsung"],
        },
        {
            "ticker": "000660.KS",
            "name": "SK Hynix",
            "keywords": ["000660", "SK하이닉스", "하이닉스", "SKHynix"],
            "hashtags": ["#하이닉스", "#000660", "#SKHynix"],
        },
        {
            "ticker": "035420.KS",
            "name": "NAVER Corp",
            "keywords": ["035420", "NAVER", "네이버", "NVR"],
            "hashtags": ["#네이버", "#035420", "#NAVER"],
        },
        {
            "ticker": "051910.KS",
            "name": "LG Chem",
            "keywords": ["051910", "LG화학", "LGChem", "엘지화학"],
            "hashtags": ["#LG화학", "#051910", "#LGChem"],
        },
        {
            "ticker": "006400.KS",
            "name": "Samsung SDI",
            "keywords": ["006400", "삼성SDI", "SamsungSDI", "SDI"],
            "hashtags": ["#삼성SDI", "#006400", "#SamsungSDI"],
        },
    ],
}

LOOKBACK_DAYS = 10
BAR_INTERVAL = "1h"  # Hourly bars over 10 days → ~100 bars per ticker for OOS validation
