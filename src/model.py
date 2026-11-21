"""
model.py
--------
Walk-forward (expanding-window) out-of-sample validation for n+1 price prediction.

Model: Ridge Regression with StandardScaler
Features: price momentum / volatility + social sentiment signals + news sentiment
Validation: purely OOS — at bar t, only bars [0..t-1] are used for training.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLS = [
    "return_1",             # 1-bar log return
    "volatility_6",         # 6-bar rolling std of returns
    "momentum_6",           # 6-bar price momentum
    "social_mentions",      # raw mention count per bar
    "social_sentiment",     # engagement-weighted sentiment
    "rolling_social_sent_3",# 3-bar rolling avg social sentiment
    "total_engagement",     # sum of engagement weights
    "news_count",           # news articles per bar
    "news_sentiment",       # Exa news sentiment score
    "rolling_news_sent_3",  # 3-bar rolling avg news sentiment
]


def walk_forward_predict_n_plus_1(aligned_df: pd.DataFrame, min_train_bars: int = 20) -> dict:
    """
    Rigorous expanding-window walk-forward validation.

    At each step t (t >= min_train_bars):
      - Train on bars [0 .. t-1]
      - Predict return at bar t (i.e., Close[t+1] / Close[t] - 1)
      - Compare OOS prediction to actual

    Final n+1 forecast:
      - Train on all bars [0 .. n-1]
      - Predict Close[n+1] from bar n features

    Returns dict with:
      current_price, predicted_price_n_plus_1, predicted_return_pct,
      oos_mae, oos_rmse, naive_persistence_mae,
      directional_accuracy_pct,
      oos_timestamps, oos_actuals, oos_preds
    """
    df = aligned_df.copy().reset_index(drop=True)
    df["target_return_next"] = df["Close"].shift(-1) / df["Close"] - 1.0

    trainable = df.iloc[:-1].copy()  # all bars except the last (no next close for last bar)
    if len(trainable) <= min_train_bars:
        min_train_bars = max(5, len(trainable) // 2)

    oos_preds: list[float] = []
    oos_actuals: list[float] = []
    oos_prev_close: list[float] = []
    oos_timestamps: list = []

    for t in range(min_train_bars, len(trainable)):
        train_slice = trainable.iloc[:t]
        test_row = trainable.iloc[[t]]

        X_train = train_slice[FEATURE_COLS].values
        y_train = train_slice["target_return_next"].values
        X_test = test_row[FEATURE_COLS].values

        # Ridge regression — robust to multicollinearity across correlated sentiment features
        pipeline = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        pipeline.fit(X_train, y_train)
        pred_ret = float(pipeline.predict(X_test)[0])

        prev_close = float(test_row["Close"].iloc[0])
        actual_next_close = float(df.loc[t + 1, "Close"])
        pred_next_close = prev_close * (1.0 + pred_ret)

        oos_preds.append(pred_next_close)
        oos_actuals.append(actual_next_close)
        oos_prev_close.append(prev_close)
        oos_timestamps.append(df.loc[t + 1, "timestamp_utc"])

    arr_preds = np.array(oos_preds)
    arr_actuals = np.array(oos_actuals)
    arr_prev = np.array(oos_prev_close)

    mae = float(mean_absolute_error(arr_actuals, arr_preds)) if len(arr_actuals) > 0 else float("nan")
    rmse = float(np.sqrt(mean_squared_error(arr_actuals, arr_preds))) if len(arr_actuals) > 0 else float("nan")
    naive_mae = float(mean_absolute_error(arr_actuals, arr_prev)) if len(arr_actuals) > 0 else float("nan")

    actual_dir = np.sign(arr_actuals - arr_prev)
    pred_dir = np.sign(arr_preds - arr_prev)
    directional_accuracy = float(np.mean(actual_dir == pred_dir)) if len(actual_dir) > 0 else 0.0

    # ── Final n+1 forecast using all available completed bars ─────────────────
    final_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    final_pipeline.fit(trainable[FEATURE_COLS].values, trainable["target_return_next"].values)

    latest_row = df.iloc[[-1]]
    current_price = float(latest_row["Close"].iloc[0])
    n_plus_1_return = float(final_pipeline.predict(latest_row[FEATURE_COLS].values)[0])
    n_plus_1_price = round(current_price * (1.0 + n_plus_1_return), 2)

    return {
        "current_price": round(current_price, 2),
        "predicted_price_n_plus_1": n_plus_1_price,
        "predicted_return_pct": round(n_plus_1_return * 100.0, 3),
        "oos_mae": round(mae, 4),
        "oos_rmse": round(rmse, 4),
        "naive_persistence_mae": round(naive_mae, 4),
        "directional_accuracy_pct": round(directional_accuracy * 100.0, 2),
        "oos_timestamps": oos_timestamps,
        "oos_actuals": arr_actuals.tolist(),
        "oos_preds": arr_preds.tolist(),
    }
