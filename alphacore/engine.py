import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


STRATEGY_WEIGHTS = {
    "momentum": 0.22,
    "trend_following": 0.18,
    "relative_strength": 0.15,
    "volatility_regime": 0.12,
    "turtle_breakout": 0.10,
    "volume_accumulation": 0.08,
    "statistical_signals": 0.06,
    "mean_reversion": 0.04,
    "market_breadth": 0.03,
    "pattern_recognition": 0.02,
}


def normalize(series):
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).fillna(0)
    min_val = series.rolling(252, min_periods=20).min()
    max_val = series.rolling(252, min_periods=20).max()
    score = 100 * (series - min_val) / (max_val - min_val)
    return score.clip(0, 100).fillna(50)


def fetch_data(ticker, period="3y"):
    stock = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if stock.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)

    stock = stock[["Open", "High", "Low", "Close", "Volume"]].dropna()

    spy = yf.download("SPY", period=period, auto_adjust=True, progress=False)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy = spy[["Close"]].rename(columns={"Close": "SPY_Close"})

    return stock.join(spy, how="inner").dropna()


def detect_market_regime(row):
    if row["market_breadth"] > 65 and row["trend_following"] > 60:
        return "Bull Trend"

    if row["market_breadth"] < 35 and row["trend_following"] < 40:
        return "Bear Trend"

    if row["volatility_regime"] < 35:
        return "High Volatility"

    return "Sideways / Mixed"


def regime_score(regime):
    return {
        "Bull Trend": 1.0,
        "Sideways / Mixed": 0.5,
        "High Volatility": 0.25,
        "Bear Trend": 0.0,
    }.get(regime, 0.5)


def build_time_decay_weights(length):
    return np.linspace(0.25, 1.0, length)


def build_strategy_scores(data):
    df = data.copy()

    close = df["Close"].squeeze()
    open_ = df["Open"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    spy_close = df["SPY_Close"].squeeze()

    df["return_5"] = close.pct_change(5)
    df["return_10"] = close.pct_change(10)
    df["return_20"] = close.pct_change(20)

    df["spy_return_20"] = spy_close.pct_change(20)
    df["relative_strength_raw"] = df["return_20"] - df["spy_return_20"]

    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    df["rsi"] = RSIIndicator(close=close, window=14).rsi()

    adx = ADXIndicator(high=high, low=low, close=close, window=14)
    df["adx"] = adx.adx()

    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / close

    macd = MACD(close=close)
    df["macd_diff"] = macd.macd_diff()

    bb = BollingerBands(close=close)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_position"] = (close - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    df["volume_ratio"] = volume / volume.rolling(20).mean()

    df["breakout_20"] = close / close.rolling(20).max()
    df["breakout_55"] = close / close.rolling(55).max()

    df["zscore_20"] = (
        close - close.rolling(20).mean()
    ) / close.rolling(20).std()

    df["green_day"] = (close > open_).astype(int)
    df["pattern_strength"] = df["green_day"].rolling(5).mean()

    df["spy_trend"] = spy_close / spy_close.rolling(50).mean()

    df["momentum"] = normalize(df["return_20"])

    df["trend_following"] = normalize(
        ((close > df["sma_20"]).astype(int) * 25)
        + ((close > df["sma_50"]).astype(int) * 25)
        + ((close > df["sma_200"]).astype(int) * 25)
        + df["adx"].clip(0, 25)
    )

    df["relative_strength"] = normalize(df["relative_strength_raw"])
    df["volatility_regime"] = 100 - normalize(df["atr_pct"])
    df["turtle_breakout"] = normalize((df["breakout_20"] + df["breakout_55"]) / 2)
    df["volume_accumulation"] = normalize(df["volume_ratio"] * df["return_5"])
    df["statistical_signals"] = 100 - normalize(abs(df["zscore_20"]))
    df["mean_reversion"] = normalize((30 - df["rsi"]).clip(lower=0))
    df["market_breadth"] = normalize(df["spy_trend"])
    df["pattern_recognition"] = normalize(df["pattern_strength"])

    return df


def calculate_strategy_performance(df, strategy_cols, target_col="target_14"):
    performance = {}

    for strategy in strategy_cols:
        valid = df[[strategy, target_col]].dropna()

        if len(valid) < 50:
            performance[strategy] = {
                "hit_rate": 50.0,
                "avg_return_when_strong": 0.0,
                "sample_size": len(valid),
            }
            continue

        threshold = valid[strategy].quantile(0.65)
        strong_signals = valid[valid[strategy] >= threshold]

        if strong_signals.empty:
            hit_rate = 50.0
            avg_return = 0.0
        else:
            hit_rate = (strong_signals[target_col] > 0).mean() * 100
            avg_return = strong_signals[target_col].mean() * 100

        performance[strategy] = {
            "hit_rate": hit_rate,
            "avg_return_when_strong": avg_return,
            "sample_size": len(strong_signals),
        }

    return performance


def build_adaptive_weights(strategy_performance):
    raw_scores = {}

    for strategy, stats in strategy_performance.items():
        hit_component = max(stats["hit_rate"] - 50, 0)
        return_component = max(stats["avg_return_when_strong"], 0)
        seed_component = STRATEGY_WEIGHTS.get(strategy, 0.01) * 100

        raw_scores[strategy] = (
            hit_component * 0.50
            + return_component * 0.30
            + seed_component * 0.20
        )

    total = sum(raw_scores.values())

    if total <= 0:
        return STRATEGY_WEIGHTS.copy()

    return {
        strategy: score / total
        for strategy, score in raw_scores.items()
    }


def calculate_weighted_score(df, weights, output_col):
    df[output_col] = sum(
        df[strategy] * weight
        for strategy, weight in weights.items()
    )
    return df


def find_similar_setups(df, strategy_cols, top_n=25):
    if len(df) < top_n + 30:
        return {
            "similar_count": 0,
            "avg_14_day_return": 0.0,
            "success_rate": 0.0,
        }

    historical = df.iloc[:-14].copy()
    latest = df.iloc[[-1]][strategy_cols]

    historical_features = historical[strategy_cols]

    similarities = cosine_similarity(
        historical_features,
        latest
    ).flatten()

    historical["similarity"] = similarities

    top_matches = historical.sort_values(
        "similarity",
        ascending=False
    ).head(top_n)

    valid = top_matches[["target_14", "similarity"]].dropna()

    if valid.empty:
        return {
            "similar_count": 0,
            "avg_14_day_return": 0.0,
            "success_rate": 0.0,
        }

    return {
        "similar_count": len(valid),
        "avg_14_day_return": valid["target_14"].mean() * 100,
        "success_rate": (valid["target_14"] > 0).mean() * 100,
        "avg_similarity": valid["similarity"].mean() * 100,
    }


def build_meta_dataset(data):
    df = build_strategy_scores(data)
    strategy_cols = list(STRATEGY_WEIGHTS.keys())

    df["target_5"] = df["Close"].shift(-5) / df["Close"] - 1
    df["target_10"] = df["Close"].shift(-10) / df["Close"] - 1
    df["target_14"] = df["Close"].shift(-14) / df["Close"] - 1

    df["regime"] = df.apply(detect_market_regime, axis=1)
    df["regime_score"] = df["regime"].apply(regime_score)

    df = calculate_weighted_score(
        df,
        STRATEGY_WEIGHTS,
        "seed_weighted_score"
    )

    clean_df = df.dropna().copy()

    strategy_performance = calculate_strategy_performance(
        clean_df,
        strategy_cols,
        target_col="target_14"
    )

    adaptive_weights = build_adaptive_weights(strategy_performance)

    clean_df = calculate_weighted_score(
        clean_df,
        adaptive_weights,
        "adaptive_weighted_score"
    )

    similar_setup = find_similar_setups(clean_df, strategy_cols)

    clean_df["similar_setup_return"] = similar_setup["avg_14_day_return"]
    clean_df["similar_setup_success_rate"] = similar_setup["success_rate"]

    meta_features = strategy_cols + [
        "seed_weighted_score",
        "adaptive_weighted_score",
        "regime_score",
        "similar_setup_return",
        "similar_setup_success_rate",
    ]

    return (
        clean_df,
        strategy_cols,
        meta_features,
        strategy_performance,
        adaptive_weights,
        similar_setup,
    )


def train_meta_neural_network(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("meta_neural", MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        ))
    ])

    sample_weights = build_time_decay_weights(len(X_train))

    try:
        model.fit(
            X_train,
            y_train,
            meta_neural__sample_weight=sample_weights
        )
        weighting_mode = "Meta neural network with adaptive weights and time-decay training"
    except TypeError:
        model.fit(X_train, y_train)
        weighting_mode = "Meta neural network with adaptive weights"

    return model, weighting_mode


def calculate_backtest_metrics(backtest_df):
    metrics = {}

    for horizon in [5, 10, 14]:
        pred_col = f"predicted_{horizon}"
        actual_col = f"actual_{horizon}"

        valid = backtest_df[[pred_col, actual_col]].dropna()

        if valid.empty:
            continue

        metrics[f"{horizon}_day"] = {
            "direction_accuracy": (
                np.sign(valid[pred_col]) == np.sign(valid[actual_col])
            ).mean() * 100,
            "mae": mean_absolute_error(valid[actual_col], valid[pred_col]),
            "avg_predicted_return": valid[pred_col].mean() * 100,
            "avg_actual_return": valid[actual_col].mean() * 100,
            "bullish_hit_rate": (
                valid[valid[pred_col] > 0][actual_col] > 0
            ).mean() * 100 if not valid[valid[pred_col] > 0].empty else None,
            "bearish_hit_rate": (
                valid[valid[pred_col] < 0][actual_col] < 0
            ).mean() * 100 if not valid[valid[pred_col] < 0].empty else None,
            "samples": len(valid),
        }

    return metrics


def run_backtest(ticker, period="3y", min_train_size=260, step_size=10):
    data = fetch_data(ticker, period=period)

    (
        df,
        strategy_cols,
        meta_features,
        strategy_performance,
        adaptive_weights,
        similar_setup,
    ) = build_meta_dataset(data)

    if len(df) < min_train_size + 50:
        raise ValueError("Not enough historical data to run backtest.")

    rows = []

    for i in range(min_train_size, len(df) - 14, step_size):
        train_df = df.iloc[:i]
        test_row = df.iloc[[i]]

        X_train = train_df[meta_features]
        y_train = train_df[["target_5", "target_10", "target_14"]]

        model, _ = train_meta_neural_network(X_train, y_train)

        prediction = model.predict(test_row[meta_features])[0]

        rows.append({
            "date": df.index[i],
            "close": float(test_row["Close"].iloc[0]),
            "regime": test_row["regime"].iloc[0],
            "predicted_5": prediction[0],
            "predicted_10": prediction[1],
            "predicted_14": prediction[2],
            "actual_5": float(test_row["target_5"].iloc[0]),
            "actual_10": float(test_row["target_10"].iloc[0]),
            "actual_14": float(test_row["target_14"].iloc[0]),
        })

    backtest_df = pd.DataFrame(rows)

    return {
        "ticker": ticker.upper(),
        "backtest": backtest_df,
        "metrics": calculate_backtest_metrics(backtest_df),
        "samples": len(backtest_df),
        "period": period,
        "min_train_size": min_train_size,
        "step_size": step_size,
    }


def train_and_predict(ticker):
    data = fetch_data(ticker)

    (
        df,
        strategy_cols,
        meta_features,
        strategy_performance,
        adaptive_weights,
        similar_setup,
    ) = build_meta_dataset(data)

    if len(df) < 250:
        raise ValueError("Not enough historical data to train the meta neural network.")

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[meta_features]
    y_train = train_df[["target_5", "target_10", "target_14"]]

    X_test = test_df[meta_features]
    y_test = test_df[["target_5", "target_10", "target_14"]]

    model, weighting_mode = train_meta_neural_network(X_train, y_train)

    preds_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_test)

    latest_features = df[meta_features].tail(1)
    prediction = model.predict(latest_features)[0]

    latest_close = float(df["Close"].iloc[-1])

    latest_scores = df[strategy_cols].tail(1).T
    latest_scores.columns = ["score"]

    current_regime = df["regime"].iloc[-1]

    forecast = pd.DataFrame({
        "Horizon": ["5 Trading Days", "10 Trading Days", "14 Trading Days"],
        "Expected Return %": prediction * 100,
        "Predicted Price": latest_close * (1 + prediction),
    })

    confidence = max(0, min(100, 100 - mae * 1000))

    top_signal = latest_scores["score"].idxmax()
    weakest_signal = latest_scores["score"].idxmin()

    if prediction[2] > 0.02:
        bias = "Bullish"
    elif prediction[2] < -0.02:
        bias = "Bearish"
    else:
        bias = "Neutral"

    adaptive_weights_df = pd.DataFrame({
        "strategy": list(adaptive_weights.keys()),
        "adaptive_weight": [v * 100 for v in adaptive_weights.values()],
        "seed_weight": [
            STRATEGY_WEIGHTS.get(k, 0) * 100 for k in adaptive_weights.keys()
        ],
    })

    strategy_performance_df = pd.DataFrame(strategy_performance).T.reset_index()
    strategy_performance_df = strategy_performance_df.rename(
        columns={"index": "strategy"}
    )

    return {
        "ticker": ticker.upper(),
        "latest_price": latest_close,
        "forecast": forecast,
        "scores": latest_scores,
        "confidence": confidence,
        "top_signal": top_signal,
        "weakest_signal": weakest_signal,
        "bias": bias,
        "model_mae": mae,
        "weighting_mode": weighting_mode,
        "current_regime": current_regime,
        "meta_features": meta_features,
        "adaptive_weights": adaptive_weights_df,
        "strategy_performance": strategy_performance_df,
        "similar_setup": similar_setup,
    }
