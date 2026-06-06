import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

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

    data = stock.join(spy, how="inner")
    return data.dropna()


def detect_market_regime(row):
    if row["market_breadth"] > 65 and row["trend_following"] > 60:
        return "Bull Trend"

    if row["market_breadth"] < 35 and row["trend_following"] < 40:
        return "Bear Trend"

    if row["volatility_regime"] < 35:
        return "High Volatility"

    return "Sideways / Mixed"


def regime_score(regime):
    mapping = {
        "Bull Trend": 1.0,
        "Sideways / Mixed": 0.5,
        "High Volatility": 0.25,
        "Bear Trend": 0.0,
    }
    return mapping.get(regime, 0.5)


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


def build_meta_dataset(data):
    df = build_strategy_scores(data)
    strategy_cols = list(STRATEGY_WEIGHTS.keys())

    df["seed_weighted_score"] = sum(
        df[col] * weight for col, weight in STRATEGY_WEIGHTS.items()
    )

    df["regime"] = df.apply(detect_market_regime, axis=1)
    df["regime_score"] = df["regime"].apply(regime_score)

    df["target_5"] = df["Close"].shift(-5) / df["Close"] - 1
    df["target_10"] = df["Close"].shift(-10) / df["Close"] - 1
    df["target_14"] = df["Close"].shift(-14) / df["Close"] - 1

    meta_features = strategy_cols + [
        "seed_weighted_score",
        "regime_score",
    ]

    df = df.dropna()

    return df, strategy_cols, meta_features


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
        weighting_mode = "Meta neural network with time-decay training"
    except TypeError:
        model.fit(X_train, y_train)
        weighting_mode = "Meta neural network standard training"

    return model, weighting_mode


def calculate_backtest_metrics(backtest_df):
    metrics = {}

    for horizon in [5, 10, 14]:
        pred_col = f"predicted_{horizon}"
        actual_col = f"actual_{horizon}"

        valid = backtest_df[[pred_col, actual_col]].dropna()

        if valid.empty:
            continue

        direction_correct = (
            np.sign(valid[pred_col]) == np.sign(valid[actual_col])
        ).mean()

        mae = mean_absolute_error(valid[actual_col], valid[pred_col])

        avg_predicted = valid[pred_col].mean()
        avg_actual = valid[actual_col].mean()

        bullish_trades = valid[valid[pred_col] > 0]
        bearish_trades = valid[valid[pred_col] < 0]

        bullish_hit_rate = None
        bearish_hit_rate = None

        if not bullish_trades.empty:
            bullish_hit_rate = (bullish_trades[actual_col] > 0).mean()

        if not bearish_trades.empty:
            bearish_hit_rate = (bearish_trades[actual_col] < 0).mean()

        metrics[f"{horizon}_day"] = {
            "direction_accuracy": direction_correct * 100,
            "mae": mae,
            "avg_predicted_return": avg_predicted * 100,
            "avg_actual_return": avg_actual * 100,
            "bullish_hit_rate": None if bullish_hit_rate is None else bullish_hit_rate * 100,
            "bearish_hit_rate": None if bearish_hit_rate is None else bearish_hit_rate * 100,
            "samples": len(valid),
        }

    return metrics


def run_backtest(ticker, period="3y", min_train_size=260, step_size=10):
    data = fetch_data(ticker, period=period)
    df, strategy_cols, meta_features = build_meta_dataset(data)

    if len(df) < min_train_size + 50:
        raise ValueError("Not enough historical data to run backtest.")

    rows = []

    for i in range(min_train_size, len(df) - 14, step_size):
        train_df = df.iloc[:i]
        test_row = df.iloc[[i]]

        X_train = train_df[meta_features]
        y_train = train_df[["target_5", "target_10", "target_14"]]

        model, weighting_mode = train_meta_neural_network(X_train, y_train)

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
    metrics = calculate_backtest_metrics(backtest_df)

    return {
        "ticker": ticker.upper(),
        "backtest": backtest_df,
        "metrics": metrics,
        "samples": len(backtest_df),
        "period": period,
        "min_train_size": min_train_size,
        "step_size": step_size,
    }


def train_and_predict(ticker):
    data = fetch_data(ticker)
    df, strategy_cols, meta_features = build_meta_dataset(data)

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
    }
