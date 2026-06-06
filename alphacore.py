import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
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
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0)
    min_val = series.rolling(252, min_periods=20).min()
    max_val = series.rolling(252, min_periods=20).max()
    score = 100 * (series - min_val) / (max_val - min_val)
    return score.clip(0, 100).fillna(50)


def fetch_data(ticker, period="3y"):
    stock = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if stock.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    stock = stock[["Open", "High", "Low", "Close", "Volume"]].dropna()

    spy = yf.download("SPY", period=period, auto_adjust=True, progress=False)
    spy = spy[["Close"]].rename(columns={"Close": "SPY_Close"})

    data = stock.join(spy, how="inner")
    return data.dropna()


def build_features(data):
    df = data.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["return_5"] = close.pct_change(5)
    df["return_10"] = close.pct_change(10)
    df["return_20"] = close.pct_change(20)

    df["spy_return_20"] = df["SPY_Close"].pct_change(20)
    df["relative_strength_raw"] = df["return_20"] - df["spy_return_20"]

    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    df["rsi"] = RSIIndicator(close).rsi()

    adx = ADXIndicator(high, low, close)
    df["adx"] = adx.adx()

    atr = AverageTrueRange(high, low, close)
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / close

    macd = MACD(close)
    df["macd_diff"] = macd.macd_diff()

    bb = BollingerBands(close)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_position"] = (close - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    df["volume_ratio"] = volume / volume.rolling(20).mean()

    df["breakout_20"] = close / close.rolling(20).max()
    df["breakout_55"] = close / close.rolling(55).max()

    df["zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()

    df["green_day"] = (close > df["Open"]).astype(int)
    df["pattern_strength"] = df["green_day"].rolling(5).mean()

    df["spy_trend"] = df["SPY_Close"] / df["SPY_Close"].rolling(50).mean()

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

    strategy_cols = list(STRATEGY_WEIGHTS.keys())

    df["weighted_score"] = sum(df[col] * weight for col, weight in STRATEGY_WEIGHTS.items())

    df["target_5"] = close.shift(-5) / close - 1
    df["target_10"] = close.shift(-10) / close - 1
    df["target_14"] = close.shift(-14) / close - 1

    df = df.dropna()

    return df, strategy_cols


def train_and_predict(ticker):
    data = fetch_data(ticker)
    df, strategy_cols = build_features(data)

    X = df[strategy_cols + ["weighted_score"]]
    y = df[["target_5", "target_10", "target_14"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("nn", MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            max_iter=1500,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_test)

    latest_features = X.tail(1)
    prediction = model.predict(latest_features)[0]

    latest_close = df["Close"].iloc[-1]
    latest_scores = df[strategy_cols].tail(1).T
    latest_scores.columns = ["score"]

    forecast = pd.DataFrame({
        "Horizon": ["5 Trading Days", "10 Trading Days", "14 Trading Days"],
        "Expected Return %": prediction * 100,
        "Predicted Price": latest_close * (1 + prediction)
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
    }
