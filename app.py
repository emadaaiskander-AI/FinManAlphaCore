import streamlit as st
import pandas as pd

from alphacore.engine import train_and_predict, run_backtest, STRATEGY_WEIGHTS


st.set_page_config(
    page_title="FinMan AlphaCore",
    layout="wide"
)


st.title("FinMan AlphaCore")
st.subheader("Target Architecture: Adaptive Meta Neural Technical Forecast Engine")

st.markdown(
    """
FinMan AlphaCore evaluates technical strategy engines, calculates per-stock historical strategy performance, 
builds adaptive weights, compares the current setup against similar historical setups, and feeds the results into a meta neural network.
"""
)

ticker = st.text_input("Enter stock ticker", value="AAPL").upper().strip()

col_a, col_b = st.columns(2)

with col_a:
    run_prediction = st.button("Run AlphaCore Prediction")

with col_b:
    run_backtest_button = st.button("Run Prediction + Backtest")


def display_prediction(result):
    st.success(f"Prediction completed for {result['ticker']}")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Latest Price", f"${result['latest_price']:.2f}")
    c2.metric("Bias", result["bias"])
    c3.metric("Confidence", f"{result['confidence']:.1f}%")
    c4.metric("Model MAE", f"{result['model_mae']:.4f}")
    c5.metric("Regime", result["current_regime"])

    st.divider()

    st.subheader("Forecast")

    forecast = result["forecast"].copy()
    forecast["Expected Return %"] = forecast["Expected Return %"].map(
        lambda x: f"{x:.2f}%"
    )
    forecast["Predicted Price"] = forecast["Predicted Price"].map(
        lambda x: f"${x:.2f}"
    )

    st.dataframe(forecast, use_container_width=True)

    st.divider()

    st.subheader("Similar Historical Setup")

    similar = result["similar_setup"]

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("Similar Matches", similar.get("similar_count", 0))
    s2.metric("Avg 14D Return", f"{similar.get('avg_14_day_return', 0):.2f}%")
    s3.metric("Success Rate", f"{similar.get('success_rate', 0):.1f}%")
    s4.metric("Avg Similarity", f"{similar.get('avg_similarity', 0):.1f}%")

    st.divider()

    st.subheader("Strategy Scores")

    scores = result["scores"].copy()
    scores["Seed Weight %"] = scores.index.map(
        lambda x: STRATEGY_WEIGHTS.get(x, 0) * 100
    )

    scores = scores.reset_index().rename(
        columns={
            "index": "Strategy",
            "score": "Score",
        }
    )

    scores["Score"] = scores["Score"].map(lambda x: round(float(x), 2))
    scores["Seed Weight %"] = scores["Seed Weight %"].map(lambda x: round(float(x), 2))

    st.dataframe(scores, use_container_width=True)

    st.divider()

    st.subheader("Adaptive Weights")

    adaptive_weights = result["adaptive_weights"].copy()
    adaptive_weights["adaptive_weight"] = adaptive_weights["adaptive_weight"].map(
        lambda x: round(float(x), 2)
    )
    adaptive_weights["seed_weight"] = adaptive_weights["seed_weight"].map(
        lambda x: round(float(x), 2)
    )

    st.dataframe(adaptive_weights, use_container_width=True)

    st.divider()

    st.subheader("Strategy Historical Performance")

    strategy_performance = result["strategy_performance"].copy()

    for col in ["hit_rate", "avg_return_when_strong"]:
        if col in strategy_performance.columns:
            strategy_performance[col] = strategy_performance[col].map(
                lambda x: round(float(x), 2)
            )

    st.dataframe(strategy_performance, use_container_width=True)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Signal Attribution")
        st.write(f"**Top Supporting Signal:** {result['top_signal']}")
        st.write(f"**Weakest Signal:** {result['weakest_signal']}")
        st.write(f"**Training Mode:** {result['weighting_mode']}")

    with right:
        st.subheader("Meta Neural Inputs")
        for feature in result["meta_features"]:
            st.write(f"- {feature}")

    st.divider()

    st.warning(
        "Disclaimer: FinMan AlphaCore is an experimental technical-analysis model. "
        "It is not financial advice and should not be used as the sole basis for trading or investment decisions."
    )


def display_backtest(result):
    st.subheader("Backtest Results")

    st.write(f"**Ticker:** {result['ticker']}")
    st.write(f"**Period:** {result['period']}")
    st.write(f"**Samples:** {result['samples']}")
    st.write(f"**Step Size:** {result['step_size']}")

    metrics_rows = []

    for horizon, metrics in result["metrics"].items():
        metrics_rows.append({
            "Horizon": horizon.replace("_", " ").title(),
            "Samples": metrics["samples"],
            "Direction Accuracy %": round(metrics["direction_accuracy"], 2),
            "MAE": round(metrics["mae"], 4),
            "Avg Predicted Return %": round(metrics["avg_predicted_return"], 2),
            "Avg Actual Return %": round(metrics["avg_actual_return"], 2),
            "Bullish Hit Rate %": (
                None if metrics["bullish_hit_rate"] is None
                else round(metrics["bullish_hit_rate"], 2)
            ),
            "Bearish Hit Rate %": (
                None if metrics["bearish_hit_rate"] is None
                else round(metrics["bearish_hit_rate"], 2)
            ),
        })

    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

    st.subheader("Recent Backtest Rows")
    st.dataframe(result["backtest"].tail(25), use_container_width=True)


if run_prediction or run_backtest_button:
    if not ticker:
        st.error("Please enter a valid ticker.")
    else:
        with st.spinner(
            "Running AlphaCore target architecture: Yahoo data, strategy scores, adaptive weights, similar setup engine, meta neural prediction..."
        ):
            try:
                prediction_result = train_and_predict(ticker)
                display_prediction(prediction_result)

                if run_backtest_button:
                    with st.spinner("Running historical backtest..."):
                        backtest_result = run_backtest(
                            ticker=ticker,
                            period="3y",
                            step_size=20,
                        )
                        display_backtest(backtest_result)

            except Exception as e:
                st.error(f"Error: {e}")
