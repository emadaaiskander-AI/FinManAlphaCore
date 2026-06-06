import streamlit as st
import pandas as pd
from alphacore import train_and_predict, STRATEGY_WEIGHTS

st.set_page_config(
    page_title="FinMan AlphaCore",
    layout="wide"
)

st.title("FinMan AlphaCore")
st.subheader("Multi-Strategy Adaptive Technical Forecast Engine")

ticker = st.text_input("Enter stock ticker", value="AAPL").upper()

if st.button("Run AlphaCore Prediction"):
    with st.spinner("Fetching market data, calculating signals, and training neural network..."):
        try:
            result = train_and_predict(ticker)

            st.success(f"Prediction completed for {result['ticker']}")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Latest Price", f"${result['latest_price']:.2f}")
            col2.metric("Bias", result["bias"])
            col3.metric("Confidence", f"{result['confidence']:.1f}%")
            col4.metric("Model MAE", f"{result['model_mae']:.4f}")

            st.divider()

            st.subheader("Forecast")
            forecast = result["forecast"].copy()
            forecast["Expected Return %"] = forecast["Expected Return %"].map(lambda x: f"{x:.2f}%")
            forecast["Predicted Price"] = forecast["Predicted Price"].map(lambda x: f"${x:.2f}")
            st.dataframe(forecast, use_container_width=True)

            st.subheader("Strategy Scores")
            scores = result["scores"].copy()
            scores["weight"] = scores.index.map(lambda x: STRATEGY_WEIGHTS[x] * 100)
            scores = scores.reset_index().rename(columns={"index": "Strategy", "score": "Score", "weight": "Seed Weight %"})
            scores["Score"] = scores["Score"].map(lambda x: round(x, 2))
            scores["Seed Weight %"] = scores["Seed Weight %"].map(lambda x: round(x, 2))
            st.dataframe(scores, use_container_width=True)

            st.subheader("Signal Attribution")
            st.write(f"**Top Supporting Signal:** {result['top_signal']}")
            st.write(f"**Weakest Signal:** {result['weakest_signal']}")

            st.caption(
                "Disclaimer: This is an experimental technical-analysis model. "
                "It is not financial advice and should not be used as the sole basis for trading decisions."
            )

        except Exception as e:
            st.error(f"Error: {e}")
