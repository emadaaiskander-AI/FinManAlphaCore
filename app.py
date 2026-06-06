import streamlit as st
import pandas as pd

from alphacore.engine import train_and_predict, STRATEGY_WEIGHTS


st.set_page_config(
    page_title="FinMan AlphaCore",
    layout="wide"
)


st.title("FinMan AlphaCore")
st.subheader("Meta Neural Technical Forecast Engine")

st.markdown(
    """
FinMan AlphaCore evaluates multiple technical trading strategy engines, converts them into normalized strategy scores, 
then uses a meta neural network to forecast short-term price movement.
"""
)

ticker = st.text_input("Enter stock ticker", value="AAPL").upper().strip()

run_button = st.button("Run AlphaCore Prediction")


if run_button:
    if not ticker:
        st.error("Please enter a valid ticker.")
    else:
        with st.spinner(
            "Fetching Yahoo Finance data, calculating strategy scores, training meta neural network, and generating forecast..."
        ):
            try:
                result = train_and_predict(ticker)

                st.success(f"Prediction completed for {result['ticker']}")

                col1, col2, col3, col4, col5 = st.columns(5)

                col1.metric("Latest Price", f"${result['latest_price']:.2f}")
                col2.metric("Bias", result["bias"])
                col3.metric("Confidence", f"{result['confidence']:.1f}%")
                col4.metric("Model MAE", f"{result['model_mae']:.4f}")
                col5.metric("Regime", result["current_regime"])

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
                scores["Seed Weight %"] = scores["Seed Weight %"].map(
                    lambda x: round(float(x), 2)
                )

                st.dataframe(scores, use_container_width=True)

                st.divider()

                col_a, col_b = st.columns(2)

                with col_a:
                    st.subheader("Signal Attribution")
                    st.write(f"**Top Supporting Signal:** {result['top_signal']}")
                    st.write(f"**Weakest Signal:** {result['weakest_signal']}")
                    st.write(f"**Training Mode:** {result['weighting_mode']}")

                with col_b:
                    st.subheader("Meta Neural Inputs")
                    for feature in result["meta_features"]:
                        st.write(f"- {feature}")

                st.divider()

                st.warning(
                    "Disclaimer: FinMan AlphaCore is an experimental technical-analysis model. "
                    "It is not financial advice and should not be used as the sole basis for trading or investment decisions."
                )

            except Exception as e:
                st.error(f"Error: {e}")
