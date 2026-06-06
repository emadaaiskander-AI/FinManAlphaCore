import streamlit as st
import subprocess
import re


st.set_page_config(
    page_title="FinMan AlphaCore",
    page_icon="📈",
    layout="wide",
)


def extract(pattern, text, default="N/A"):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default


def reliability_label(similar_count):
    if similar_count >= 100:
        return "High"
    if similar_count >= 25:
        return "Medium"
    return "Low"


def run_prediction(ticker):
    result = subprocess.run(
        ["python", "predict.py", "--ticker", ticker],
        capture_output=True,
        text=True,
        timeout=180,
    )

    output = result.stdout + "\n" + result.stderr

    similar_count = int(float(extract(r"similar_count:\s*([0-9.]+)", output, "0")))

    return {
        "raw_output": output,
        "ticker": ticker.upper(),
        "latest_price": extract(r"Latest Price:\s*\$?([0-9.]+)", output, "0"),
        "regime": extract(r"Current Regime:\s*(.+)", output),
        "bias": extract(r"Bias:\s*(.+)", output),
        "confidence": float(extract(r"Confidence:\s*([0-9.]+)%", output, "0")),
        "mae": extract(r"Model MAE:\s*([0-9.]+)", output),
        "expected_5_day_return": extract(r"5 Trading Days\s+([\-0-9.]+)", output, "0"),
        "predicted_5_day_price": extract(r"5 Trading Days\s+[\-0-9.]+\s+([0-9.]+)", output, "0"),
        "expected_10_day_return": extract(r"10 Trading Days\s+([\-0-9.]+)", output, "0"),
        "predicted_10_day_price": extract(r"10 Trading Days\s+[\-0-9.]+\s+([0-9.]+)", output, "0"),
        "expected_14_day_return": extract(r"14 Trading Days\s+([\-0-9.]+)", output, "0"),
        "predicted_14_day_price": extract(r"14 Trading Days\s+[\-0-9.]+\s+([0-9.]+)", output, "0"),
        "similar_count": similar_count,
        "similar_success_rate": float(extract(r"success_rate:\s*([0-9.]+)", output, "0")),
        "avg_similarity": float(extract(r"avg_similarity:\s*([0-9.]+)", output, "0")),
        "avg_14_day_return": extract(r"avg_14_day_return:\s*([\-0-9.]+)", output, "0"),
        "reliability": reliability_label(similar_count),
        "top_signal": extract(r"Top Signal\s*-+\s*(.+)", output),
        "weakest_signal": extract(r"Weakest Signal\s*-+\s*(.+)", output),
    }


def quality_label(confidence, success_rate, similar_count):
    if success_rate >= 70 and similar_count >= 100 and confidence >= 55:
        return "Strong Evidence"
    if success_rate >= 65 and similar_count >= 25:
        return "Moderate Evidence"
    if success_rate >= 80 and similar_count < 25:
        return "Interesting but Low Sample"
    return "Weak / Needs Review"


st.title("FinMan AlphaCore Prediction")
st.caption("Technology Experiment • Not Financial Advice")

ticker = st.text_input("Enter stock ticker", value="AAPL").upper().strip()

run_button = st.button("Run Prediction")

if run_button and ticker:
    with st.spinner(f"Running AlphaCore prediction for {ticker}..."):
        try:
            result = run_prediction(ticker)

            evidence_label = quality_label(
                result["confidence"],
                result["similar_success_rate"],
                result["similar_count"],
            )

            st.subheader(f"{result['ticker']} Prediction Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Bias", result["bias"])
            col2.metric("Confidence", f"{result['confidence']:.1f}%")
            col3.metric("Historical Accuracy", f"{result['similar_success_rate']:.1f}%")
            col4.metric("Reliability", result["reliability"])

            st.divider()

            col5, col6, col7, col8 = st.columns(4)

            col5.metric("Similar Setups", result["similar_count"])
            col6.metric("Avg Similarity", f"{result['avg_similarity']:.1f}%")
            col7.metric("Evidence Label", evidence_label)
            col8.metric("Current Regime", result["regime"])

            st.divider()

            st.subheader("Forecast")

            forecast_data = {
                "Horizon": ["5 Trading Days", "10 Trading Days", "14 Trading Days"],
                "Expected Return %": [
                    result["expected_5_day_return"],
                    result["expected_10_day_return"],
                    result["expected_14_day_return"],
                ],
                "Predicted Price": [
                    f"${result['predicted_5_day_price']}",
                    f"${result['predicted_10_day_price']}",
                    f"${result['predicted_14_day_price']}",
                ],
            }

            st.table(forecast_data)

            st.subheader("Signal Explanation")

            col9, col10, col11 = st.columns(3)

            col9.metric("Top Signal", result["top_signal"])
            col10.metric("Weakest Signal", result["weakest_signal"])
            col11.metric("Model MAE", result["mae"])

            st.info(
                "Confidence reflects the model's current conviction. "
                "Historical accuracy reflects how similar setups performed in the past. "
                "Similar setup count and reliability help judge how much evidence supports the result."
            )

            st.subheader("Raw AlphaCore Report")

            with st.expander("Show full report"):
                st.code(result["raw_output"])

            st.divider()

            st.caption(
                "Disclaimer: FinMan AlphaCore is an experimental AI and market research project. "
                "This is a technology demonstration and not financial advice, investment advice, "
                "trading advice, or a recommendation to buy or sell any security."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
