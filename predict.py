import argparse
from alphacore.engine import train_and_predict


def main():
    parser = argparse.ArgumentParser(
        description="Run FinMan AlphaCore Meta Neural prediction"
    )

    parser.add_argument(
        "--ticker",
        required=True,
        help="Stock ticker symbol (e.g. AAPL)"
    )

    args = parser.parse_args()

    result = train_and_predict(args.ticker)

    print("\n==============================")
    print("FinMan AlphaCore")
    print("Meta Neural Prediction Report")
    print("==============================")

    print(f"Ticker: {result['ticker']}")
    print(f"Latest Price: ${result['latest_price']:.2f}")
    print(f"Current Regime: {result['current_regime']}")
    print(f"Bias: {result['bias']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Model MAE: {result['model_mae']:.4f}")
    print(f"Training Mode: {result['weighting_mode']}")

    print("\nForecast")
    print("------------------------------------")
    print(result["forecast"].to_string(index=False))

    print("\nTop Signal")
    print("------------------------------------")
    print(result["top_signal"])

    print("\nWeakest Signal")
    print("------------------------------------")
    print(result["weakest_signal"])

    print("\nStrategy Scores")
    print("------------------------------------")
    print(result["scores"].to_string())

    print("\nMeta Features Used")
    print("------------------------------------")
    for feature in result["meta_features"]:
        print(f"- {feature}")

    print("\n==============================")
    print("End of Report")
    print("==============================")


if __name__ == "__main__":
    main()
