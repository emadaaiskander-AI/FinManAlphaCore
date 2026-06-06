import argparse
from alphacore.engine import train_and_predict, run_backtest


def print_prediction(result):
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


def print_backtest(result):
    print("\n==============================")
    print("FinMan AlphaCore")
    print("Backtest Report")
    print("==============================")

    print(f"Ticker: {result['ticker']}")
    print(f"Period: {result['period']}")
    print(f"Samples: {result['samples']}")
    print(f"Minimum Training Size: {result['min_train_size']}")
    print(f"Step Size: {result['step_size']}")

    print("\nBacktest Metrics")
    print("------------------------------------")

    for horizon, metrics in result["metrics"].items():
        print(f"\n{horizon.replace('_', ' ').title()}")
        print(f"Samples: {metrics['samples']}")
        print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Average Predicted Return: {metrics['avg_predicted_return']:.2f}%")
        print(f"Average Actual Return: {metrics['avg_actual_return']:.2f}%")

        bullish = metrics["bullish_hit_rate"]
        bearish = metrics["bearish_hit_rate"]

        print(
            "Bullish Hit Rate: "
            + ("N/A" if bullish is None else f"{bullish:.2f}%")
        )

        print(
            "Bearish Hit Rate: "
            + ("N/A" if bearish is None else f"{bearish:.2f}%")
        )

    print("\nRecent Backtest Rows")
    print("------------------------------------")
    print(result["backtest"].tail(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Run FinMan AlphaCore Meta Neural prediction and backtest"
    )

    parser.add_argument(
        "--ticker",
        required=True,
        help="Stock ticker symbol, e.g. AAPL"
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run historical backtest"
    )

    parser.add_argument(
        "--period",
        default="3y",
        help="Yahoo Finance period, e.g. 1y, 2y, 3y, 5y"
    )

    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="Backtest step size in trading days"
    )

    args = parser.parse_args()

    prediction = train_and_predict(args.ticker)
    print_prediction(prediction)

    if args.backtest:
        backtest = run_backtest(
            ticker=args.ticker,
            period=args.period,
            step_size=args.step_size,
        )
        print_backtest(backtest)

    print("\n==============================")
    print("End of Report")
    print("==============================")


if __name__ == "__main__":
    main()
