from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import re
import json

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
BLOG_DIR = PROJECT_DIR / "blog"
POSTS_DIR = BLOG_DIR / "posts"
DATA_DIR = BLOG_DIR / "data"

POSTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")

MAX_TICKERS_PER_SECTOR = 50
BENCHMARK = "SPY"


SECTORS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN",
        "QCOM", "TXN", "IBM", "INTU", "NOW", "AMAT", "MU", "LRCX", "ADI", "PANW",
        "SNPS", "CDNS", "KLAC", "APH", "MSI", "NXPI", "FTNT", "ADSK", "ROP", "TEL",
        "WDAY", "MCHP", "ANET", "GLW", "ON", "DELL", "HPQ", "HPE", "FICO", "KEYS",
        "MPWR", "TER", "TYL", "NTAP", "ZBRA", "ENPH", "SWKS", "STX", "WDC", "GEN"
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "ISRG", "DHR", "PFE",
        "AMGN", "BSX", "SYK", "GILD", "VRTX", "REGN", "MDT", "CI", "ELV", "BMY",
        "ZTS", "HCA", "CVS", "MCK", "COR", "BDX", "EW", "A", "IDXX", "IQV",
        "RMD", "DXCM", "MTD", "WST", "STE", "ALGN", "COO", "HOLX", "LH", "DGX",
        "MOH", "HUM", "CAH", "WAT", "TECH", "BAX", "INCY", "PODD", "MRNA", "BIIB"
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
        "MMC", "CB", "PGR", "AON", "ICE", "CME", "USB", "PNC", "COF", "TFC",
        "MCO", "AFL", "MET", "PRU", "ALL", "TRV", "AIG", "BK", "AMP", "MSCI",
        "DFS", "STT", "FITB", "HBAN", "RF", "CFG", "NDAQ", "BRO", "AJG", "WTW",
        "CINF", "L", "WRB", "MTB", "KEY", "SYF", "EG", "RJF", "TROW", "BEN"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB",
        "KMI", "OKE", "HAL", "BKR", "FANG", "DVN", "HES", "CTRA", "EQT", "TRGP",
        "APA", "MRO", "MUR", "NOV", "CHRD", "MTDR", "RRC", "VNOM", "PR", "AR",
        "SM", "CIVI", "PBF", "SUN", "DINO", "PAA", "ET", "EPD", "MPLX", "WES",
        "ENB", "TRP", "CNQ", "SU", "IMO", "TOU.TO", "MEG.TO", "CVE", "OVV", "NFG"
    ],
    "Consumer": [
        "AMZN", "TSLA", "HD", "MCD", "LOW", "BKNG", "TJX", "SBUX", "NKE", "CMG",
        "ORLY", "AZO", "MAR", "HLT", "ROST", "YUM", "DHI", "LEN", "EBAY", "TSCO",
        "ULTA", "DRI", "BBY", "TGT", "F", "GM", "KMX", "POOL", "NVR", "PHM",
        "LULU", "DECK", "DPZ", "EXPE", "RCL", "CCL", "WYNN", "MGM", "CZR", "HAS",
        "BBWI", "TPR", "RL", "GPC", "LKQ", "BWA", "APTV", "ETSY", "MHK", "WHR"
    ],
    "Industrials": [
        "GE", "CAT", "RTX", "HON", "UNP", "BA", "ETN", "DE", "LMT", "UPS",
        "ADP", "TDG", "NOC", "GD", "ITW", "WM", "EMR", "PH", "CSX", "NSC",
        "FDX", "CARR", "JCI", "MMM", "ROK", "AME", "OTIS", "PCAR", "FAST", "CMI",
        "URI", "GWW", "IR", "PWR", "XYL", "HWM", "DOV", "VRSK", "CTAS", "EFX",
        "RSG", "WAB", "LHX", "TXT", "MAS", "ALLE", "GNRC", "AOS", "CHRW", "PAYX"
    ],
}


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
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )

    output = result.stdout + "\n" + result.stderr

    similar_count = int(float(extract(r"similar_count:\s*([0-9.]+)", output, "0")))

    return {
        "ticker": ticker,
        "latest_price": float(extract(r"Latest Price:\s*\$?([0-9.]+)", output, "0")),
        "regime": extract(r"Current Regime:\s*(.+)", output),
        "bias": extract(r"Bias:\s*(.+)", output),
        "confidence": float(extract(r"Confidence:\s*([0-9.]+)%", output, "0")),
        "mae": extract(r"Model MAE:\s*([0-9.]+)", output),
        "expected_14_day_return": float(extract(r"14 Trading Days\s+([\-0-9.]+)", output, "0")),
        "predicted_14_day_price": extract(r"14 Trading Days\s+[\-0-9.]+\s+([0-9.]+)", output, "0"),
        "similar_count": similar_count,
        "similar_success_rate": float(extract(r"success_rate:\s*([0-9.]+)", output, "0")),
        "avg_similarity": float(extract(r"avg_similarity:\s*([0-9.]+)", output, "0")),
        "reliability": reliability_label(similar_count),
        "top_signal": extract(r"Top Signal\s*-+\s*(.+)", output),
        "weakest_signal": extract(r"Weakest Signal\s*-+\s*(.+)", output),
    }


def score_pick(p):
    return (
        p["confidence"] * 0.30
        + p["similar_success_rate"] * 0.30
        + p["expected_14_day_return"] * 0.20
        + min(p["similar_count"], 100) * 0.15
        + p["avg_similarity"] * 0.05
    )


def save_today_picks(daily_picks):
    picks_file = DATA_DIR / "picks_history.json"

    history = json.loads(picks_file.read_text(encoding="utf-8")) if picks_file.exists() else []
    history = [h for h in history if h["date"] != today]

    history.append({
        "date": today,
        "benchmark": BENCHMARK,
        "picks": daily_picks,
    })

    history = sorted(history, key=lambda x: x["date"])
    picks_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def get_price_on_or_after(price_df, ticker, date_str):
    if ticker not in price_df.columns:
        return None

    date = pd.to_datetime(date_str)
    series = price_df[ticker].dropna()
    series = series[series.index >= date]

    return float(series.iloc[0]) if not series.empty else None


def get_latest_price(price_df, ticker):
    if ticker not in price_df.columns:
        return None

    series = price_df[ticker].dropna()
    return float(series.iloc[-1]) if not series.empty else None


def update_real_performance(picks_history):
    perf_file = DATA_DIR / "performance_history.json"
    graph_file = BLOG_DIR / "performance.svg"

    if not picks_history:
        return {
            "period": "Last 2 Weeks",
            "alphacore_basket": "+0.0%",
            "spy": "+0.0%",
            "excess_return": "+0.0%",
        }

    cutoff_date = datetime.now() - timedelta(days=21)
    active_history = [
        h for h in picks_history
        if pd.to_datetime(h["date"]) >= pd.to_datetime(cutoff_date)
    ]

    tickers = {BENCHMARK}

    for h in active_history:
        for p in h["picks"]:
            tickers.add(p["ticker"])

    tickers = sorted(tickers)

    start_date = (pd.to_datetime(active_history[0]["date"]) - timedelta(days=5)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        group_by="column",
    )

    alpha_return = 0.0
    spy_return = 0.0

    if not data.empty:
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else pd.DataFrame({tickers[0]: data["Close"]})

        basket_returns = []
        spy_returns = []

        for h in active_history:
            pick_returns = []

            for p in h["picks"]:
                start_price = get_price_on_or_after(close, p["ticker"], h["date"])
                latest_price = get_latest_price(close, p["ticker"])

                if start_price and latest_price and start_price > 0:
                    pick_returns.append((latest_price / start_price - 1) * 100)

            if pick_returns:
                basket_returns.append(sum(pick_returns) / len(pick_returns))

            spy_start = get_price_on_or_after(close, BENCHMARK, h["date"])
            spy_latest = get_latest_price(close, BENCHMARK)

            if spy_start and spy_latest and spy_start > 0:
                spy_returns.append((spy_latest / spy_start - 1) * 100)

        alpha_return = sum(basket_returns) / len(basket_returns) if basket_returns else 0.0
        spy_return = sum(spy_returns) / len(spy_returns) if spy_returns else 0.0

    excess = alpha_return - spy_return

    perf_history = json.loads(perf_file.read_text(encoding="utf-8")) if perf_file.exists() else []
    perf_history = [p for p in perf_history if p["date"] != today]

    perf_history.append({
        "date": today,
        "alphacore": round(alpha_return, 2),
        "spy": round(spy_return, 2),
        "excess": round(excess, 2),
    })

    perf_history = sorted(perf_history, key=lambda x: x["date"])
    perf_file.write_text(json.dumps(perf_history, indent=2), encoding="utf-8")

    df = pd.DataFrame(perf_history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(14)

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["alphacore"], marker="o", label="AlphaCore Basket")
    plt.plot(df["date"], df["spy"], marker="o", label="SPY Benchmark")
    plt.title("Rolling 2-Week Performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return %")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graph_file, format="svg")
    plt.close()

    return {
        "period": "Last 2 Weeks",
        "alphacore_basket": f"{alpha_return:+.2f}%",
        "spy": f"{spy_return:+.2f}%",
        "excess_return": f"{excess:+.2f}%",
    }


daily_picks = []

for sector, tickers in SECTORS.items():
    results = []
    selected_tickers = tickers[:MAX_TICKERS_PER_SECTOR]

    print(f"\nRunning sector: {sector}")
    print(f"Tickers: {len(selected_tickers)}")

    for ticker in selected_tickers:
        try:
            prediction = run_prediction(ticker)
            prediction["sector"] = sector
            prediction["score"] = score_pick(prediction)
            results.append(prediction)

            print(
                f"Completed {ticker}: "
                f"{prediction['bias']} | "
                f"Conf {prediction['confidence']:.1f}% | "
                f"Hist {prediction['similar_success_rate']:.1f}% | "
                f"Count {prediction['similar_count']} | "
                f"Reliability {prediction['reliability']} | "
                f"Score {prediction['score']:.2f}"
            )

        except Exception as e:
            print(f"Failed {ticker}: {e}")

    if results:
        best = sorted(results, key=lambda x: x["score"], reverse=True)[0]
        daily_picks.append(best)
        print(f"Best for {sector}: {best['ticker']}")


picks_history = save_today_picks(daily_picks)
performance = update_real_performance(picks_history)


disclaimer = """
**Disclaimer – Technology Experiment / Not Financial Advice**

FinMan AlphaCore is an experimental artificial intelligence and market research project created for educational, research, and technology demonstration purposes.

The information presented is generated from automated model analysis and historical market data. It is not financial advice, investment advice, trading advice, legal advice, tax advice, or a recommendation to buy, sell, or hold any security.

Model predictions may be inaccurate. Past performance does not guarantee future results. Investing involves risk, including possible loss of principal.
"""


rows = "\n".join(
    f"| {p['sector']} | {p['ticker']} | {p['bias']} | {p['confidence']:.1f}% | "
    f"{p['similar_success_rate']:.1f}% | {p['similar_count']} | {p['avg_similarity']:.1f}% | "
    f"{p['reliability']} | {p['expected_14_day_return']:.2f}% | "
    f"${p['predicted_14_day_price']} | {p['top_signal']} |"
    for p in daily_picks
)


history_rows = ""
perf_file = DATA_DIR / "performance_history.json"

if perf_file.exists():
    perf_history = json.loads(perf_file.read_text(encoding="utf-8"))
    for h in sorted(perf_history, key=lambda x: x["date"], reverse=True)[:10]:
        history_rows += (
            f"| {h['date']} | {h['alphacore']:+.2f}% | "
            f"{h['spy']:+.2f}% | {h['excess']:+.2f}% |\n"
        )


post = f"""# FinMan AlphaCore Daily Picks – {today}

## Today's Picks

| Sector | Pick | Bias | Confidence | Historical Accuracy | Similar Setups | Avg Similarity | Reliability | Expected 14-Day Return | Predicted 14-Day Price | Top Signal |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
{rows}

## Rolling 2-Week Performance

| Period | AlphaCore Basket | SPY | Excess Return |
|---|---:|---:|---:|
| {performance['period']} | {performance['alphacore_basket']} | {performance['spy']} | {performance['excess_return']} |

![Rolling 2-Week Performance](../performance.svg)

## Historical Performance

| Date | AlphaCore Basket | SPY | Excess Return |
|---|---:|---:|---:|
{history_rows}

## History

This page is part of the daily AlphaCore public tracking archive.

{disclaimer}
"""


post_file = POSTS_DIR / f"{today}.md"
latest_file = BLOG_DIR / "latest.md"
json_file = DATA_DIR / "today.json"

post_file.write_text(post, encoding="utf-8")
latest_file.write_text(post, encoding="utf-8")

json_file.write_text(
    json.dumps(
        {
            "date": today,
            "daily_picks": daily_picks,
            "performance": performance,
            "disclaimer": disclaimer.strip(),
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"\nGenerated blog post: {post_file}")
print(f"Updated latest page: {latest_file}")
print(f"Updated JSON data: {json_file}")
print(f"Updated picks history: {DATA_DIR / 'picks_history.json'}")
print(f"Updated performance history: {DATA_DIR / 'performance_history.json'}")
print(f"Updated graph: {BLOG_DIR / 'performance.svg'}")
