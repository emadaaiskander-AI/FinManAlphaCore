from pathlib import Path
import json
import time

import pandas as pd
import yfinance as yf


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "blog" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "sector_tickers.json"

MAX_TOTAL_TICKERS = 2000
MAX_PER_SECTOR = 400

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


SECTOR_MAP = {
    "Technology": ["Technology", "Communication Services"],
    "Healthcare": ["Healthcare"],
    "Financials": ["Financial Services", "Financial"],
    "Energy": ["Energy"],
    "Consumer": ["Consumer Cyclical", "Consumer Defensive"],
    "Industrials": ["Industrials", "Basic Materials"],
    "Real Estate": ["Real Estate"],
    "Utilities": ["Utilities"],
}


def load_exchange_tickers():
    tickers = []

    nasdaq = pd.read_csv(NASDAQ_LISTED, sep="|")
    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
    nasdaq = nasdaq[nasdaq["Financial Status"] == "N"]
    nasdaq = nasdaq[~nasdaq["Security Name"].str.contains("ETF|Fund|Warrant|Right|Unit", case=False, na=False)]
    tickers.extend(nasdaq["Symbol"].dropna().tolist())

    other = pd.read_csv(OTHER_LISTED, sep="|")
    other = other[other["Test Issue"] == "N"]
    other = other[other["ETF"] == "N"]
    other = other[~other["Security Name"].str.contains("ETF|Fund|Warrant|Right|Unit", case=False, na=False)]
    tickers.extend(other["ACT Symbol"].dropna().tolist())

    clean = []
    for t in tickers:
        t = str(t).strip().replace(".", "-")
        if t and "$" not in t and "^" not in t:
            clean.append(t)

    return sorted(set(clean))


def map_sector(yf_sector):
    for target_sector, possible_names in SECTOR_MAP.items():
        if yf_sector in possible_names:
            return target_sector
    return "Other"


def get_sector(ticker):
    try:
        info = yf.Ticker(ticker).fast_info
        _ = info.get("last_price", None)

        full_info = yf.Ticker(ticker).info
        sector = full_info.get("sector")
        if not sector:
            return "Other"

        return map_sector(sector)

    except Exception:
        return None


def main():
    tickers = load_exchange_tickers()

    print(f"Loaded exchange tickers: {len(tickers)}")
    print(f"Target total tickers: {MAX_TOTAL_TICKERS}")

    sectors = {sector: [] for sector in SECTOR_MAP.keys()}
    sectors["Other"] = []

    added = 0

    for ticker in tickers:
        if added >= MAX_TOTAL_TICKERS:
            break

        sector = get_sector(ticker)

        if sector is None:
            print(f"Skipped {ticker}: unable to read sector")
            continue

        if len(sectors[sector]) >= MAX_PER_SECTOR:
            continue

        sectors[sector].append(ticker)
        added += 1

        print(f"Added {ticker} -> {sector} | Total: {added}")

        time.sleep(0.15)

    sectors = {k: v for k, v in sectors.items() if v}

    OUTPUT_FILE.write_text(json.dumps(sectors, indent=2), encoding="utf-8")

    print(f"\nCreated: {OUTPUT_FILE}")
    print(f"Total tickers saved: {sum(len(v) for v in sectors.values())}")

    for sector, sector_tickers in sectors.items():
        print(f"{sector}: {len(sector_tickers)}")


if __name__ == "__main__":
    main()