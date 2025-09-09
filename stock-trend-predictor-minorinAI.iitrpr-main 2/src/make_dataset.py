# dataset generator: can make synthetic or fetch yahoo finance data

import argparse
import numpy as np
import pandas as pd

def make_synthetic(days: int, seed: int = 42) -> pd.DataFrame:
    # generate a simple random walk close price and derive ohlcv around it
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    step = rng.normal(loc=0.0, scale=1.2, size=len(dates))
    close = 100 + np.cumsum(step)
    open_ = close + rng.normal(0, 0.5, size=len(dates))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1.0, size=len(dates)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1.0, size=len(dates)))
    volume = rng.integers(800_000, 3_000_000, size=len(dates))

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_.round(2),
        "High": high.round(2),
        "Low": low.round(2),
        "Close": close.round(2),
        "Volume": volume
    })
    return df

def make_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    # fetch data from yahoo finance
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return df

def main():
    # parse args
    p = argparse.ArgumentParser(description="make dataset (synthetic or yahoo finance)")
    p.add_argument("--out", required=True, help="output csv path, e.g., data/sample_stock.csv")
    p.add_argument("--days", type=int, default=180, help="number of business days (for synthetic)")
    p.add_argument("--seed", type=int, default=42, help="random seed for synthetic data")
    p.add_argument("--ticker", type=str, help="yahoo finance ticker (e.g., AAPL, RELIANCE.NS)")
    p.add_argument("--start", type=str, default="2023-01-01", help="start date for yahoo finance (yyyy-mm-dd)")
    p.add_argument("--end", type=str, default="2023-12-31", help="end date for yahoo finance (yyyy-mm-dd)")
    args = p.parse_args()

    # decide mode
    if args.ticker:
        print(f"fetching yahoo finance data for {args.ticker}...")
        df = make_yahoo(args.ticker, args.start, args.end)
    else:
        print("generating synthetic random walk data...")
        df = make_synthetic(days=args.days, seed=args.seed)

    # save
    df.to_csv(args.out, index=False)
    print(f"saved data to {args.out}")

if __name__ == "__main__":
    main()
