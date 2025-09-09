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

def main():
    # parse args
    p = argparse.ArgumentParser(description="make a tiny synthetic ohlcv dataset")
    p.add_argument("--out", required=True, help="output csv path, e.g., data/sample_stock.csv")
    p.add_argument("--days", type=int, default=180, help="number of business days")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = p.parse_args()

    # generate and save
    df = make_synthetic(days=args.days, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"saved synthetic data to {args.out}")

if __name__ == "__main__":
    main()
