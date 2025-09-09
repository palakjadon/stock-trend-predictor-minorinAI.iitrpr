# read raw csv, add features, create target, save a clean csv

import argparse
import pandas as pd
from features import add_features, create_target

def main():
    # parse args
    p = argparse.ArgumentParser(description="preprocess ohlcv csv into feature/target csv")
    p.add_argument("--input", required=True, help="path to raw csv with Date,Open,High,Low,Close,Volume")
    p.add_argument("--output", required=True, help="path to save cleaned csv, e.g., data/clean_stock.csv")
    args = p.parse_args()

    # load and sort
    df = pd.read_csv(args.input, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # add features + target
    df = add_features(df)
    df = create_target(df)

    # save
    df.to_csv(args.output, index=False)
    print(f"saved cleaned data to {args.output}")

if __name__ == "__main__":
    main()
