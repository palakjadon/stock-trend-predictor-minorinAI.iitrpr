# load the saved model and plot price with predicted up signals

import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # parse args
    p = argparse.ArgumentParser(description="evaluate model and create price plot with predicted up markers")
    p.add_argument("--input", required=True, help="clean csv from preprocess step")
    p.add_argument("--model", required=True, help="path to model.joblib from training")
    p.add_argument("--out", required=True, help="folder to save plots")
    args = p.parse_args()

    # load data + model
    df = pd.read_csv(args.input, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    features = bundle["features"]

    # predict on full set for visualization
    preds = model.predict(df[features].values)
    df["Pred"] = preds

    # plot close price and mark predicted ups
    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="close")
    ups = df[df["Pred"] == 1]
    ax.scatter(ups["Date"], ups["Close"], marker="^", s=30, label="predicted up")
    ax.set_title("close price with predicted up signals")
    ax.set_xlabel("date"); ax.set_ylabel("price")
    ax.legend()
    plt.tight_layout()
    outpath = os.path.join(args.out, "price_with_predictions.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    print(f"saved price plot to {outpath}")

if __name__ == "__main__":
    main()
