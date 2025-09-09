# train a logistic regression model and save metrics + plots

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

FEATURES = ["SMA_5", "SMA_10", "Return_1", "Return_3", "RSI_14", "Volume"]

def plot_confusion(cm: np.ndarray, labels, outpath: str):
    # simple confusion matrix image
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def main():
    # parse args
    p = argparse.ArgumentParser(description="train logistic regression on clean features")
    p.add_argument("--input", required=True, help="clean csv from preprocess step")
    p.add_argument("--model", required=True, help="path to save model, e.g., results/model.joblib")
    p.add_argument("--metrics", required=True, help="path to save metrics.json")
    p.add_argument("--plots", required=True, help="folder to save confusion matrix plot")
    args = p.parse_args()

    # load
    df = pd.read_csv(args.input, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    X = df[FEATURES].values
    y = df["Target"].values

    # time-aware split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # eval
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    # ensure output dirs
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    os.makedirs(args.plots, exist_ok=True)

    # save model + metrics + confusion matrix
    joblib.dump({"model": clf, "features": FEATURES}, args.model)
    with open(args.metrics, "w") as f:
        json.dump({"accuracy": acc, "f1": f1}, f, indent=2)
    plot_confusion(cm, ["down", "up"], os.path.join(args.plots, "confusion_matrix.png"))

    print(f"training done. accuracy={acc:.3f}, f1={f1:.3f}")
    print(f"saved model to {args.model}")
    print(f"saved metrics to {args.metrics}")
    print(f"saved confusion matrix to {os.path.join(args.plots, 'confusion_matrix.png')}")

if __name__ == "__main__":
    main()
