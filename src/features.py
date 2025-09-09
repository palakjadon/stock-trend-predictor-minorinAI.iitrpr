# feature engineering helpers

import numpy as np
import pandas as pd

def sma(series: pd.Series, window: int) -> pd.Series:
    # simple moving average
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    # relative strength index using ewma (stable + simple)
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # adds sma, rsi, and simple returns
    out = df.copy()
    out["SMA_5"] = sma(out["Close"], 5)
    out["SMA_10"] = sma(out["Close"], 10)
    out["Return_1"] = out["Close"].pct_change(1)
    out["Return_3"] = out["Close"].pct_change(3)
    out["RSI_14"] = rsi(out["Close"], 14)
    # fill edge na values created by rolling/shift
    out = out.fillna(method="ffill").fillna(method="bfill")
    return out

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    # target is 1 if next day close > today close else 0
    out = df.copy()
    out["Target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out = out.dropna(subset=["Target"])
    return out
