# stock market trend predictor 📈

this project predicts whether a stock's price will go **up or down tomorrow** using past stock data and some simple indicators.  


## 🔍 what this project does
- takes in stock price data (date, open, high, low, close, volume)  
- calculates extra features like:
  - sma → average of past prices  
  - rsi → momentum signal (overbought/oversold)  
  - returns → percentage change over 1-day & 3-days  
- creates a target:  
  - 1 = tomorrow goes up  
  - 0 = tomorrow goes down  
- trains a **logistic regression model** (a very simple, explainable ml algorithm)  
- shows results in two ways:
  - **confusion_matrix.png** → how well predictions match reality  
  - **price_with_predictions.png** → stock chart with “up” arrows on predicted rise days  


## ⚙️ how to set up
1. install **python 3.9+**  
2. open terminal in this folder  
3. (optional) create a virtual environment:  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # windows: .venv\Scripts\activate


4. install required libraries:

   ```bash
   pip install -r requirements.txt
   ```
5. (if you want real stock data) also install:

   ```bash
   pip install yfinance
   ```

## 📊 how to use

### step 1: get stock data

* **synthetic (fake, for demo):**

  ```bash
  python src/make_dataset.py --out data/sample_stock.csv --days 180 --seed 42
  ```
* **real stock (from yahoo finance):**

  ```bash
  python src/make_dataset.py --out data/apple.csv --ticker AAPL --start 2023-01-01 --end 2023-12-31
  ```

  examples of tickers:

  * apple → `AAPL`
  * microsoft → `MSFT`
  * google → `GOOG`
  * reliance (india) → `RELIANCE.NS`

### step 2: preprocess data

```bash
python src/preprocess.py --input data/apple.csv --output data/apple_clean.csv
```

### step 3: train model

```bash
python src/train.py --input data/apple_clean.csv --model results/model.joblib --metrics results/metrics.json --plots results
```

### step 4: evaluate predictions

```bash
python src/evaluate.py --input data/apple_clean.csv --model results/model.joblib --out results
```


## 📂 what you’ll get in `results/`

* `metrics.json` → accuracy and f1 score
* `confusion_matrix.png` → 2x2 table of predicted vs actual
* `price_with_predictions.png` → stock chart with arrows for predicted “up” days


## 🌟 why this project is useful

* very easy to understand for beginners in ML & finance
* shows how basic indicators + logistic regression can already find some patterns
* can be extended to advanced models (XGBoost, LSTM) or sentiment analysis


## 📌 notes

* accuracy will not be very high (markets are noisy!).
* use this as a **learning project**, not for real trading decisions.

```
