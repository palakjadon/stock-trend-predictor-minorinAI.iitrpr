# stock market trend predictor (ai)

this repo predicts next-day stock direction (up/down) from historical ohlcv data using simple technical indicators and a logistic regression model.

## prediction logic (clear + simple)
- input: daily ohlcv (open, high, low, close, volume)
- features:
  - sma_5, sma_10 (simple moving averages on close)
  - rsi_14 (momentum)
  - return_1, return_3 (pct changes)
- target:
  - 1 if tomorrow’s close > today’s close, else 0
- model:
  - logistic regression (fast, interpretable)
- outputs:
  - metrics.json (accuracy, f1)
  - confusion_matrix.png (plot)
  - price_with_predictions.png (plot with predicted “up” markers)

## quickstart (local)
1) install python 3.9+  
2) create a virtualenv (optional but recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/make_dataset.py --out data/sample_stock.csv --days 180 --seed 42
python src/make_dataset.py --out data/apple.csv --ticker AAPL --start 2023-01-01 --end 2023-12-31
python src/preprocess.py --input data/sample_stock.csv --output data/clean_stock.csv
python src/train.py --input data/clean_stock.csv --model results/model.joblib --metrics results/metrics.json --plots results
python src/evaluate.py --input data/clean_stock.csv --model results/model.joblib --out results


