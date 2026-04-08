# ONNX Model Assets

This directory holds pre-converted ONNX model files used by the TradeView application.
These files are **loaded at runtime via ONNX Runtime** — no TensorFlow, PyTorch, or Transformers
runtime is required on the server.

---

## Expected Files

| File | Description | Required? |
|---|---|---|
| `lstm_price.onnx` | LSTM stock price forecasting model | Optional |
| `sentiment_distilbert.onnx` | DistilBERT SST-2 sentiment classifier | Optional |
| `sentiment_vocab.json` | Tokenizer vocabulary for the sentiment model | Optional |

> If the `.onnx` files are missing, the app automatically falls back to:
> - **Price forecasting** → `sklearn` LinearRegression (trend extrapolation)
> - **Sentiment** → NLTK VADER (keyword-based scoring)

---

## How to Generate These Files

Run `scripts/export_models.py` from a local environment that has TensorFlow and `optimum` installed:

```bash
# One-time setup (local machine only, not needed on Render)
pip install tensorflow optimum[onnxruntime] onnx

# Export both models
python scripts/export_models.py
```

The script will:
1. Train an LSTM model on sample data (or use a pre-trained Keras `.h5` file if present)
2. Convert it to `models/lstm_price.onnx`
3. Export DistilBERT SST-2 to `models/sentiment_distilbert.onnx` + `models/sentiment_vocab.json`

---

## Re-deploying After Export

After generating the `.onnx` files, commit them to your repo and redeploy:

```bash
git add models/lstm_price.onnx models/sentiment_distilbert.onnx models/sentiment_vocab.json
git commit -m "Add pre-trained ONNX models"
git push
```

Render will pick them up on the next deploy. The app will automatically use the ONNX path.
