"""
scripts/export_models.py
========================
Run this ONCE on your LOCAL machine (with TensorFlow + optimum installed) to produce
the ONNX model files used by TradeView on Render.

Requirements (local only, NOT on Render):
    pip install tensorflow optimum[onnxruntime] onnx tf2onnx yfinance scikit-learn

Usage:
    python scripts/export_models.py

Output:
    models/lstm_price.onnx
    models/sentiment_distilbert.onnx
    models/sentiment_vocab.json
"""

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEQUENCE_LENGTH = 60
FORECAST_DAYS = 30
EPOCHS = 50


# ─────────────────────────────────────────────────────────
# 1. LSTM Price Forecasting Model → ONNX
# ─────────────────────────────────────────────────────────

def build_and_export_lstm():
    """Train a price LSTM on NIFTY50 data, then convert to ONNX."""
    logger.info("=== Exporting LSTM Price Model ===")

    try:
        import tensorflow as tf
        import tf2onnx
        import yfinance as yf
        from sklearn.preprocessing import MinMaxScaler
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install with: pip install tensorflow tf2onnx yfinance scikit-learn")
        return False

    # ── Fetch training data ──────────────────────────────
    logger.info("Fetching NIFTY50 data (5 years) for training...")
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="5y")
    if df.empty:
        logger.error("Failed to fetch NIFTY50 data.")
        return False

    prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    # ── Build sequences ─────────────────────────────────
    X, y = [], []
    for i in range(len(scaled) - SEQUENCE_LENGTH - FORECAST_DAYS + 1):
        X.append(scaled[i:i + SEQUENCE_LENGTH])
        y.append(scaled[i + SEQUENCE_LENGTH])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info(f"Training on {len(X_train)} sequences, validating on {len(X_val)}...")

    # ── Model architecture ───────────────────────────────
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, 1)),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=32, callbacks=callbacks, verbose=1)

    # ── Convert to ONNX ─────────────────────────────────
    out_path = str(MODELS_DIR / "lstm_price.onnx")
    logger.info(f"Converting to ONNX → {out_path}")
    input_signature = [tf.TensorSpec(shape=(None, SEQUENCE_LENGTH, 1), dtype=tf.float32, name="input")]
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

    with open(out_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    logger.info(f"✅ lstm_price.onnx saved ({Path(out_path).stat().st_size / 1024:.1f} KB)")
    return True


# ─────────────────────────────────────────────────────────
# 2. DistilBERT Sentiment Model → ONNX
# ─────────────────────────────────────────────────────────

def export_sentiment_model():
    """Export DistilBERT SST-2 to ONNX using Hugging Face Optimum."""
    logger.info("=== Exporting DistilBERT Sentiment Model ===")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install with: pip install optimum[onnxruntime] transformers")
        return False

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    out_dir = MODELS_DIR / "sentiment_export"

    logger.info(f"Downloading and exporting '{model_name}' to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
    ort_model.save_pretrained(str(out_dir))

    # Copy the key ONNX file and vocab
    import shutil
    onnx_src = out_dir / "model.onnx"
    if not onnx_src.exists():
        # Try alternate name
        candidates = list(out_dir.glob("*.onnx"))
        onnx_src = candidates[0] if candidates else None

    if onnx_src:
        shutil.copy(onnx_src, MODELS_DIR / "sentiment_distilbert.onnx")
        logger.info(f"✅ sentiment_distilbert.onnx saved ({onnx_src.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        logger.error("Could not locate exported .onnx file.")
        return False

    # Save tokenizer vocab as JSON for the fast tokenizer path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(MODELS_DIR))
    logger.info("✅ Tokenizer files saved to models/")
    return True


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    logger.info("Starting ONNX model export pipeline...\n")
    results["lstm_price"] = build_and_export_lstm()
    results["sentiment"] = export_sentiment_model()

    logger.info("\n=== Export Summary ===")
    for name, ok in results.items():
        status = "✅ Success" if ok else "❌ Failed"
        logger.info(f"  {name}: {status}")

    if not all(results.values()):
        logger.warning("\nSome exports failed. Check the errors above.")
        sys.exit(1)

    logger.info("\nAll done! Commit the files in models/ and redeploy to Render.")
    logger.info("  git add models/\n  git commit -m 'Add ONNX models'\n  git push")
