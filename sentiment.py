"""
sentiment.py - Sentiment analysis with ONNX Runtime inference and NLTK VADER fallback.
Uses a pre-converted DistilBERT ONNX model if available, otherwise falls back to NLTK VADER.
No PyTorch or Transformers runtime dependency required.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_distilbert.onnx"
SENTIMENT_VOCAB_PATH = MODELS_DIR / "sentiment_vocab.json"

# Cached ONNX session (loaded once, reused across requests)
_ONNX_SESSION = None
_TOKENIZER = None

# DistilBERT SST-2 label mapping (index → label)
_LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}


def _load_onnx_session():
    """Load ONNX Runtime session once and cache it."""
    global _ONNX_SESSION
    if _ONNX_SESSION is not None:
        return _ONNX_SESSION

    if not SENTIMENT_MODEL_PATH.exists():
        logger.info("ONNX sentiment model not found. Will use NLTK VADER.")
        return None

    try:
        import onnxruntime as ort
        _ONNX_SESSION = ort.InferenceSession(
            str(SENTIMENT_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX sentiment model loaded successfully.")
        return _ONNX_SESSION
    except Exception as e:
        logger.warning(f"Failed to load ONNX sentiment model: {e}. Falling back to VADER.")
        return None


def _load_tokenizer():
    """Load fast tokenizer (tokenizers library) once and cache it."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    if not SENTIMENT_MODEL_PATH.exists():
        return None

    try:
        from tokenizers import Tokenizer
        if SENTIMENT_VOCAB_PATH.exists():
            _TOKENIZER = Tokenizer.from_file(str(SENTIMENT_VOCAB_PATH))
            logger.info("Fast tokenizer loaded from vocab file.")
            return _TOKENIZER
    except ImportError:
        pass

    # Fallback: use transformers tokenizer (tokenization only, no torch needed)
    try:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        logger.info("Transformers tokenizer loaded (no torch needed).")
        return _TOKENIZER
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}")
        return None


def _run_onnx_sentiment(text: str) -> str | None:
    """
    Run a single text through the ONNX DistilBERT session.
    Returns 'POSITIVE' or 'NEGATIVE', or None on failure.
    """
    session = _load_onnx_session()
    tokenizer = _load_tokenizer()

    if session is None or tokenizer is None:
        return None

    try:
        # Check if tokenizer is from 'tokenizers' (no __call__) or 'transformers'
        if hasattr(tokenizer, "encode") and not hasattr(tokenizer, "__call__"):
            # tokenizers.Tokenizer path
            tokenizer.enable_truncation(max_length=128)
            tokenizer.enable_padding(length=128)
            encoded = tokenizer.encode(text)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        else:
            # transformers.AutoTokenizer path
            encoded = tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=128,
                padding="max_length",
            )
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if "token_type_ids" in [inp.name for inp in session.get_inputs()]:
            inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        logits = session.run(None, inputs)[0]
        label_idx = int(np.argmax(logits, axis=-1)[0])
        return _LABEL_MAP.get(label_idx, "NEGATIVE")
    except Exception as e:
        logger.warning(f"ONNX inference error: {e}")
        return None


def get_vader_analyzer():
    """Load NLTK VADER SentimentIntensityAnalyzer, downloading lexicon if needed."""
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except (LookupError, AttributeError):
            logger.info("Downloading NLTK vader_lexicon...")
            nltk.download("vader_lexicon", quiet=True)

        return SentimentIntensityAnalyzer()
    except Exception as e:
        logger.error(f"VADER initialization failed: {e}")
        return None


def analyze_sentiment(titles):
    """
    Analyze sentiment of a list of headlines.
    Returns: 'Positive', 'Negative', or 'Neutral'

    Priority:
      1. ONNX DistilBERT (models/sentiment_distilbert.onnx) — fast, no Torch
      2. NLTK VADER — lightweight keyword-based fallback
    """
    if not titles:
        return "Neutral"

    # Strategy 1: ONNX DistilBERT
    session = _load_onnx_session()
    if session is not None:
        try:
            results = [_run_onnx_sentiment(t) for t in titles[:5]]
            results = [r for r in results if r is not None]
            if results:
                positive = sum(1 for r in results if r == "POSITIVE")
                negative = sum(1 for r in results if r == "NEGATIVE")
                return "Positive" if positive > negative else "Negative" if negative > positive else "Neutral"
        except Exception as e:
            logger.warning(f"ONNX sentiment analysis failed: {e}")

    # Strategy 2: NLTK VADER fallback
    sia = get_vader_analyzer()
    if sia:
        try:
            scores = [sia.polarity_scores(t)["compound"] for t in titles[:5]]
            avg = sum(scores) / len(scores) if scores else 0
            if avg > 0.05:
                return "Positive"
            elif avg < -0.05:
                return "Negative"
            else:
                return "Neutral"
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")

    return "Neutral"