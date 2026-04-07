"""
sentiment.py - Sentiment analysis with graceful fallback.
Uses DistilBERT if transformers+torch are available, otherwise falls back to NLTK VADER.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Transformers and NLTK are loaded lazily to speed up app startup
_SENTIMENT_PIPELINE = None


def get_sentiment_pipeline():
    """Lazily load the transformers sentiment pipeline."""
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is not None:
        return _SENTIMENT_PIPELINE

    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading transformers sentiment pipeline (DistilBERT)...")
        _SENTIMENT_PIPELINE = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
        )
        return _SENTIMENT_PIPELINE
    except (ImportError, Exception) as e:
        logger.warning(f"Transformers sentiment not available: {e}. Using NLTK VADER.")
        return None


def get_vader_analyzer():
    """Lazily load NLTK VADER and check for lexicon."""
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # Check for lexicon without triggering a full download if possible
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except (LookupError, AttributeError):
            logger.info("Downloading NLTK vader_lexicon...")
            nltk.download('vader_lexicon', quiet=True)
            
        return SentimentIntensityAnalyzer()
    except Exception as e:
        logger.error(f"VADER initialization failed: {e}")
        return None


def analyze_sentiment(titles):
    """
    Analyze sentiment of a list of headlines.
    Returns: 'Positive', 'Negative', or 'Neutral'
    """
    if not titles:
        return "Neutral"

    # Strategy 1: Try Transformers
    pipeline = get_sentiment_pipeline()
    if pipeline is not None:
        try:
            sentiments = [pipeline(title)[0]['label'] for title in titles[:5]]
            positive = sum(1 for s in sentiments if s == "POSITIVE")
            negative = sum(1 for s in sentiments if s == "NEGATIVE")
            return "Positive" if positive > negative else "Negative" if negative > positive else "Neutral"
        except Exception as e:
            logger.warning(f"Transformers analysis failed: {e}")

    # Strategy 2: Fallback to VADER
    sia = get_vader_analyzer()
    if sia:
        try:
            scores = [sia.polarity_scores(title)['compound'] for title in titles[:5]]
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