"""
sentiment.py - Sentiment analysis with graceful fallback.
Uses DistilBERT if transformers+torch are available, otherwise falls back to NLTK VADER.
"""

import logging

logger = logging.getLogger(__name__)

# Try importing transformers — fall back to VADER if unavailable
try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.info("transformers not available — using NLTK VADER for sentiment")


def analyze_sentiment(titles):
    """
    Analyze sentiment of a list of headlines.
    Returns: 'Positive', 'Negative', or 'Neutral'
    """
    if not titles:
        return "Neutral"

    if HAS_TRANSFORMERS:
        try:
            classifier = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
            )
            sentiments = [classifier(title)[0]['label'] for title in titles[:5]]
            positive = sum(1 for s in sentiments if s == "POSITIVE")
            negative = sum(1 for s in sentiments if s == "NEGATIVE")
            return "Positive" if positive > negative else "Negative" if negative > positive else "Neutral"
        except Exception as e:
            logger.warning(f"Transformers sentiment failed, falling back to VADER: {e}")

    # Fallback: NLTK VADER
    try:
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)

        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

        scores = [sia.polarity_scores(title)['compound'] for title in titles[:5]]
        avg = sum(scores) / len(scores) if scores else 0

        if avg > 0.05:
            return "Positive"
        elif avg < -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"VADER sentiment also failed: {e}")
        return "Neutral"