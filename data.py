import yfinance as yf
import pandas as pd
from sentiment import analyze_sentiment
import requests
import logging


logger = logging.getLogger(__name__)


def fetch_json_with_retries(url, *, params=None, headers=None, timeout=10, attempts=2):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            logger.warning("JSON fetch attempt %s/%s failed for %s: %s", attempt, attempts, url, exc)
    raise last_error

def fetch_stock_data(ticker, period="5y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logger.warning("Error fetching historical data for %s: %s", ticker, e)
        return pd.DataFrame()

def fetch_realtime_data(ticker):
    historical = fetch_stock_data(ticker, period="1d")
    if not historical.empty:
        last_price = historical['Close'].iloc[-1]
        volume = historical['Volume'].iloc[-1]
        return {"last_price": last_price, "volume": volume}
    return {"last_price": "N/A", "volume": "N/A"}

def fetch_news_sentiment(ticker):
    """Fetch news sentiment for a ticker using NewsAPI or fallback to VADER."""
    try:
        import json, os
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        api_key = ""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get('newsapi_key', '')

        if api_key and api_key != "YOUR_API_KEY":
            payload = fetch_json_with_retries(
                "https://newsapi.org/v2/everything",
                params={
                    "q": ticker,
                    "language": "en",
                    "domains": "economictimes.indiatimes.com,moneycontrol.com",
                    "apiKey": api_key,
                },
                timeout=10,
                attempts=2,
            )
            articles = payload.get('articles', [])
            titles = [article['title'] for article in articles[:5]]
            return analyze_sentiment(titles)
        else:
            # Fallback: use VADER on the ticker name
            from impact_scorer import analyze_sentiment_vader
            from news_scraper import scrape_google_news as scrape_gn
            articles = scrape_gn(keyword=ticker.replace('.NS', ''), days=3)
            if articles:
                scores = [analyze_sentiment_vader(a['headline'])['compound'] for a in articles[:5]]
                avg = sum(scores) / len(scores) if scores else 0
                if avg > 0.05:
                    return "Positive"
                elif avg < -0.05:
                    return "Negative"
                else:
                    return "Neutral"
            return "Neutral"
    except Exception as e:
        logger.warning("Error fetching news sentiment for %s: %s", ticker, e)
        return "Neutral"


def fetch_news_sentiment_with_impact(impact_data):
    """
    Convert pre-computed impact data to sentiment string.
    Used by the merged pipeline to avoid re-fetching news.
    """
    if impact_data is None:
        return "Neutral"
    
    impact_score = impact_data.get('impact_score', 50)
    if impact_score >= 65:
        return "Strongly Positive"
    elif impact_score >= 55:
        return "Positive"
    elif impact_score >= 45:
        return "Neutral"
    elif impact_score >= 35:
        return "Negative"
    else:
        return "Strongly Negative"
