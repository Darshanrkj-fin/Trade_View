"""
impact_scorer.py - News Sentiment & Impact Score Module
Analyzes news articles for sentiment and extracts mentioned Indian stocks,
then computes an impact score per stock.
"""

import re
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────────
# Indian Stock Name → NSE Ticker Mapping
# ─────────────────────────────────────────────────
STOCK_NAME_TO_TICKER = {
    # Nifty 50 & Major Large Caps
    'reliance': 'RELIANCE.NS', 'reliance industries': 'RELIANCE.NS', 'ril': 'RELIANCE.NS',
    'tcs': 'TCS.NS', 'tata consultancy': 'TCS.NS',
    'hdfc bank': 'HDFCBANK.NS', 'hdfcbank': 'HDFCBANK.NS',
    'infosys': 'INFY.NS', 'infy': 'INFY.NS',
    'icici bank': 'ICICIBANK.NS', 'icici': 'ICICIBANK.NS',
    'hindustan unilever': 'HINDUNILVR.NS', 'hul': 'HINDUNILVR.NS',
    'sbi': 'SBIN.NS', 'state bank': 'SBIN.NS', 'state bank of india': 'SBIN.NS',
    'bharti airtel': 'BHARTIARTL.NS', 'airtel': 'BHARTIARTL.NS',
    'itc': 'ITC.NS',
    'kotak': 'KOTAKBANK.NS', 'kotak mahindra': 'KOTAKBANK.NS', 'kotak bank': 'KOTAKBANK.NS',
    'larsen': 'LT.NS', 'l&t': 'LT.NS', 'larsen & toubro': 'LT.NS',
    'asian paints': 'ASIANPAINT.NS',
    'axis bank': 'AXISBANK.NS', 'axis': 'AXISBANK.NS',
    'maruti': 'MARUTI.NS', 'maruti suzuki': 'MARUTI.NS',
    'wipro': 'WIPRO.NS',
    'hcl tech': 'HCLTECH.NS', 'hcl technologies': 'HCLTECH.NS',
    'ultratech': 'ULTRACEMCO.NS', 'ultratech cement': 'ULTRACEMCO.NS',
    'titan': 'TITAN.NS', 'titan company': 'TITAN.NS',
    'bajaj finance': 'BAJFINANCE.NS',
    'sun pharma': 'SUNPHARMA.NS', 'sun pharmaceutical': 'SUNPHARMA.NS',
    'tata motors': 'TATAMOTORS.NS',
    'adani enterprises': 'ADANIENT.NS', 'adani': 'ADANIENT.NS',
    'tata steel': 'TATASTEEL.NS',
    'ntpc': 'NTPC.NS',
    'power grid': 'POWERGRID.NS', 'powergrid': 'POWERGRID.NS',
    'tech mahindra': 'TECHM.NS',
    'bajaj finserv': 'BAJAJFINSV.NS',
    'hindalco': 'HINDALCO.NS',
    'ongc': 'ONGC.NS',
    'coal india': 'COALINDIA.NS',
    'nestle': 'NESTLEIND.NS', 'nestle india': 'NESTLEIND.NS',
    'dr reddy': 'DRREDDY.NS', "dr reddy's": 'DRREDDY.NS',
    'cipla': 'CIPLA.NS',
    'britannia': 'BRITANNIA.NS',
    'divis lab': 'DIVISLAB.NS', "divi's lab": 'DIVISLAB.NS',
    'eicher motors': 'EICHERMOT.NS', 'eicher': 'EICHERMOT.NS',
    'grasim': 'GRASIM.NS',
    'hero motocorp': 'HEROMOTOCO.NS', 'hero moto': 'HEROMOTOCO.NS',
    'indusind bank': 'INDUSINDBK.NS', 'indusind': 'INDUSINDBK.NS',
    'jsw steel': 'JSWSTEEL.NS',
    'm&m': 'M&M.NS', 'mahindra': 'M&M.NS', 'mahindra & mahindra': 'M&M.NS',
    'tata consumer': 'TATACONSUM.NS',
    'upl': 'UPL.NS',
    'vedanta': 'VEDL.NS',
    'zomato': 'ZOMATO.NS',
    'paytm': 'PAYTM.NS',
    'nykaa': 'NYKAA.NS',
    'adani ports': 'ADANIPORTS.NS',
    'adani green': 'ADANIGREEN.NS',
    'adani power': 'ADANIPOWER.NS',
    'bajaj auto': 'BAJAJ-AUTO.NS',
    'bhel': 'BHEL.NS',
    'bpcl': 'BPCL.NS',
    'gail': 'GAIL.NS',
    'godrej': 'GODREJCP.NS', 'godrej consumer': 'GODREJCP.NS',
    'havells': 'HAVELLS.NS',
    'hdfc life': 'HDFCLIFE.NS',
    'icici prudential': 'ICICIPRULI.NS',
    'ioc': 'IOC.NS', 'indian oil': 'IOC.NS',
    'irctc': 'IRCTC.NS',
    'pidilite': 'PIDILITIND.NS',
    'sbi life': 'SBILIFE.NS',
    'siemens': 'SIEMENS.NS',
    'tata power': 'TATAPOWER.NS',
    'tata elxsi': 'TATAELXSI.NS',
    'torrent pharma': 'TORNTPHARM.NS',
    'varun beverages': 'VBL.NS',
    'bank of baroda': 'BANKBARODA.NS',
    'canara bank': 'CANBK.NS',
    'pnb': 'PNB.NS', 'punjab national bank': 'PNB.NS',
    'bandhan bank': 'BANDHANBNK.NS',
    'yes bank': 'YESBANK.NS',
    'idfc first': 'IDFCFIRSTB.NS', 'idfc first bank': 'IDFCFIRSTB.NS',
}

# Add NSE ticker pattern match (e.g., "RELIANCE.NS" directly in text)
NSE_TICKER_PATTERN = re.compile(r'\b([A-Z][A-Z0-9&-]{1,15})\.NS\b')


def analyze_sentiment_vader(text):
    """
    Analyze sentiment using NLTK VADER.
    Returns dict with neg, neu, pos, compound scores.
    """
    if not text or not isinstance(text, str):
        return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
    return sia.polarity_scores(text)


def try_ml_sentiment(text, model=None, vectorizer=None):
    """
    Try to use the trained ensemble ML model for sentiment prediction.
    Falls back to VADER if model is not available.
    """
    if model is not None and vectorizer is not None:
        try:
            from src.preprocessing import clean_text
            cleaned = clean_text(text)
            X = vectorizer.transform([cleaned])
            prediction = model.predict(X)[0]
            return prediction  # 'POSITIVE' or 'NEGATIVE'
        except Exception as e:
            logger.debug(f"ML model prediction failed: {e}")

    # Fallback to VADER
    scores = analyze_sentiment_vader(text)
    if scores['compound'] >= 0.05:
        return 'POSITIVE'
    elif scores['compound'] <= -0.05:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'


def extract_stocks_from_text(text):
    """
    Extract Indian stock names/tickers mentioned in a news headline or article text.
    Returns a list of NSE ticker symbols.
    """
    if not text:
        return []

    found_tickers = set()

    # 1. Direct NSE ticker pattern match
    for match in NSE_TICKER_PATTERN.finditer(text):
        found_tickers.add(match.group(0))

    # 2. Company name matching (case-insensitive)
    text_lower = text.lower()
    for name, ticker in STOCK_NAME_TO_TICKER.items():
        # Use word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, text_lower):
            found_tickers.add(ticker)

    return list(found_tickers)


def calculate_impact_score(positive_count, negative_count):
    """
    Calculate news impact score on a 0-100 scale.
    50 = neutral, >50 = bullish, <50 = bearish.
    """
    total = positive_count + negative_count
    if total == 0:
        return 50.0  # Neutral

    raw_score = ((positive_count - negative_count) / total) * 100
    # Normalize from [-100, 100] to [0, 100]
    normalized = ((raw_score + 100) / 200) * 100
    return round(normalized, 2)


def score_and_extract_stocks(news_df, model=None, vectorizer=None):
    """
    Main function: Score news articles and extract mentioned stocks.
    
    Args:
        news_df: DataFrame from news_scraper with columns: headline, text, link, source
        model: Optional trained ML model
        vectorizer: Optional fitted TF-IDF vectorizer
    
    Returns:
        dict: {
            'articles': DataFrame with sentiment scores added,
            'stocks': DataFrame with columns: ticker, impact_score, sentiment, article_count, headlines
        }
    """
    if news_df is None or news_df.empty:
        logger.warning("No news articles to score")
        return {
            'articles': pd.DataFrame(),
            'stocks': pd.DataFrame(columns=['ticker', 'impact_score', 'sentiment', 'article_count', 'headlines'])
        }

    # Score each article
    sentiments = []
    compound_scores = []
    extracted_tickers_list = []

    for _, row in news_df.iterrows():
        # Combine headline and text for analysis
        full_text = f"{row.get('headline', '')} {row.get('text', '')}"

        # Sentiment analysis
        vader_scores = analyze_sentiment_vader(full_text)
        sentiment_label = try_ml_sentiment(full_text, model, vectorizer)

        sentiments.append(sentiment_label)
        compound_scores.append(vader_scores['compound'])

        # Extract stocks mentioned
        tickers = extract_stocks_from_text(full_text)
        extracted_tickers_list.append(tickers)

    news_df = news_df.copy()
    news_df['sentiment'] = sentiments
    news_df['compound_score'] = compound_scores
    news_df['mentioned_tickers'] = extracted_tickers_list

    # Aggregate by stock
    stock_data = {}
    for idx, row in news_df.iterrows():
        for ticker in row['mentioned_tickers']:
            if ticker not in stock_data:
                stock_data[ticker] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'scores': [],
                    'headlines': []
                }

            stock_data[ticker]['scores'].append(row['compound_score'])
            stock_data[ticker]['headlines'].append(row['headline'])

            if row['sentiment'] == 'POSITIVE':
                stock_data[ticker]['positive'] += 1
            elif row['sentiment'] == 'NEGATIVE':
                stock_data[ticker]['negative'] += 1
            else:
                stock_data[ticker]['neutral'] += 1

    # Build stock summary DataFrame
    stock_rows = []
    for ticker, data in stock_data.items():
        impact = calculate_impact_score(data['positive'], data['negative'])
        avg_score = np.mean(data['scores']) if data['scores'] else 0
        total_articles = data['positive'] + data['negative'] + data['neutral']

        if avg_score > 0.05:
            sentiment = 'BULLISH'
        elif avg_score < -0.05:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'

        stock_rows.append({
            'ticker': ticker,
            'impact_score': impact,
            'avg_compound': round(avg_score, 4),
            'sentiment': sentiment,
            'positive_count': data['positive'],
            'negative_count': data['negative'],
            'article_count': total_articles,
            'headlines': ' | '.join(data['headlines'][:5])  # Top 5 headlines
        })

    stocks_df = pd.DataFrame(stock_rows)
    if not stocks_df.empty:
        stocks_df = stocks_df.sort_values('impact_score', ascending=False).reset_index(drop=True)

    logger.info(f"Scored {len(news_df)} articles, found {len(stocks_df)} unique stocks")

    return {
        'articles': news_df,
        'stocks': stocks_df
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Impact Scorer - Testing Mode")
    print("=" * 60)

    # Test with sample headlines
    test_data = pd.DataFrame({
        'headline': [
            'Reliance Industries shares jump 5% on strong quarterly results',
            'TCS wins major deal worth $2 billion from European client',
            'HDFC Bank faces regulatory scrutiny over loan practices',
            'Infosys cuts revenue guidance amid global slowdown',
            'Tata Motors EV sales surge 200% in latest quarter',
            'Adani Group stocks crash after fraud allegations',
            'SBI posts record profit, declares highest-ever dividend',
            'Wipro announces layoffs amid cost-cutting drive',
        ],
        'text': [''] * 8,
        'link': ['https://example.com'] * 8,
        'source': ['Test'] * 8,
        'timestamp': [pd.Timestamp.now().isoformat()] * 8
    })

    result = score_and_extract_stocks(test_data)
    print("\nArticle Sentiments:")
    print(result['articles'][['headline', 'sentiment', 'compound_score']].to_string())
    print("\nStock Impact Scores:")
    print(result['stocks'].to_string())
