"""
stock_recommender.py - Pipeline Orchestrator
Runs the full pipeline: Scrape News → Score Impact → Extract Stocks → Analyze → Recommend
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime

from news_scraper import scrape_all_news
from impact_scorer import score_and_extract_stocks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def score_recommendation_components(impact_score, sentiment, technical_data):
    """Build a more explicit composite score from news, forecast, technical, and risk inputs."""
    sentiment_bonus = {"BULLISH": 10, "NEUTRAL": 2, "BEARISH": -10}.get(str(sentiment).upper(), 0)
    components = {
        "news": 0.0,
        "forecast": 0.0,
        "technical": 0.0,
        "risk": 0.0,
        "sentiment": sentiment_bonus,
    }
    reasons = []

    components["news"] = clamp((float(impact_score) - 50.0) / 50.0, -1.0, 1.0) * 30
    if impact_score >= 65:
        reasons.append(f"strong news impact ({impact_score})")
    elif impact_score >= 50:
        reasons.append(f"moderate news impact ({impact_score})")
    else:
        reasons.append(f"weak news impact ({impact_score})")

    if technical_data is None:
        total_score = sum(components.values())
        return total_score, components, reasons

    pred_change = float(technical_data.get("pred_change_pct", 0.0))
    confidence = float(technical_data.get("confidence", 0.0))
    rsi = float(technical_data.get("rsi", 50.0))
    volatility = float(technical_data.get("volatility", 0.0))

    components["forecast"] = clamp(pred_change / 10.0, -1.0, 1.0) * 28
    if pred_change > 0:
        reasons.append(f"forecast points up ({pred_change:+.1f}%)")
    else:
        reasons.append(f"forecast points down ({pred_change:+.1f}%)")

    rsi_signal = 0.0
    if rsi < 30:
        rsi_signal = 1.0
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi < 50:
        rsi_signal = 0.45
        reasons.append(f"RSI supportive ({rsi:.0f})")
    elif rsi > 70:
        rsi_signal = -0.8
        reasons.append(f"RSI overbought ({rsi:.0f})")
    else:
        rsi_signal = -0.1
        reasons.append(f"RSI neutral ({rsi:.0f})")
    components["technical"] = rsi_signal * 16 + ((confidence - 0.5) * 16)

    volatility_penalty = clamp(volatility / max(float(technical_data.get("last_price", 1.0)), 1.0), 0.0, 0.25)
    components["risk"] = ((confidence - volatility_penalty) - 0.4) * 20
    reasons.append(f"model confidence {confidence:.0%}")
    reasons.append(f"volatility context {volatility:.2f}")

    total_score = sum(components.values())
    return total_score, components, reasons


def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    default_config = {
        "forecast_days": 30,
        "sequence_length": 60,
        "chart_style": "dark",
        "volume_overlay": True,
        "prediction_range_percent": 20,
        "impact_score_threshold": 55,
        "max_stocks_to_analyze": 10,
        "news_sources": ["livemint", "google_news"],
        "use_ml_model": False,
        "newsapi_key": ""
    }
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults for any missing keys
            for key, val in default_config.items():
                if key not in config:
                    config[key] = val
            return config
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
    return default_config


def load_ml_model():
    """
    Try to load the pre-trained ensemble ML model for sentiment.
    Returns (model, vectorizer) or (None, None) if not available.
    """
    try:
        from src.models import load_model_and_vectorizer
        model, vectorizer = load_model_and_vectorizer()
        logger.info("Loaded pre-trained ML model for sentiment analysis")
        return model, vectorizer
    except (FileNotFoundError, ImportError, Exception) as e:
        logger.info(f"ML model not available, using VADER: {e}")
        return None, None


def analyze_stock_technical(ticker, forecast_days=30):
    """
    Run TradeWise-style technical analysis on a single stock.
    
    Returns dict with analysis results or None if data not available.
    """
    try:
        from data import fetch_stock_data, fetch_realtime_data
        from indicators import calculate_indicators
        from model import predict_lstm
        from risk import assess_risk
        from backtest import backtest_strategy

        logger.info(f"Running technical analysis on {ticker}...")

        # Fetch historical data
        historical = fetch_stock_data(ticker, period="2y")
        if historical is None or historical.empty:
            logger.warning(f"No data for {ticker}")
            return None

        # Technical indicators
        indicators = calculate_indicators(historical)
        if not indicators:
            return None

        # Real-time data
        realtime = fetch_realtime_data(ticker)

        # LSTM prediction
        result = predict_lstm(historical['Close'].values, forecast_days=forecast_days)
        if isinstance(result, dict):
            predictions = result['predictions']
            confidence = result['confidence']
        else:
            predictions = result
            confidence = 0.5

        # Risk assessment
        risk = assess_risk(historical, predictions)
        
        # Skip heavy rolling-window backtest on Recommendation pipeline to prevent Gateway Timeout
        backtest_30 = {"profit": 0, "accuracy": 0, "buy_hold": 0}

        last_price = historical['Close'].iloc[-1]
        pred_direction = "LONG" if predictions[-1] > last_price else "SHORT"
        pred_change = ((predictions[-1] - last_price) / last_price) * 100

        # RSI status
        rsi = indicators.get('RSI', 50)
        if rsi > 70:
            rsi_status = "Overbought"
        elif rsi < 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Neutral"

        return {
            'ticker': ticker,
            'last_price': round(last_price, 2),
            'realtime_price': realtime.get('last_price', 'N/A'),
            'predicted_price': round(predictions[-1], 2),
            'pred_direction': pred_direction,
            'pred_change_pct': round(pred_change, 2),
            'confidence': round(confidence, 4),
            'rsi': round(rsi, 2),
            'rsi_status': rsi_status,
            'ma50': round(indicators.get('MA50', 0), 2),
            'macd': round(indicators.get('MACD', 0), 4),
            'volatility': round(risk.get('volatility', 0), 2),
            'stop_loss': round(risk.get('stop_loss', 0), 2),
            'backtest_30_profit': round(float(backtest_30.get('profit', 0)), 2),
            'backtest_30_accuracy': round(float(backtest_30.get('accuracy', 0)), 2),
            'backtest_30_buy_hold': round(float(backtest_30.get('buy_hold', 0)), 2),
            'historical': historical,
            'predictions': predictions,
            'indicators': indicators,
        }

    except Exception as e:
        logger.error(f"Technical analysis failed for {ticker}: {e}")
        return None


def generate_recommendation(impact_score, sentiment, technical_data):
    """
    Generate BUY/HOLD/AVOID recommendation based on impact score + technical analysis.
    
    Logic:
    - BUY: Impact score > 60 AND (positive prediction OR RSI < 70) AND sentiment is BULLISH
    - AVOID: Impact score < 40 OR (negative prediction AND RSI > 70) OR sentiment is BEARISH
    - HOLD: Everything else
    """
    score, components, reasons = score_recommendation_components(impact_score, sentiment, technical_data)
    reason_text = ' | '.join(reasons)
    component_text = (
        f"composite={score:.1f} "
        f"(news={components['news']:.1f}, forecast={components['forecast']:.1f}, "
        f"technical={components['technical']:.1f}, risk={components['risk']:.1f}, sentiment={components['sentiment']:.1f})"
    )

    if technical_data is None:
        if score >= 18:
            return 'BUY', f"{reason_text} | {component_text}"
        if score <= -8:
            return 'AVOID', f"{reason_text} | {component_text}"
        return 'HOLD', f"{reason_text} | {component_text}"

    if score >= 32:
        return 'BUY', f"{reason_text} | {component_text}"
    if score >= 8:
        return 'HOLD', f"{reason_text} | {component_text}"
    return 'AVOID', f"{reason_text} | {component_text}"


def get_recommendations(progress_callback=None):
    """
    Full pipeline: Scrape → Score → Analyze → Recommend
    
    Args:
        progress_callback: Optional function(message) for progress updates
    
    Returns:
        dict with:
            'recommendations': DataFrame of stocks with recommendations
            'all_articles': DataFrame of all scraped articles with sentiment
            'stocks_found': DataFrame of stocks found in news with impact scores
    """
    config = load_config()

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    log("Step 1/4: Scraping financial news...")
    news_df = scrape_all_news()
    if news_df.empty:
        log("No news articles found. Try again later.")
        return {'recommendations': pd.DataFrame(), 'all_articles': pd.DataFrame(), 'stocks_found': pd.DataFrame()}

    log(f"Collected {len(news_df)} news articles")

    # Try loading ML model
    model, vectorizer = None, None
    if config.get('use_ml_model', False):
        model, vectorizer = load_ml_model()

    log("Step 2/4: Analyzing sentiment & extracting stocks...")
    result = score_and_extract_stocks(news_df, model=model, vectorizer=vectorizer)
    articles_df = result['articles']
    stocks_df = result['stocks']

    if stocks_df.empty:
        log("No stocks identified in news articles.")
        return {'recommendations': pd.DataFrame(), 'all_articles': articles_df, 'stocks_found': stocks_df}

    log(f"Found {len(stocks_df)} stocks in news")

    # Filter by impact score threshold
    threshold = config.get('impact_score_threshold', 55)
    max_stocks = config.get('max_stocks_to_analyze', 10)

    # Take top stocks (both high-impact bullish AND high-impact bearish for AVOID signals)
    stocks_to_analyze = stocks_df.head(max_stocks)

    log(f"Step 3/4: Running technical analysis on top {len(stocks_to_analyze)} stocks...")

    recommendations = []
    for i, (_, stock_row) in enumerate(stocks_to_analyze.iterrows()):
        ticker = stock_row['ticker']
        impact_score = stock_row['impact_score']
        sentiment = stock_row['sentiment']

        log(f"  Analyzing {ticker} ({i+1}/{len(stocks_to_analyze)})...")

        try:
            technical = analyze_stock_technical(ticker, forecast_days=config.get('forecast_days', 30))
        except Exception as e:
            logger.warning(f"Technical analysis failed for {ticker}: {e}")
            technical = None

        recommendation, reasons = generate_recommendation(impact_score, sentiment, technical)
        composite_score, score_breakdown, _ = score_recommendation_components(impact_score, sentiment, technical)

        rec_data = {
            'ticker': ticker,
            'impact_score': impact_score,
            'news_sentiment': sentiment,
            'article_count': stock_row['article_count'],
            'recommendation': recommendation,
            'reasons': reasons,
            'composite_score': round(composite_score, 2),
            'score_breakdown': score_breakdown,
            'headlines': stock_row.get('headlines', ''),
        }

        if technical:
            rec_data.update({
                'last_price': technical['last_price'],
                'predicted_price': technical['predicted_price'],
                'pred_change_pct': technical['pred_change_pct'],
                'confidence': technical['confidence'],
                'rsi': technical['rsi'],
                'stop_loss': technical['stop_loss'],
                'backtest_30_profit': technical['backtest_30_profit'],
                'backtest_30_accuracy': technical['backtest_30_accuracy'],
                'backtest_30_buy_hold': technical['backtest_30_buy_hold'],
            })
        else:
            rec_data.update({
                'last_price': None,
                'predicted_price': None,
                'pred_change_pct': None,
                'confidence': None,
                'rsi': None,
                'stop_loss': None,
                'backtest_30_profit': None,
                'backtest_30_accuracy': None,
                'backtest_30_buy_hold': None,
            })

        recommendations.append(rec_data)

    log("Step 4/4: Generating final recommendations...")

    rec_df = pd.DataFrame(recommendations)

    # Sort: BUY first, then HOLD, then AVOID. Within each, sort by impact score
    rec_order = {'BUY': 0, 'HOLD': 1, 'AVOID': 2}
    rec_df['_sort'] = rec_df['recommendation'].map(rec_order)
    rec_df = rec_df.sort_values(['_sort', 'impact_score'], ascending=[True, False]).drop('_sort', axis=1)
    rec_df = rec_df.reset_index(drop=True)

    log(f"Complete! {len(rec_df[rec_df['recommendation'] == 'BUY'])} BUY, "
        f"{len(rec_df[rec_df['recommendation'] == 'HOLD'])} HOLD, "
        f"{len(rec_df[rec_df['recommendation'] == 'AVOID'])} AVOID")

    return {
        'recommendations': rec_df,
        'all_articles': articles_df,
        'stocks_found': stocks_df,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Stock Recommender - Full Pipeline Test")
    print("=" * 70)

    results = get_recommendations(progress_callback=print)

    if not results['recommendations'].empty:
        print("\n" + "=" * 70)
        print("FINAL RECOMMENDATIONS")
        print("=" * 70)
        display_cols = ['ticker', 'recommendation', 'impact_score', 'news_sentiment',
                       'last_price', 'predicted_price', 'pred_change_pct', 'rsi']
        available_cols = [c for c in display_cols if c in results['recommendations'].columns]
        print(results['recommendations'][available_cols].to_string())
    else:
        print("\nNo recommendations generated.")
