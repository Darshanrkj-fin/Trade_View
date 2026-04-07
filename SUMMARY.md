# Smart Stock Analyzer

Smart Stock Analyzer is a comprehensive, open-source quantitative stock analysis desktop application built in Python using PyQt5. The application seamlessly integrates historical market data, real-time pricing, natural language processing for sentiment analysis, and machine learning models to provide deep insights into the Indian Stock Market (NSE/BSE).

## Core Features

### 1. Unified Analysis Pipeline
The project merges previously disparate tools into a single, cohesive interface:
- **News Scraping & Sentiment Analysis**: Automatically fetches recent financial news headlines via Beautiful Soup and processes them using NLTK VADER to determine the market sentiment of a stock.
- **Machine Learning Price Predictions**: Uses LSTM (Long Short-Term Memory) Neural Networks backed by TensorFlow to predict forward price action, with a seamless fallback to `scikit-learn` Linear Regression models to guarantee compatibility across diverse local environments.
- **Ensemble Impact Scoring**: Calculates a dynamic "Impact Score" that quantifies the combined momentum of technical indicators and breaking news.

### 2. Institutional Technical Indicators
Beyond standard moving averages and RSI, TradeWise features natively programmed institutional-grade metrics:
- **Liquidity Sweeps**: Algorithmically detects "stop hunts" where price runs past established swing highs or lows and immediately reverses.
- **Order Flow Analysis**: Estimates intraday buying versus selling pressure by analyzing the close relative to the high-low range.
- **Anchored VWAP**: Plots the true average price over a rolling anchor window.
- **Volume Profile**: Scans historical price distributions to identify key support/resistance zones like the Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).

### 3. Quantitative Portfolio Optimization
The application features a modern Portfolio Builder that executes a 4-step quantitative pipeline:
1. **Deep RankNet**: Extracts momentum and volatility patterns to generate a pairwise alpha selection score.
2. **Markowitz Optimization (MVO)**: Analyzes the covariance matrix of expected returns to mathematically find the Efficient Frontier—the exact weighting that minimizes risk while maximizing returns.
3. **CAPM (Capital Asset Pricing Model)**: Computes the portfolio's Beta relative to the NIFTY50 index to determine the baseline Required Return.
4. **Interactive Dashboard & Rebalancing**: Generates Plotly pie charts and an Efficient Frontier curve alongside actionable, color-coded `BUY`, `SELL`, and `HOLD` recommendations with exact Rupee amounts needed to rebalance back to the mathematically optimal state.

### 4. Out-of-Sample Backtesting
The built-in backtester does not "cheat" by looking at future data. It dynamically splits historical price data, generates mock predictions based firmly on the past, evaluates Model Accuracy via Mean Absolute Percentage Error (MAPE), and runs a simulated trading competition to determine if the AI strategy beats a traditional "Buy & Hold" approach.

## Tech Stack
- **Frontend / GUI**: PyQt5, PyQtWebEngine
- **Interactive Charting**: Plotly
- **Data Acquisition**: `yfinance`, `beautifulsoup4`, `requests`
- **Machine Learning**: `tensorflow` (Optional), `scikit-learn`
- **Core Math / Finance**: `numpy`, `pandas`, `scipy`, `ta` (Technical Analysis Library)
- **Natural Language Processing**: `nltk`

## Architecture Overview
- `main.py`: The graphical user interface entry point and async worker manager.
- `model.py`: Handlers for LSTM training/predictions and fallback ML models.
- `indicators.py`: Financial mathematics engine that powers standard and advanced analytical plots.
- `portfolio_optimizer.py`: The quantitative portfolio management backend utilizing `scipy.optimize`.
- `backtest.py`: Out-of-sample prediction simulator and accuracy tracker.
- `data.py`: Centralized handlers for scraping and API data acquisition.
- `sentiment.py`: Text processing models utilizing NLTK.
