from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from benchmark import benchmark_forecasts
from backtest import backtest_strategy
from data import fetch_news_sentiment, fetch_realtime_data, fetch_stock_data
from indicators import calculate_indicators
from model import predict_lstm
from portfolio_optimizer import run_portfolio_optimization
from risk import assess_risk
from stock_recommender import get_recommendations
from terminal_app.cache import TTLCache
from terminal_app.serializers import frame_records, serialize_value
from terminal_app.settings import settings
from terminal_app.validators import validate_forecast_days, validate_holdings, validate_ticker


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
BACKTEST_PERIODS = (30, 60, 90)

market_status_cache = TTLCache(ttl_seconds=settings.market_status_ttl)
analysis_cache = TTLCache(ttl_seconds=settings.analysis_ttl)
news_cache = TTLCache(ttl_seconds=settings.news_ttl)
benchmark_cache = TTLCache(ttl_seconds=settings.analysis_ttl)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def term_dates(last_date: pd.Timestamp, forecast_days: int) -> list[str]:
    dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq="D")[1:]
    return [ts.isoformat() for ts in dates.to_pydatetime()]


def get_frontend_config() -> dict[str, Any]:
    config = load_config()
    return {
        "forecast_days": config.get("forecast_days", 30),
        "impact_score_threshold": config.get("impact_score_threshold", 55),
        "max_stocks_to_analyze": config.get("max_stocks_to_analyze", 10),
        "term_options": [
            {"label": "Short Term", "value": 7},
            {"label": "Medium Term", "value": 30},
            {"label": "Long Term", "value": 90},
        ],
    }


def confidence_band(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def news_reason_summary(row: pd.Series) -> str:
    recommendation = str(row.get("recommendation", "")).upper()
    sentiment = row.get("news_sentiment", "mixed")
    impact = row.get("impact_score", "n/a")
    pred_change = row.get("pred_change_pct")
    pred_text = "forecast unavailable" if pd.isna(pred_change) else f"forecast {float(pred_change):.1f}%"

    if recommendation == "BUY":
        return f"Positive news flow, impact {impact}, {pred_text}."
    if recommendation == "AVOID":
        return f"Weak setup or negative news flow, impact {impact}, {pred_text}."
    return f"Mixed signal stack with {sentiment} sentiment, impact {impact}, {pred_text}."


def build_news_performance_summary(rec_df: pd.DataFrame) -> dict[str, Any]:
    if rec_df.empty:
        return {
            "coverage": 0,
            "buy_avg_backtest_profit": None,
            "hold_avg_backtest_profit": None,
            "avoid_avg_backtest_profit": None,
            "buy_avg_accuracy": None,
            "hold_avg_accuracy": None,
            "avoid_avg_accuracy": None,
        }

    performance_df = rec_df.copy()
    for column in ("backtest_30_profit", "backtest_30_accuracy", "backtest_30_buy_hold"):
        if column not in performance_df:
            performance_df[column] = np.nan

    def rec_avg(rec_name: str, column: str):
        subset = performance_df.loc[performance_df["recommendation"] == rec_name, column].dropna()
        return None if subset.empty else round(float(subset.mean()), 2)

    return {
        "coverage": int(performance_df["backtest_30_accuracy"].notna().sum()),
        "buy_avg_backtest_profit": rec_avg("BUY", "backtest_30_profit"),
        "hold_avg_backtest_profit": rec_avg("HOLD", "backtest_30_profit"),
        "avoid_avg_backtest_profit": rec_avg("AVOID", "backtest_30_profit"),
        "buy_avg_accuracy": rec_avg("BUY", "backtest_30_accuracy"),
        "hold_avg_accuracy": rec_avg("HOLD", "backtest_30_accuracy"),
        "avoid_avg_accuracy": rec_avg("AVOID", "backtest_30_accuracy"),
        "buy_avg_buy_hold": rec_avg("BUY", "backtest_30_buy_hold"),
        "hold_avg_buy_hold": rec_avg("HOLD", "backtest_30_buy_hold"),
        "avoid_avg_buy_hold": rec_avg("AVOID", "backtest_30_buy_hold"),
    }


def get_market_status() -> dict[str, Any]:
    cached = market_status_cache.get("market_status")
    if cached is not None:
        return cached

    indices = {
        "sensex": {"symbol": "^BSESN", "label": "SENSEX"},
        "nifty": {"symbol": "^NSEI", "label": "NIFTY"},
    }
    snapshot: dict[str, Any] = {}
    for key, meta in indices.items():
        try:
            ticker = yf.Ticker(meta["symbol"])
            data = ticker.history(period="1d")
        except Exception as exc:
            logger.warning("Failed to fetch market status for %s: %s", meta["symbol"], exc)
            data = pd.DataFrame()

        if data.empty:
            snapshot[key] = {"label": meta["label"], "available": False}
            continue
        current = float(data["Close"].iloc[-1])
        open_price = float(data["Open"].iloc[-1])
        change_pct = ((current - open_price) / open_price) * 100 if open_price else 0.0
        snapshot[key] = {
            "label": meta["label"],
            "symbol": meta["symbol"],
            "available": True,
            "current": round(current, 2),
            "open": round(open_price, 2),
            "change_pct": round(change_pct, 2),
            "direction": "up" if change_pct >= 0 else "down",
            "fetched_at": pd.Timestamp.utcnow().isoformat(),
        }
    return market_status_cache.set(
        "market_status",
        {"available": all(item.get("available") for item in snapshot.values()), "indices": snapshot},
    )


def build_analysis_payload(raw_ticker: object, raw_forecast_days: object) -> dict[str, Any]:
    ticker = validate_ticker(raw_ticker)
    forecast_days = validate_forecast_days(raw_forecast_days)
    cache_key = f"{ticker}:{forecast_days}"
    cached = analysis_cache.get(cache_key)
    if cached is not None:
        return cached

    historical = fetch_stock_data(ticker, period="2y")
    if historical.empty:
        raise ValueError(f"No historical data found for {ticker}")

    indicators = calculate_indicators(historical)
    realtime = fetch_realtime_data(ticker)
    news_sentiment = fetch_news_sentiment(ticker)
    prediction_result = predict_lstm(historical["Close"].values, forecast_days=forecast_days)

    if isinstance(prediction_result, dict):
        predictions = np.asarray(prediction_result["predictions"], dtype=float)
        confidence = float(prediction_result.get("confidence", 0.0))
        model_info = prediction_result.get("model_info", {})
    else:
        predictions = np.asarray(prediction_result, dtype=float)
        confidence = 0.5
        model_info = {"engine": "unknown", "label": "Unknown", "is_fallback": True}

    risk = assess_risk(historical, predictions)
    last_price = float(historical["Close"].iloc[-1])
    pred_change_pct = ((predictions[-1] - last_price) / last_price) * 100 if last_price else 0.0
    fetched_at = pd.Timestamp.utcnow().isoformat()
    model_quality = model_info.get("confidence_band") or confidence_band(confidence)

    backtests = []
    for days in BACKTEST_PERIODS:
        if len(historical) < days * 2:
            backtests.append({"period": days, "profit": 0, "buy_hold": 0, "accuracy": 0})
            continue
        result = backtest_strategy(historical, test_days=days)
        backtests.append({"period": days, **serialize_value(result)})

    indicator_snapshot = {
        "RSI": indicators.get("RSI"),
        "MA50": indicators.get("MA50"),
        "MA100": indicators.get("MA100"),
        "MA200": indicators.get("MA200"),
        "MACD": indicators.get("MACD"),
        "Bollinger": indicators.get("Bollinger"),
        "AVWAP": indicators.get("AVWAP"),
        "AVWAP_signal": indicators.get("AVWAP_signal"),
        "VP_POC": indicators.get("VP_POC"),
        "VP_signal": indicators.get("VP_signal"),
        "LS_signal": indicators.get("LS_signal"),
        "OF_signal": indicators.get("OF_signal"),
    }

    payload = {
        "ticker": ticker,
        "forecast_days": forecast_days,
        "last_price": round(last_price, 2),
        "realtime_price": serialize_value(realtime.get("last_price")),
        "volume": serialize_value(realtime.get("volume")),
        "news_sentiment": news_sentiment,
        "confidence": round(confidence, 4),
        "predicted_price": round(float(predictions[-1]), 2),
        "pred_change_pct": round(float(pred_change_pct), 2),
        "risk": serialize_value(risk),
        "indicators": serialize_value(indicator_snapshot),
        "backtests": backtests,
        "model_info": serialize_value(model_info),
        "fetched_at": fetched_at,
        "analysis_summary": {
            "forecast_quality": model_quality,
            "forecast_method": model_info.get("label", "Unknown"),
            "forecast_basis": model_info.get("forecast_basis", "unknown"),
            "direction": "bullish" if pred_change_pct >= 0 else "bearish",
            "risk_summary": f"Stop-loss basis: {risk.get('stop_loss_basis', 'n/a')}",
            "notes": model_info.get("notes", []),
            "validation": model_info.get("validation", {}),
        },
        "degradation": {
            "is_partial": bool(model_info.get("is_fallback")) or realtime.get("last_price") == "N/A",
            "reasons": [
                reason for reason in [
                    "baseline_forecast" if model_info.get("is_fallback") else None,
                    "realtime_data_unavailable" if realtime.get("last_price") == "N/A" else None,
                ] if reason
            ],
        },
        "chart_series": {
            "historical": {
                "dates": [ts.isoformat() for ts in historical.index.to_pydatetime()],
                "close": serialize_value(historical["Close"].values),
                "volume": serialize_value(historical["Volume"].values),
            },
            "forecast": {
                "dates": term_dates(historical.index[-1], forecast_days),
                "predictions": serialize_value(predictions),
            },
        },
    }
    try:
        payload["benchmark_summary"] = benchmark_forecasts(historical["Close"].values, forecast_days=forecast_days)
    except ValueError:
        payload["benchmark_summary"] = {
            "windows_evaluated": 0,
            "better_model": "insufficient_data",
            "notes": ["Not enough historical data to run calibration windows."],
        }
    return analysis_cache.set(cache_key, payload)


def build_benchmark_payload(raw_ticker: object, raw_forecast_days: object) -> dict[str, Any]:
    ticker = validate_ticker(raw_ticker)
    forecast_days = validate_forecast_days(raw_forecast_days)
    cache_key = f"{ticker}:{forecast_days}"
    cached = benchmark_cache.get(cache_key)
    if cached is not None:
        return cached

    historical = fetch_stock_data(ticker, period="2y")
    if historical.empty:
        raise ValueError(f"No historical data found for {ticker}")

    summary = benchmark_forecasts(historical["Close"].values, forecast_days=forecast_days)
    payload = {
        "ticker": ticker,
        "forecast_days": forecast_days,
        "benchmark_summary": serialize_value(summary),
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
    }
    return benchmark_cache.set(cache_key, payload)


def build_news_payload() -> dict[str, Any]:
    cached = news_cache.get("news_payload")
    if cached is not None:
        return cached

    results = get_recommendations()
    rec_df = results.get("recommendations", pd.DataFrame())
    articles_df = results.get("all_articles", pd.DataFrame())
    stocks_df = results.get("stocks_found", pd.DataFrame())
    source_status = getattr(articles_df, "attrs", {}).get("source_status", {})

    if not rec_df.empty:
        rec_df = rec_df.copy()
        rec_df["explainability"] = rec_df.apply(
            lambda row: "full_technical" if pd.notna(row.get("predicted_price")) else "impact_only",
            axis=1,
        )
        rec_df["source_count"] = rec_df["article_count"]
        rec_df["reason_summary"] = rec_df.apply(news_reason_summary, axis=1)
        rec_df["confidence_band"] = rec_df["confidence"].apply(
            lambda value: confidence_band(float(value)) if pd.notna(value) else "low"
        )

    summary_counts = {
        "recommendations": int(len(rec_df)),
        "articles": int(len(articles_df)),
        "stocks_found": int(len(stocks_df)),
        "buy": int((rec_df["recommendation"] == "BUY").sum()) if not rec_df.empty else 0,
        "hold": int((rec_df["recommendation"] == "HOLD").sum()) if not rec_df.empty else 0,
        "avoid": int((rec_df["recommendation"] == "AVOID").sum()) if not rec_df.empty else 0,
    }

    payload = {
        "recommendations": frame_records(rec_df),
        "stocks_found": frame_records(stocks_df),
        "articles": frame_records(articles_df.head(20)),
        "summary_counts": summary_counts,
        "source_status": source_status,
        "performance_summary": build_news_performance_summary(rec_df),
        "explanation_summary": {
            "coverage": "technical_plus_news" if not rec_df.empty and rec_df["explainability"].eq("full_technical").any() else "impact_only",
            "top_reason": None if rec_df.empty else rec_df.iloc[0].get("reason_summary"),
            "top_composite_score": None if rec_df.empty else rec_df.iloc[0].get("composite_score"),
        },
        "degradation": {
            "is_partial": any(status != "ok" for status in source_status.values()) if source_status else False,
            "reasons": [f"{source}:{status}" for source, status in source_status.items() if status != "ok"],
        },
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
    }
    return news_cache.set("news_payload", payload)


def build_portfolio_payload(raw_holdings: object) -> dict[str, Any]:
    holdings = validate_holdings(raw_holdings)
    results = run_portfolio_optimization(holdings)
    payload = serialize_value(results)
    payload["assumptions"] = {
        "ranking_method": "heuristic_momentum",
        "risk_free_rate_source": "fixed_approximation",
        "weight_bounds": "minimum 1 percent allocation per asset",
    }
    payload["portfolio_summary"] = {
        "ranking_label": "Momentum Heuristic",
        "optimizer_label": "Sharpe-maximizing Markowitz allocation",
        "assumption_notes": [
            "Ranking uses recent price momentum and volatility features.",
            "Optimizer uses a fixed risk-free approximation and long-only weights.",
        ],
        "top_ranked_asset": None if not payload.get("ranknet_scores") else max(payload["ranknet_scores"], key=payload["ranknet_scores"].get),
        "turnover_ratio": payload.get("metrics", {}).get("turnover_ratio"),
        "ranking_benchmark_summary": payload.get("ranking_benchmark", {}).get("summary"),
        "most_improved_asset": payload.get("ranking_benchmark", {}).get("most_improved_asset"),
        "most_penalized_asset": payload.get("ranking_benchmark", {}).get("most_penalized_asset"),
        "ranking_validation_summary": payload.get("ranking_validation", {}).get("summary"),
        "ranking_validation_win_rate": payload.get("ranking_validation", {}).get("win_rate"),
        "ranking_validation_better_method": payload.get("ranking_validation", {}).get("better_method"),
        "ranking_validation_outperformance_spread": payload.get("ranking_validation", {}).get("outperformance_spread"),
        "ranking_validation_current_sharpe": payload.get("ranking_validation", {}).get("current_forward_sharpe"),
        "ranking_validation_legacy_sharpe": payload.get("ranking_validation", {}).get("legacy_forward_sharpe"),
    }
    return payload
