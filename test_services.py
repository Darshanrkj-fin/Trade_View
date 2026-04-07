import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portfolio_optimizer import build_ranking_validation
from terminal_app.serializers import frame_records, serialize_value
from terminal_app.services import (
    analysis_cache,
    build_analysis_payload,
    build_benchmark_payload,
    build_news_payload,
    build_portfolio_payload,
    benchmark_cache,
    market_status_cache,
    news_cache,
)


class SerializerTests(unittest.TestCase):
    def test_serialize_dataframe_with_datetime_index(self):
        frame = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.date_range("2026-01-01", periods=2))
        records = frame_records(frame)
        self.assertEqual(len(records), 2)
        self.assertIn("index", records[0])

    def test_serialize_series_and_numpy(self):
        self.assertEqual(serialize_value(pd.Series([1, 2, 3])), [1, 2, 3])
        self.assertEqual(serialize_value(np.array([1.5, 2.5])), [1.5, 2.5])

    def test_serialize_nan_and_inf(self):
        values = serialize_value(np.array([1.0, np.nan, np.inf]))
        self.assertEqual(values, [1.0, None, None])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        analysis_cache.clear()
        benchmark_cache.clear()
        news_cache.clear()
        market_status_cache.clear()

    @patch("terminal_app.services.fetch_news_sentiment", return_value="Positive")
    @patch("terminal_app.services.fetch_realtime_data", return_value={"last_price": 101.5, "volume": 2000})
    @patch("terminal_app.services.calculate_indicators")
    @patch("terminal_app.services.predict_lstm")
    @patch("terminal_app.services.benchmark_forecasts")
    @patch("terminal_app.services.fetch_stock_data")
    def test_build_analysis_payload(self, mock_history, mock_benchmark, mock_predict, mock_indicators, _mock_realtime, _mock_sentiment):
        index = pd.date_range("2025-01-01", periods=240)
        history = pd.DataFrame(
            {
                "Open": np.linspace(90, 100, 240),
                "High": np.linspace(91, 101, 240),
                "Low": np.linspace(89, 99, 240),
                "Close": np.linspace(90, 110, 240),
                "Volume": np.linspace(1000, 2000, 240),
            },
            index=index,
        )
        mock_history.return_value = history
        mock_benchmark.return_value = {"windows_evaluated": 3, "better_model": "current_model"}
        mock_indicators.return_value = {"RSI": 55, "MA50": 102, "MA100": 99, "MA200": 95, "MACD": 1.2, "Bollinger": 101}
        mock_predict.return_value = {
            "predictions": np.array([111.0, 112.0, 113.0]),
            "confidence": 0.67,
            "model_info": {
                "engine": "sklearn_linear_regression",
                "label": "Baseline Forecast",
                "is_fallback": True,
                "confidence_band": "medium",
                "forecast_basis": "trend_extrapolation",
                "notes": ["Baseline model note"],
                "validation": {"r2": 0.41, "directional_accuracy": 0.62},
            },
        }

        payload = build_analysis_payload("RELIANCE.NS", 7)
        self.assertEqual(payload["ticker"], "RELIANCE.NS")
        self.assertIn("model_info", payload)
        self.assertIn("fetched_at", payload)
        self.assertIn("chart_series", payload)
        self.assertEqual(payload["benchmark_summary"]["better_model"], "current_model")
        self.assertEqual(payload["analysis_summary"]["forecast_quality"], "medium")
        self.assertEqual(payload["analysis_summary"]["forecast_basis"], "trend_extrapolation")
        self.assertEqual(payload["analysis_summary"]["validation"]["directional_accuracy"], 0.62)

    @patch("terminal_app.services.get_recommendations")
    def test_build_news_payload_empty(self, mock_recommendations):
        mock_recommendations.return_value = {
            "recommendations": pd.DataFrame(),
            "all_articles": pd.DataFrame(),
            "stocks_found": pd.DataFrame(),
        }
        payload = build_news_payload()
        self.assertEqual(payload["summary_counts"]["recommendations"], 0)

    @patch("terminal_app.services.get_recommendations")
    def test_build_news_payload_populated(self, mock_recommendations):
        articles = pd.DataFrame([{"headline": "Sample", "source": "LiveMint"}])
        articles.attrs["source_status"] = {"livemint": "ok", "google_news": "failed"}
        mock_recommendations.return_value = {
            "recommendations": pd.DataFrame([{
                "ticker": "RELIANCE.NS",
                "recommendation": "BUY",
                "article_count": 3,
                "predicted_price": 123,
                "impact_score": 77,
                "pred_change_pct": 6.4,
                "confidence": 0.71,
                "composite_score": 38.2,
                "backtest_30_profit": 12.5,
                "backtest_30_accuracy": 61.4,
                "backtest_30_buy_hold": 8.3,
            }]),
            "all_articles": articles,
            "stocks_found": pd.DataFrame([{"ticker": "RELIANCE.NS", "impact_score": 77}]),
        }
        payload = build_news_payload()
        self.assertEqual(payload["recommendations"][0]["explainability"], "full_technical")
        self.assertEqual(payload["recommendations"][0]["source_count"], 3)
        self.assertIn("reason_summary", payload["recommendations"][0])
        self.assertEqual(payload["recommendations"][0]["confidence_band"], "medium")
        self.assertEqual(payload["explanation_summary"]["top_composite_score"], 38.2)
        self.assertEqual(payload["performance_summary"]["buy_avg_backtest_profit"], 12.5)
        self.assertEqual(payload["performance_summary"]["coverage"], 1)
        self.assertTrue(payload["degradation"]["is_partial"])

    @patch("terminal_app.services.fetch_news_sentiment", return_value="Positive")
    @patch("terminal_app.services.fetch_realtime_data", return_value={"last_price": "N/A", "volume": "N/A"})
    @patch("terminal_app.services.calculate_indicators")
    @patch("terminal_app.services.predict_lstm")
    @patch("terminal_app.services.benchmark_forecasts")
    @patch("terminal_app.services.fetch_stock_data")
    def test_build_analysis_payload_marks_degraded_response(self, mock_history, mock_benchmark, mock_predict, mock_indicators, _mock_realtime, _mock_sentiment):
        index = pd.date_range("2025-01-01", periods=240)
        history = pd.DataFrame(
            {
                "Open": np.linspace(90, 100, 240),
                "High": np.linspace(91, 101, 240),
                "Low": np.linspace(89, 99, 240),
                "Close": np.linspace(90, 110, 240),
                "Volume": np.linspace(1000, 2000, 240),
            },
            index=index,
        )
        mock_history.return_value = history
        mock_benchmark.return_value = {"windows_evaluated": 3, "better_model": "baseline_model"}
        mock_indicators.return_value = {"RSI": 55, "MA50": 102, "MA100": 99, "MA200": 95, "MACD": 1.2, "Bollinger": 101}
        mock_predict.return_value = {
            "predictions": np.array([111.0, 112.0, 113.0]),
            "confidence": 0.67,
            "model_info": {
                "engine": "sklearn_linear_regression",
                "label": "Baseline Forecast",
                "is_fallback": True,
                "confidence_band": "medium",
                "forecast_basis": "trend_extrapolation",
                "notes": ["Baseline model note"],
                "validation": {"r2": 0.41, "directional_accuracy": 0.62},
            },
        }

        payload = build_analysis_payload("INFY.NS", 7)
        self.assertTrue(payload["degradation"]["is_partial"])

    @patch("terminal_app.services.benchmark_forecasts")
    @patch("terminal_app.services.fetch_stock_data")
    def test_build_benchmark_payload(self, mock_history, mock_benchmark):
        index = pd.date_range("2025-01-01", periods=240)
        history = pd.DataFrame({"Close": np.linspace(90, 110, 240)}, index=index)
        mock_history.return_value = history
        mock_benchmark.return_value = {"windows_evaluated": 3, "better_model": "current_model"}

        payload = build_benchmark_payload("RELIANCE.NS", 30)
        self.assertEqual(payload["ticker"], "RELIANCE.NS")
        self.assertEqual(payload["benchmark_summary"]["windows_evaluated"], 3)

    @patch("terminal_app.services.run_portfolio_optimization")
    def test_build_portfolio_payload_valid(self, mock_optimize):
        mock_optimize.return_value = {
            "optimized_weights": {"RELIANCE.NS": 0.6, "INFY.NS": 0.4},
            "current_weights": {"RELIANCE.NS": 0.5, "INFY.NS": 0.5},
            "trade_suggestions": {},
            "ranknet_scores": {"RELIANCE.NS": 0.8, "INFY.NS": 0.4},
            "ranking_context": {"RELIANCE.NS": {"summary": "positive 1M momentum"}},
            "ranking_benchmark": {
                "summary": "Current ranking adds stronger penalties for volatility and drawdown.",
                "most_improved_asset": "RELIANCE.NS",
                "most_penalized_asset": "INFY.NS",
            },
            "ranking_validation": {
                "summary": "Current heuristic delivered stronger forward picks than the legacy ranking.",
                "better_method": "current",
                "win_rate": 66.67,
                "outperformance_spread": 2.45,
                "current_forward_sharpe": 1.18,
                "legacy_forward_sharpe": 0.64,
            },
            "capm_metrics": {},
            "efficient_frontier": {
                "vols": [0.1],
                "rets": [0.2],
                "current_point": {"volatility": 0.12, "return": 0.16},
                "optimal_point": {"volatility": 0.1, "return": 0.2},
            },
            "metrics": {"optimal": {"return": 0.2, "volatility": 0.1, "sharpe": 1.5}, "turnover_ratio": 0.18},
        }
        payload = build_portfolio_payload(
            [{"ticker": "RELIANCE.NS", "qty": 10, "buy_price": 2500}, {"ticker": "INFY.NS", "qty": 5, "buy_price": 1500}]
        )
        self.assertIn("assumptions", payload)
        self.assertEqual(payload["assumptions"]["ranking_method"], "heuristic_momentum")
        self.assertEqual(payload["portfolio_summary"]["top_ranked_asset"], "RELIANCE.NS")
        self.assertEqual(payload["portfolio_summary"]["turnover_ratio"], 0.18)
        self.assertEqual(payload["portfolio_summary"]["most_improved_asset"], "RELIANCE.NS")
        self.assertEqual(payload["portfolio_summary"]["ranking_validation_win_rate"], 66.67)
        self.assertEqual(payload["portfolio_summary"]["ranking_validation_better_method"], "current")
        self.assertEqual(payload["portfolio_summary"]["ranking_validation_current_sharpe"], 1.18)
        self.assertEqual(payload["efficient_frontier"]["optimal_point"]["return"], 0.2)

    def test_build_portfolio_payload_invalid(self):
        with self.assertRaises(ValueError):
            build_portfolio_payload([{"ticker": "RELIANCE.NS", "qty": 10, "buy_price": 2500}])

    def test_build_ranking_validation_returns_rich_summary(self):
        close_a = np.concatenate([
            np.linspace(100, 130, 126),
            np.linspace(130, 146, 21),
            np.linspace(146, 166, 21),
        ])
        close_b = np.concatenate([
            np.linspace(100, 145, 126),
            np.linspace(145, 132, 21),
            np.linspace(132, 118, 21),
        ])
        historical_data = {
            "A.NS": pd.DataFrame({"Close": close_a}),
            "B.NS": pd.DataFrame({"Close": close_b}),
        }

        validation = build_ranking_validation(["A.NS", "B.NS"], historical_data, lookback_days=126, forward_days=21, max_windows=2)
        self.assertEqual(validation["windows_evaluated"], 2)
        self.assertIn(validation["better_method"], {"current", "legacy"})
        self.assertIsNotNone(validation["current_forward_sharpe"])
        self.assertIsNotNone(validation["legacy_forward_sharpe"])
        self.assertIsNotNone(validation["outperformance_spread"])


if __name__ == "__main__":
    unittest.main()
