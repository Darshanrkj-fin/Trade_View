import unittest
from unittest.mock import patch


class FlaskRouteTests(unittest.TestCase):
    def setUp(self):
        from app import app

        self.client = app.test_client()

    def test_home_route_loads_terminal_shell(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("TradeWise AI Terminal", html)
        self.assertIn("static/js/app.js", html)
        self.assertIn('data-view="analysis"', html)

    def test_health_route(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])

    def test_ready_route(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertIn("debug", payload["data"])

    def test_diagnostics_route(self):
        response = self.client.get("/api/diagnostics")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertIn("events", payload["data"])

    @patch("app.list_reports")
    def test_reports_route(self, mock_list_reports):
        mock_list_reports.return_value = [{"id": "abc123", "type": "analysis", "title": "Sample", "created_at": "2026-04-06T20:00:00", "metadata": {"ticker": "RELIANCE.NS"}}]
        response = self.client.get("/api/reports")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["reports"][0]["id"], "abc123")
        self.assertEqual(payload["data"]["reports"][0]["metadata"]["ticker"], "RELIANCE.NS")

    @patch("app.get_report")
    def test_report_detail_route(self, mock_get_report):
        mock_get_report.return_value = {"id": "abc123", "type": "analysis", "title": "Sample", "payload": {"ticker": "RELIANCE.NS"}}
        response = self.client.get("/api/reports/abc123")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["payload"]["ticker"], "RELIANCE.NS")

    def test_save_report_invalid_payload(self):
        response = self.client.post("/api/reports", json={"type": "analysis"})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["ok"])

    @patch("app.save_report")
    def test_save_report_valid_payload(self, mock_save_report):
        mock_save_report.return_value = {"id": "abc123", "type": "analysis", "title": "Saved report", "created_at": "2026-04-06T20:00:00", "metadata": {"ticker": "RELIANCE.NS"}}
        response = self.client.post("/api/reports", json={"type": "analysis", "title": "Saved report", "payload": {"ticker": "RELIANCE.NS"}, "metadata": {"ticker": "RELIANCE.NS"}})
        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["id"], "abc123")
        self.assertEqual(payload["data"]["metadata"]["ticker"], "RELIANCE.NS")

    @patch("app.update_report_metadata")
    def test_pin_report_valid(self, mock_update_report_metadata):
        mock_update_report_metadata.return_value = {"id": "abc123", "type": "analysis", "title": "Saved report", "metadata": {"pinned": True}}
        response = self.client.post("/api/reports/abc123/pin", json={"pinned": True})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["data"]["metadata"]["pinned"])

    @patch("app.delete_report")
    def test_delete_report_valid(self, mock_delete_report):
        response = self.client.delete("/api/reports/abc123")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["id"], "abc123")

    def test_config_route(self):
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertIn("term_options", payload["data"])

    def test_analyze_stock_missing_ticker(self):
        response = self.client.post("/api/analyze-stock", json={})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["ok"])

    @patch("app.build_analysis_payload")
    def test_analyze_stock_valid_input(self, mock_payload):
        mock_payload.return_value = {"ticker": "RELIANCE.NS", "chart_series": {}, "model_info": {}}
        response = self.client.post("/api/analyze-stock", json={"ticker": "RELIANCE.NS", "forecast_days": 30})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["ticker"], "RELIANCE.NS")

    def test_benchmark_missing_ticker(self):
        response = self.client.post("/api/benchmarks", json={})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["ok"])

    @patch("app.build_benchmark_payload")
    def test_benchmark_valid_input(self, mock_payload):
        mock_payload.return_value = {
            "ticker": "RELIANCE.NS",
            "forecast_days": 30,
            "benchmark_summary": {"windows_evaluated": 3, "better_model": "current_model"},
        }
        response = self.client.post("/api/benchmarks", json={"ticker": "RELIANCE.NS", "forecast_days": 30})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["benchmark_summary"]["windows_evaluated"], 3)

    @patch("app.build_news_payload")
    def test_news_route(self, mock_payload):
        mock_payload.return_value = {
            "recommendations": [],
            "stocks_found": [],
            "articles": [],
            "summary_counts": {"recommendations": 0, "articles": 0, "stocks_found": 0, "buy": 0, "hold": 0, "avoid": 0},
        }
        response = self.client.post("/api/news-recommendations", json={})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])

    @patch("app.build_news_payload", side_effect=RuntimeError("scraper unavailable"))
    def test_news_route_failure(self, _mock_payload):
        response = self.client.post("/api/news-recommendations", json={})
        self.assertEqual(response.status_code, 500)
        payload = response.get_json()
        self.assertFalse(payload["ok"])

    def test_portfolio_optimize_invalid_input(self):
        response = self.client.post("/api/portfolio/optimize", json={"holdings": []})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["ok"])

    @patch("app.build_portfolio_payload")
    def test_portfolio_optimize_valid_input(self, mock_payload):
        mock_payload.return_value = {"optimized_weights": {"RELIANCE.NS": 0.5, "INFY.NS": 0.5}}
        response = self.client.post(
            "/api/portfolio/optimize",
            json={"holdings": [{"ticker": "RELIANCE.NS", "qty": 10, "buy_price": 2500}, {"ticker": "INFY.NS", "qty": 5, "buy_price": 1500}]},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])


if __name__ == "__main__":
    unittest.main()
