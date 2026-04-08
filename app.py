from __future__ import annotations

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for headless environments

import logging
import os
import time
import uuid
import webbrowser
from collections import deque
from threading import Timer
from typing import Any

from flask import Flask, g, jsonify, render_template, request

from terminal_app.settings import settings
from terminal_app.services import (
    get_frontend_config,
    get_market_status,
    build_analysis_payload,
    build_benchmark_payload,
    build_news_payload,
    build_portfolio_payload,
)
from terminal_app.report_store import (
    list_reports,
    get_report,
    save_report,
    delete_report,
    update_report_metadata,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
diagnostics_buffer = deque(maxlen=settings.diagnostics_window)


def local_browser_url() -> str:
    return f"http://127.0.0.1:{settings.port}"


def should_auto_open_browser() -> bool:
    return settings.open_browser and "PORT" not in os.environ


def open_browser_later(delay_seconds: float = 1.25) -> None:
    url = local_browser_url()
    def _open():
        try:
            webbrowser.open(url)
            logger.info("Opened browser at %s", url)
        except Exception as exc:
            logger.warning("Could not auto-open browser at %s: %s", url, exc)
    Timer(delay_seconds, _open).start()


def api_response(ok: bool, message: str, data: Any = None, error: str | None = None, status: int = 200):
    payload = {"ok": ok, "message": message, "data": data, "error": error}
    return jsonify(payload), status


def handle_api_error(exc: Exception, default_message: str, status: int = 500):
    logger.exception("%s: %s", default_message, exc)
    diagnostics_buffer.appendleft({
        "level": "error",
        "message": default_message,
        "detail": str(exc),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return api_response(False, default_message, None, str(exc), status=status)


@app.before_request
def before_request():
    g.request_started_at = time.perf_counter()
    g.request_id = uuid.uuid4().hex[:8]


@app.after_request
def after_request(response):
    started = getattr(g, "request_started_at", None)
    duration_ms = round((time.perf_counter() - started) * 1000, 2) if started else None
    logger.info("request_complete method=%s path=%s status=%s duration_ms=%s", 
                request.method, request.path, response.status_code, duration_ms)
    diagnostics_buffer.appendleft({
        "level": "info" if response.status_code < 400 else "error",
        "message": f"{request.method} {request.path}",
        "detail": f"status={response.status_code}, duration_ms={duration_ms}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    response.headers["X-TradeWise-Request-Id"] = getattr(g, "request_id", "unknown")
    if duration_ms is not None:
        response.headers["X-TradeWise-Duration-Ms"] = str(duration_ms)
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    # Keep this completely pure — no heavy imports.
    return api_response(True, "Server is healthy", {"status": "ok", "timestamp": time.time()})


@app.get("/ready")
def ready():
    try:
        get_frontend_config()
        return api_response(True, "Server is ready", {"status": "ready", "debug": settings.debug})
    except Exception as exc:
        return handle_api_error(exc, "Server is not ready", status=500)


@app.get("/api/diagnostics")
def diagnostics_route():
    return api_response(True, "Loaded diagnostics", {
        "debug": settings.debug,
        "host": settings.host,
        "port": settings.port,
        "cache_ttls": {
            "market_status": settings.market_status_ttl,
            "analysis": settings.analysis_ttl,
            "news": settings.news_ttl,
        },
        "events": list(diagnostics_buffer),
    })


@app.get("/api/config")
def config_route():
    try:
        return api_response(True, "Loaded configuration", get_frontend_config())
    except Exception as exc:
        return handle_api_error(exc, "Failed to load configuration", status=500)


@app.get("/api/market-status")
def market_status_route():
    try:
        return api_response(True, "Loaded market status", get_market_status())
    except Exception as exc:
        return handle_api_error(exc, "Failed to load market status", status=500)


@app.post("/api/analyze-stock")
def analyze_stock_route():
    payload = request.get_json(silent=True) or {}
    try:
        data = build_analysis_payload(payload.get("ticker", ""), payload.get("forecast_days", 30))
        return api_response(True, f"Analysis completed for {data['ticker']}", data)
    except ValueError as exc:
        return api_response(False, str(exc), None, str(exc), status=400)
    except Exception as exc:
        return handle_api_error(exc, "Analysis failed", status=500)


@app.post("/api/benchmarks")
def benchmark_route():
    payload = request.get_json(silent=True) or {}
    try:
        data = build_benchmark_payload(payload.get("ticker", ""), payload.get("forecast_days", 30))
        return api_response(True, f"Benchmark completed for {data['ticker']}", data)
    except ValueError as exc:
        return api_response(False, str(exc), None, str(exc), status=400)
    except Exception as exc:
        return handle_api_error(exc, "Benchmark failed", status=500)


@app.post("/api/news-recommendations")
def news_recommendations_route():
    try:
        return api_response(True, "News recommendation scan completed", build_news_payload())
    except Exception as exc:
        return handle_api_error(exc, "News recommendation scan failed", status=500)


@app.post("/api/portfolio/optimize")
def portfolio_optimize_route():
    payload = request.get_json(silent=True) or {}
    try:
        data = build_portfolio_payload(payload.get("holdings", []))
        return api_response(True, "Portfolio optimization completed", data)
    except ValueError as exc:
        return api_response(False, str(exc), None, str(exc), status=400)
    except Exception as exc:
        return handle_api_error(exc, "Portfolio optimization failed", status=500)


@app.get("/api/reports")
def reports_route():
    try:
        return api_response(True, "Loaded report history", {"reports": list_reports()})
    except Exception as exc:
        return handle_api_error(exc, "Failed to load report history", status=500)


@app.get("/api/reports/<report_id>")
def report_detail_route(report_id: str):
    try:
        return api_response(True, "Loaded report snapshot", get_report(report_id))
    except FileNotFoundError as exc:
        return api_response(False, str(exc), None, str(exc), status=404)
    except Exception as exc:
        return handle_api_error(exc, "Failed to load report snapshot", status=500)


@app.post("/api/reports")
def save_report_route():
    payload = request.get_json(silent=True) or {}
    report_type = str(payload.get("type", "")).strip().lower()
    title = str(payload.get("title", "")).strip()
    report_payload = payload.get("payload")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    if not report_type or report_payload is None:
        return api_response(False, "Report type and payload are required.", None, "Invalid report payload.", status=400)

    try:
        saved = save_report(report_type, title or f"{report_type.title()} Snapshot", report_payload, metadata=metadata)
        return api_response(True, "Report snapshot saved", saved, status=201)
    except Exception as exc:
        return handle_api_error(exc, "Failed to save report snapshot", status=500)


@app.delete("/api/reports/<report_id>")
def delete_report_route(report_id: str):
    try:
        delete_report(report_id)
        return api_response(True, "Report deleted", {"id": report_id})
    except FileNotFoundError as exc:
        return api_response(False, str(exc), None, str(exc), status=404)
    except Exception as exc:
        return handle_api_error(exc, "Failed to delete report", status=500)


@app.post("/api/reports/<report_id>/pin")
def pin_report_route(report_id: str):
    payload = request.get_json(silent=True) or {}
    pinned = bool(payload.get("pinned", True))
    try:
        updated = update_report_metadata(report_id, {"pinned": pinned})
        return api_response(True, "Report pin state updated", updated)
    except FileNotFoundError as exc:
        return api_response(False, str(exc), None, str(exc), status=404)
    except Exception as exc:
        return handle_api_error(exc, "Failed to update report pin state", status=500)


if __name__ == "__main__":
    if should_auto_open_browser():
        logger.info("Launching local browser at %s", local_browser_url())
        open_browser_later()

    if settings.debug:
        app.run(debug=True, host=settings.host, port=settings.port)
    else:
        try:
            from waitress import serve
            logger.info("Starting Waitress production server on %s:%s", settings.host, settings.port)
            serve(app, host=settings.host, port=settings.port)
        except ImportError:
            logger.warning("waitress is not installed; falling back to Flask development server.")
            app.run(debug=False, host=settings.host, port=settings.port)
