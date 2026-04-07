from __future__ import annotations

import numpy as np

from model import predict_lstm, predict_sklearn_fallback


def _directional_accuracy(predictions, actual):
    predictions = np.asarray(predictions, dtype=float).flatten()
    actual = np.asarray(actual, dtype=float).flatten()
    if len(predictions) < 2 or len(actual) < 2:
        return 0.5

    pred_dir = np.sign(np.diff(predictions))
    actual_dir = np.sign(np.diff(actual))
    return float(np.mean(pred_dir == actual_dir)) if len(pred_dir) == len(actual_dir) else 0.5


def _forecast_metrics(predictions, actual):
    predictions = np.asarray(predictions, dtype=float).flatten()
    actual = np.asarray(actual, dtype=float).flatten()
    if len(predictions) == 0 or len(actual) == 0 or len(predictions) != len(actual):
        return {
            "mape": None,
            "rmse": None,
            "directional_accuracy": None,
            "terminal_error_pct": None,
        }

    safe_actual = np.where(actual == 0, 1e-8, actual)
    mape = float(np.mean(np.abs((actual - predictions) / safe_actual)) * 100)
    rmse = float(np.sqrt(np.mean((actual - predictions) ** 2)))
    terminal_error_pct = float(((predictions[-1] - actual[-1]) / safe_actual[-1]) * 100)

    return {
        "mape": mape,
        "rmse": rmse,
        "directional_accuracy": _directional_accuracy(predictions, actual) * 100,
        "terminal_error_pct": terminal_error_pct,
    }


def _aggregate_metrics(windows):
    if not windows:
        return {
            "windows": 0,
            "avg_mape": None,
            "avg_rmse": None,
            "avg_directional_accuracy": None,
            "avg_terminal_error_pct": None,
        }

    return {
        "windows": len(windows),
        "avg_mape": round(float(np.mean([item["mape"] for item in windows if item["mape"] is not None])), 4),
        "avg_rmse": round(float(np.mean([item["rmse"] for item in windows if item["rmse"] is not None])), 4),
        "avg_directional_accuracy": round(
            float(np.mean([item["directional_accuracy"] for item in windows if item["directional_accuracy"] is not None])),
            2,
        ),
        "avg_terminal_error_pct": round(
            float(np.mean([item["terminal_error_pct"] for item in windows if item["terminal_error_pct"] is not None])),
            4,
        ),
    }


def benchmark_forecasts(prices, forecast_days=30, evaluation_windows=3, min_train_size=180):
    prices = np.asarray(prices, dtype=float).flatten()
    if len(prices) < min_train_size + forecast_days:
        raise ValueError(f"Not enough data to benchmark forecasts. Need at least {min_train_size + forecast_days} points.")

    max_windows = min(
        evaluation_windows,
        max(1, (len(prices) - min_train_size) // forecast_days),
    )
    current_windows = []
    baseline_windows = []

    for offset in range(max_windows, 0, -1):
        split_idx = len(prices) - (offset * forecast_days)
        train = prices[:split_idx]
        actual = prices[split_idx:split_idx + forecast_days]
        if len(train) < min_train_size or len(actual) != forecast_days:
            continue

        current_result = predict_lstm(train, forecast_days=forecast_days)
        current_predictions = (
            np.asarray(current_result["predictions"], dtype=float)
            if isinstance(current_result, dict)
            else np.asarray(current_result, dtype=float)
        )
        baseline_result = predict_sklearn_fallback(train, forecast_days=forecast_days)
        baseline_predictions = np.asarray(baseline_result["predictions"], dtype=float)

        current_metrics = _forecast_metrics(current_predictions, actual)
        baseline_metrics = _forecast_metrics(baseline_predictions, actual)

        current_windows.append({"split_index": split_idx, **current_metrics})
        baseline_windows.append({"split_index": split_idx, **baseline_metrics})

    current_summary = _aggregate_metrics(current_windows)
    baseline_summary = _aggregate_metrics(baseline_windows)
    better_model = "current_model"
    if (
        current_summary["avg_mape"] is not None
        and baseline_summary["avg_mape"] is not None
        and baseline_summary["avg_mape"] < current_summary["avg_mape"]
    ):
        better_model = "baseline_model"

    return {
        "current_model": current_summary,
        "baseline_model": baseline_summary,
        "windows_evaluated": current_summary["windows"],
        "better_model": better_model,
        "notes": [
            "Benchmark uses rolling holdout windows from the recent historical series.",
            "Lower MAPE and higher directional accuracy indicate better calibration.",
        ],
    }
