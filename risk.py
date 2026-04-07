import numpy as np

def assess_risk(historical, predictions):
    if historical.empty or 'Close' not in historical:
        return {"volatility": 0, "stop_loss": 0, "confidence": 0, "projected_range": [0, 0], "stop_loss_basis": "insufficient_data"}

    close = historical['Close'].values
    recent_window = close[-30:] if len(close) >= 30 else close
    volatility = np.std(recent_window)
    last_price = close[-1]

    # Scale stop-loss width with observed volatility instead of a fixed placeholder.
    downside_buffer = max(last_price * 0.05, volatility * 1.5)
    stop_loss = max(0, last_price - downside_buffer)
    projected_range = [max(0, last_price - volatility), last_price + volatility]

    predictions = np.asarray(predictions, dtype=float) if len(predictions) else np.array([])
    if predictions.size:
        projected_move = abs(predictions[-1] - last_price)
        confidence = projected_move / (projected_move + volatility + 1e-8)
    else:
        confidence = 0.0

    return {
        "volatility": volatility,
        "stop_loss": stop_loss,
        "confidence": confidence,
        "projected_range": projected_range,
        "stop_loss_basis": "max(5pct_price_buffer, 1.5x_recent_volatility)",
    }
