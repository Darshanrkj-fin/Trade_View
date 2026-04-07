import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TensorFlow is loaded lazily to speed up app startup
_TF_LOADED = False
_TF_AVAILABLE = False


def check_tensorflow():
    """Lazily check if TensorFlow is available and import it."""
    global _TF_LOADED, _TF_AVAILABLE
    if _TF_LOADED:
        return _TF_AVAILABLE

    try:
        import tensorflow as tf
        _TF_AVAILABLE = True
        logger.info("TensorFlow loaded successfully — using LSTM models")
    except ImportError:
        _TF_AVAILABLE = False
        logger.info("TensorFlow not available — using sklearn fallback models")
    
    _TF_LOADED = True
    return _TF_AVAILABLE


def confidence_band(confidence):
    """Map a numeric confidence score to a simple user-facing quality band."""
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def build_model_info(engine, label, confidence, is_fallback, fit_score=None, notes=None):
    return {
        "engine": engine,
        "label": label,
        "is_fallback": is_fallback,
        "confidence_band": confidence_band(confidence),
        "fit_score": None if fit_score is None else round(float(fit_score), 4),
        "forecast_basis": "trend_extrapolation" if is_fallback else "sequence_learning",
        "notes": notes or [],
    }


def create_model(sequence_length, forecast_days=30):
    """Create an LSTM model for time series prediction (requires TensorFlow)"""
    if not check_tensorflow():
        raise ImportError("TensorFlow is required for LSTM model")

    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(sequence_length, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def prepare_sequences(data, sequence_length, forecast_days):
    if len(data) < sequence_length + forecast_days:
        raise ValueError(f"Not enough data points. Need at least {sequence_length + forecast_days}, got {len(data)}")

    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_days + 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Failed to create sequences")

    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], 1))
    return X, y


def calculate_confidence(predictions, actual, scale_min, scale_max):
    if len(predictions) != len(actual):
        raise ValueError("Predictions and actual values must have same length")

    mse = np.mean((predictions - actual) ** 2)
    scale_range = max(1e-8, scale_max - scale_min)
    normalized_mse = mse / (scale_range ** 2)
    confidence = max(0.0, min(1.0, 1.0 - (normalized_mse * 10)))
    return confidence


def calculate_forecast_metrics(predictions, actual, scale_min, scale_max):
    """Derive validation metrics for both baseline and LSTM paths."""
    predictions = np.asarray(predictions, dtype=float).flatten()
    actual = np.asarray(actual, dtype=float).flatten()

    if len(predictions) == 0 or len(actual) == 0 or len(predictions) != len(actual):
        return {"mae": None, "rmse": None, "directional_accuracy": None, "confidence": 0.0}

    errors = predictions - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    predicted_direction = np.sign(np.diff(predictions))
    actual_direction = np.sign(np.diff(actual))
    directional_accuracy = (
        float(np.mean(predicted_direction == actual_direction))
        if len(predicted_direction) and len(actual_direction)
        else 0.5
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "confidence": calculate_confidence(predictions, actual, scale_min, scale_max),
    }


def predict_sklearn_fallback(data, forecast_days=30, sequence_length=60):
    """
    Fallback prediction using sklearn LinearRegression + trend extrapolation.
    Works without TensorFlow.
    """
    data = np.asarray(data, dtype=np.float64)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    # Use recent data for trend estimation
    lookback = min(sequence_length, len(scaled) - 1)
    X_train = np.arange(len(scaled) - lookback, len(scaled)).reshape(-1, 1)
    y_train = scaled[-lookback:].flatten()

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future
    X_future = np.arange(len(scaled), len(scaled) + forecast_days).reshape(-1, 1)
    pred_scaled = model.predict(X_future).reshape(-1, 1)

    # Clip to valid range and inverse transform
    pred_scaled = np.clip(pred_scaled, 0, 1)
    predictions = scaler.inverse_transform(pred_scaled).flatten()

    # Confidence based on fit quality plus directional consistency.
    from sklearn.metrics import r2_score
    y_pred_train = model.predict(X_train)
    r2 = max(0, r2_score(y_train, y_pred_train))
    metrics = calculate_forecast_metrics(
        y_pred_train,
        y_train,
        float(np.min(data)),
        float(np.max(data)),
    )
    confidence = max(0.0, min(1.0, (r2 * 0.45) + (metrics["directional_accuracy"] * 0.25) + (metrics["confidence"] * 0.30)))

    model_info = build_model_info(
        engine="sklearn_linear_regression",
        label="Baseline Forecast",
        confidence=confidence,
        is_fallback=True,
        fit_score=r2,
        notes=[
            "Uses a linear regression baseline over the recent trend window.",
            "Best suited for directional context, not long-horizon precision.",
        ],
    )
    model_info["validation"] = {
        "r2": round(float(r2), 4),
        "mae": None if metrics["mae"] is None else round(float(metrics["mae"]), 6),
        "rmse": None if metrics["rmse"] is None else round(float(metrics["rmse"]), 6),
        "directional_accuracy": None if metrics["directional_accuracy"] is None else round(float(metrics["directional_accuracy"]), 4),
    }

    return {
        'predictions': predictions,
        'confidence': confidence,
        'model': None,
        'scaler': None,
        'model_info': model_info,
    }


def predict_lstm(data, forecast_days=30, sequence_length=60, model=None, scaler=None):
    """
    Predict stock prices using LSTM model (TensorFlow) or fallback to sklearn.
    """
    # If TensorFlow is not available, use fallback
    if not check_tensorflow():
        logger.info("Using sklearn fallback predictor")
        return predict_sklearn_fallback(data, forecast_days, sequence_length)

    import tensorflow as tf
    try:
        data = np.asarray(data, dtype=np.float32)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        if len(data) < sequence_length + forecast_days:
            raise ValueError(f"Not enough data points. Need at least {sequence_length + forecast_days}, got {len(data)}")

        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
        else:
            scaled_data = scaler.transform(data)

        X, y = prepare_sequences(scaled_data, sequence_length, forecast_days)

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        if model is None:
            logger.info("Creating and training temporary LSTM model...")
            model = create_model(sequence_length, forecast_days)

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )

            model.fit(
                X_train, y_train, epochs=100, batch_size=32,
                validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0
            )

        val_predictions = model.predict(X_val, verbose=0)
        metrics = calculate_forecast_metrics(
            val_predictions.flatten(),
            y_val.flatten(),
            float(np.min(data)),
            float(np.max(data)),
        )
        confidence = max(
            0.0,
            min(1.0, (metrics["confidence"] * 0.6) + (metrics["directional_accuracy"] * 0.4)),
        )

        predictions = []
        last_sequence = scaled_data[-sequence_length:].copy()

        for _ in range(forecast_days):
            current_sequence = last_sequence[-sequence_length:].reshape(1, sequence_length, 1)
            next_pred = model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            last_sequence = np.vstack([last_sequence, next_pred])

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)

        logger.info(f"Generated predictions for next {forecast_days} days with confidence: {confidence:.2%}")

        model_info = build_model_info(
            engine="tensorflow_lstm",
            label="LSTM Forecast",
            confidence=confidence,
            is_fallback=False,
            fit_score=confidence,
            notes=[
                "Sequence model trained on the recent price history window.",
                "Confidence is derived from validation error on held-out sequences.",
            ],
        )
        model_info["validation"] = {
            "mae": None if metrics["mae"] is None else round(float(metrics["mae"]), 6),
            "rmse": None if metrics["rmse"] is None else round(float(metrics["rmse"]), 6),
            "directional_accuracy": None if metrics["directional_accuracy"] is None else round(float(metrics["directional_accuracy"]), 4),
        }

        return {
            'predictions': predictions.flatten(),
            'confidence': confidence,
            'model': model,
            'scaler': scaler,
            'model_info': model_info,
        }

    except Exception as e:
        logger.error(f"LSTM prediction failed: {e}, falling back to sklearn")
        return predict_sklearn_fallback(data, forecast_days, sequence_length)


def train_lstm(data, forecast_days=30, sequence_length=60, epochs=100):
    if not check_tensorflow():
        logger.warning("TensorFlow not available, cannot train LSTM")
        return None, None

    import tensorflow as tf
    try:
        data = np.asarray(data, dtype=np.float32)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        if len(data) < sequence_length + forecast_days:
            raise ValueError(f"Not enough data points. Need at least {sequence_length + forecast_days}, got {len(data)}")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X, y = prepare_sequences(scaled_data, sequence_length, forecast_days)

        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        logger.info("Creating and training new LSTM model...")
        model = create_model(sequence_length, forecast_days)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        model.fit(
            X_train, y_train, epochs=epochs, batch_size=32,
            validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1
        )

        logger.info("Model training completed successfully")
        return model, scaler

    except Exception as e:
        logger.error(f"Error in train_lstm: {str(e)}")
        return None, None


def create_sequences(data, seq_length=60):
    try:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    except Exception as e:
        print(f"Error in create_sequences: {e}")
        return np.array([]), np.array([])
