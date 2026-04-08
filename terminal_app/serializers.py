from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def clean_number(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if type(value).__name__ in ('bool', 'bool_', 'bool8') and hasattr(value, 'item'):
        return bool(value.item())
    elif isinstance(value, np.bool_):
        return bool(value)
    return value


def serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [serialize_value(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DatetimeIndex):
        return [ts.isoformat() for ts in value.to_pydatetime()]
    if isinstance(value, pd.Series):
        return [serialize_value(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        df = value.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
        return [serialize_value(record) for record in df.to_dict(orient="records")]
    return clean_number(value)


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    df = frame.copy()
    if df.index.name or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    return serialize_value(df)
