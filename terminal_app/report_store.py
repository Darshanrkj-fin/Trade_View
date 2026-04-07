from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from terminal_app.serializers import serialize_value


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _report_path(report_id: str) -> Path:
    return REPORTS_DIR / f"{report_id}.json"


def save_report(report_type: str, title: str, payload: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    report_id = uuid.uuid4().hex[:12]
    record = {
        "id": report_id,
        "type": report_type,
        "title": title,
        "created_at": serialize_value(pd.Timestamp.utcnow()),
        "metadata": serialize_value(metadata or {}),
        "payload": serialize_value(payload),
    }
    _report_path(report_id).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {k: v for k, v in record.items() if k != "payload"}


def list_reports(limit: int = 20) -> list[dict[str, Any]]:
    records = []
    for path in sorted(REPORTS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append({
                "id": data.get("id", path.stem),
                "type": data.get("type", "unknown"),
                "title": data.get("title", path.stem),
                "created_at": data.get("created_at"),
                "metadata": data.get("metadata", {}),
            })
        except Exception:
            continue
    return records


def get_report(report_id: str) -> dict[str, Any]:
    path = _report_path(report_id)
    if not path.exists():
        raise FileNotFoundError(f"Report {report_id} not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def delete_report(report_id: str) -> None:
    path = _report_path(report_id)
    if not path.exists():
        raise FileNotFoundError(f"Report {report_id} not found.")
    path.unlink()


def update_report_metadata(report_id: str, metadata_updates: dict[str, Any]) -> dict[str, Any]:
    path = _report_path(report_id)
    if not path.exists():
        raise FileNotFoundError(f"Report {report_id} not found.")

    record = json.loads(path.read_text(encoding="utf-8"))
    record["metadata"] = {**record.get("metadata", {}), **serialize_value(metadata_updates)}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {k: v for k, v in record.items() if k != "payload"}
