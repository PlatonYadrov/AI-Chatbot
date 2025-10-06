"""
Scheduler: оркестрация CRON/CDC/ручной старт (Airflow/Prefect). REST API заглушки.
API:
  POST /scheduler/run {source, since, full, priority}
  GET  /scheduler/runs?status=
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RunRequest:
    source: str
    since: Optional[str] = None
    full: bool = False
    priority: Optional[int] = None


def start_run(req: RunRequest) -> Dict[str, Any]:
    # Заглушка: вернёт id и принятые параметры
    return {"run_id": f"run-{int(datetime.utcnow().timestamp())}", "accepted": True, "request": req.__dict__}


def list_runs(status: Optional[str] = None) -> List[Dict[str, Any]]:
    # Заглушка
    return [{"run_id": "run-1", "status": status or "pending"}]


__all__ = ["RunRequest", "start_run", "list_runs"]

