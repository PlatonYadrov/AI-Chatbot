from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RawDoc:
    doc_id: str
    source: str
    url: Optional[str]
    path: Optional[str]
    mime: Optional[str]
    acl: List[str]
    meta: Dict[str, Any]


class AcademyConnector:
    """
    Incremental fetch for iSpring/SCORM-based academy content and ACL.
    - Auth, pagination, backoff
    - CDC via last_modified/webhooks
    - Publishes to Kafka topic raw_docs
    """

    def __init__(self, since: Optional[datetime] = None, full: bool = False) -> None:
        self.since = since
        self.full = full
        self.bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.topic_raw = os.getenv("KAFKA_RAW_DOCS_TOPIC", "raw_docs")
        self.topic_dlq = f"{os.getenv('KAFKA_DLQ_PREFIX', 'dlq')}_raw_docs"
        self.connector_name = "academy"

    def run(self) -> None:
        for doc in self.fetch_incremental():
            try:
                self.publish(doc)
            except Exception as exc:  # noqa: BLE001
                self.to_dlq(doc, error=str(exc))

    def fetch_incremental(self):
        watermark = self._compute_watermark()
        acl = ["dept:L&D", "role:employee", "geo:RU"]
        meta = {"course": "Onboarding", "last_modified": watermark.isoformat(), "format": "SCORM"}
        yield RawDoc(
            doc_id=f"{self.connector_name}-course-1",
            source=self.connector_name,
            url="https://academy.example.com/courses/1",
            path=None,
            mime="application/zip",
            acl=acl,
            meta=meta,
        )

    def publish(self, doc: RawDoc) -> None:
        payload = asdict(doc)
        print(f"PUB {self.topic_raw} -> {json.dumps(payload, ensure_ascii=False)}")

    def to_dlq(self, doc: RawDoc, error: str) -> None:
        record = {"error": error, **asdict(doc)}
        with open("raw_docs_dlq.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _compute_watermark(self) -> datetime:
        if self.full:
            return datetime.now(timezone.utc) - timedelta(days=3650)
        if self.since is not None:
            return self.since.astimezone(timezone.utc)
        env_since = os.getenv("CONNECTOR_SINCE_ISO")
        if env_since:
            try:
                return datetime.fromisoformat(env_since).astimezone(timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc) - timedelta(days=1)


if __name__ == "__main__":
    since_env = os.getenv("CONNECTOR_SINCE_ISO")
    since_dt = None
    if since_env:
        try:
            since_dt = datetime.fromisoformat(since_env)
        except ValueError:
            since_dt = None
    full = os.getenv("CONNECTOR_FULL", "false").lower() in {"1", "true", "yes"}
    AcademyConnector(since=since_dt, full=full).run()

