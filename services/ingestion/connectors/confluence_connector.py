from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, Iterable, List, Optional


@dataclass
class RawDoc:
    doc_id: str
    source: str
    url: Optional[str]
    path: Optional[str]
    mime: Optional[str]
    acl: List[str]
    meta: Dict[str, Any]


class ConfluenceConnector:
    """
    Incremental content fetcher for Confluence.

    Responsibilities:
    - Auth, pagination, retry with backoff
    - CDC via last_modified watermark; extract ACL and metadata
    - Publish messages to Kafka topic raw_docs

    Input: triggers from scheduler (HTTP/args) or CRON/webhooks
    Output: Kafka topic raw_docs (msg: {doc_id, source, url|path, mime, acl, meta})
    Errors: retry + DLQ raw_docs_dlq (local JSONL fallback if Kafka unavailable)
    """

    def __init__(self, since: Optional[datetime] = None, full: bool = False) -> None:
        self.since = since
        self.full = full
        self.bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.topic_raw = os.getenv("KAFKA_RAW_DOCS_TOPIC", "raw_docs")
        self.topic_dlq = f"{os.getenv('KAFKA_DLQ_PREFIX', 'dlq')}_raw_docs"
        self.connector_name = "confluence"

    def run(self) -> None:
        for doc in self.fetch_incremental():
            try:
                self.publish(doc)
            except Exception as exc:  # noqa: BLE001
                self.to_dlq(doc, error=str(exc))

    def fetch_incremental(self) -> Generator[RawDoc, None, None]:
        """
        Placeholder incremental fetch.
        Replace with Confluence REST API calls with pagination and lastModified filtering.
        """
        watermark = self._compute_watermark()
        # Example placeholder item
        acl = ["dept:HR", "role:employee", "geo:RU"]
        meta = {"space": "HR", "last_modified": watermark.isoformat()}
        yield RawDoc(
            doc_id=f"{self.connector_name}-example-1",
            source=self.connector_name,
            url="https://confluence.example.com/pages/1",
            path=None,
            mime="text/html",
            acl=acl,
            meta=meta,
        )

    def publish(self, doc: RawDoc) -> None:
        """Minimal publisher stub; replace with Kafka producer integration."""
        payload = asdict(doc)
        line = json.dumps(payload, ensure_ascii=False)
        # For minimal setup, print to stdout; real impl should send to Kafka
        print(f"PUB {self.topic_raw} -> {line}")

    def to_dlq(self, doc: RawDoc, error: str) -> None:
        record = {"error": error, **asdict(doc)}
        with open("raw_docs_dlq.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _compute_watermark(self) -> datetime:
        if self.full:
            # Full sync ignores watermark in real implementations
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


def _with_backoff(fn, *, retries: int = 3, base_sleep: float = 1.0):
    def wrapper(*args, **kwargs):
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception:  # noqa: BLE001
                if attempt >= retries:
                    raise
                sleep = base_sleep * (2**attempt)
                time.sleep(sleep)
                attempt += 1
    return wrapper


if __name__ == "__main__":
    since_env = os.getenv("CONNECTOR_SINCE_ISO")
    since_dt = None
    if since_env:
        try:
            since_dt = datetime.fromisoformat(since_env)
        except ValueError:
            since_dt = None
    full = os.getenv("CONNECTOR_FULL", "false").lower() in {"1", "true", "yes"}
    connector = ConfluenceConnector(since=since_dt, full=full)
    run_with_backoff = _with_backoff(connector.run)
    run_with_backoff()

