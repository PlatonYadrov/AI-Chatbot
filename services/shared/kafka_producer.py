from __future__ import annotations

import json
import os
from typing import Any, Dict


class KafkaProducerStub:
    """Заглушка продюсера Kafka: пишет в stdout вместо брокера."""

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

    def send(self, topic: str, value: Dict[str, Any]) -> None:
        print(f"KAFKA SEND [{topic}] {json.dumps(value, ensure_ascii=False)}")


def get_producer() -> KafkaProducerStub:
    return KafkaProducerStub()


__all__ = ["KafkaProducerStub", "get_producer"]

