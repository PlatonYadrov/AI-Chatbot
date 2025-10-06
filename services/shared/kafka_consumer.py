from __future__ import annotations

from typing import Any, Dict, Iterable


class KafkaConsumerStub:
    """Заглушка консюмера Kafka: имитирует итератор сообщений."""

    def __init__(self, topic: str) -> None:
        self.topic = topic

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        # Ничего не возвращаем по умолчанию (пустой поток)
        if False:
            yield {"topic": self.topic, "value": {}}
        return iter(())


def get_consumer(topic: str) -> KafkaConsumerStub:
    return KafkaConsumerStub(topic)


__all__ = ["KafkaConsumerStub", "get_consumer"]

