from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def log_info(msg: str) -> None:
    print(f"INFO {msg}")


def log_error(msg: str) -> None:
    print(f"ERROR {msg}")


@contextmanager
def timer(metric_name: str) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        dur = (time.time() - start) * 1000.0
        print(f"METRIC {metric_name}={dur:.2f}ms")


__all__ = ["log_info", "log_error", "timer"]

