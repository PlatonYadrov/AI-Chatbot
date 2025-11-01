from __future__ import annotations

"""Сервис одноразовой синхронизации вложений Confluence.

Запускается по расписанию через cron внутри контейнера. Выполняет один проход:
- Делает один HTTP-запрос к Confluence для получения списка вложений страницы
- Для новых/обновлённых вложений (по номеру версии из `_links.download`):
  скачивает файл во временный каталог под исходным именем, отправляет его
  на эндпоинт `ingestion:/ingest` для парсинга/чанкирования/векторизации,
  сохраняет последнюю обработанную версию в PostgreSQL
- Удаляет временный файл/каталог

Все комментарии и docstring на русском языке.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import requests
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


# Параметры окружения (задаются через docker-compose)
CONFLUENCE_HOST: str = os.environ["HOST"].rstrip("/")
CONFLUENCE_TOKEN: str = os.environ["TOKEN"]
PAGE_ID: str = os.environ["PAGE_ID"]
VERIFY_SSL: bool = os.environ.get("VERIFY_SSL", "false").lower() in {"1", "true", "yes"}

INGEST_URL: str = os.environ.get("INGEST_URL", "http://ingestion:6000/ingest")

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://rag_user:rag_password@postgres:5432/rag_chatbot",
)


def _version_from_download_path(dl_path: str) -> int:
    """Извлечь номер версии вложения из `_links.download`.

    Args:
        dl_path: Значение `_links.download` (относительный путь с query).

    Returns:
        Номер версии (0, если извлечь не удалось).
    """
    qs = parse_qs(urlparse(dl_path).query)
    try:
        return int(qs.get("version", ["0"])[0] or 0)
    except Exception:
        return 0


def _fetch_attachments() -> List[Dict[str, Any]]:
    """Получить список вложений (один HTTP-запрос, без пагинации).

    Returns:
        Массив объектов вложений из Confluence (`results`).
    """
    url = f"{CONFLUENCE_HOST}/rest/api/content/{PAGE_ID}/child/attachment?limit=200"
    headers = {
        "Authorization": f"Bearer {CONFLUENCE_TOKEN}",
        "Accept": "application/json",
    }
    resp = requests.get(url, headers=headers, verify=VERIFY_SSL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return list(data.get("results", []))


def _download_to_temp(download_path: str, filename: str) -> Path:
    """Скачать вложение во временный каталог под исходным именем.

    Args:
        download_path: Значение `_links.download` (относительный URL).
        filename: Имя файла (из `title`).

    Returns:
        Путь до временного файла.
    """
    url = f"{CONFLUENCE_HOST}{download_path}"
    headers = {"Authorization": f"Bearer {CONFLUENCE_TOKEN}"}
    with requests.get(url, headers=headers, verify=VERIFY_SSL, timeout=300, stream=True) as r:
        r.raise_for_status()
        tmpdir = Path(tempfile.mkdtemp(prefix="conf_dl_"))
        dest = tmpdir / filename
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return dest


def _ingest_file(path: Path, filename: str) -> None:
    """Отправить файл в сервис `ingestion` на `/ingest`.

    Args:
        path: Путь до файла.
        filename: Имя файла для multipart.
    """
    with open(path, "rb") as f:
        files = {"file": (filename, f)}
        r = requests.post(INGEST_URL, files=files, timeout=900)
        r.raise_for_status()


async def _ensure_table(engine: AsyncEngine) -> None:
    """Создать минимальную таблицу учёта версий (если отсутствует).

    Таблица хранит только `attachment_id` и последнюю обработанную `version`.
    Этого достаточно, чтобы определять новые/обновлённые вложения без лишней логики.
    """
    ddl = (
        """
        CREATE TABLE IF NOT EXISTS confluence_attachments (
            attachment_id BIGINT PRIMARY KEY,
            version INT NOT NULL
        );
        """
    )
    async with engine.begin() as conn:
        await conn.execute(text(ddl))


async def _get_current_version(session: AsyncSession, attachment_id: int) -> Optional[int]:
    """Получить последнюю обработанную версию вложения из БД.

    Args:
        session: Асинхронная сессия БД.
        attachment_id: Идентификатор вложения.

    Returns:
        Версия (int) или None, если записи нет.
    """
    q = text("SELECT version FROM confluence_attachments WHERE attachment_id = :aid")
    res = await session.execute(q, {"aid": attachment_id})
    row = res.first()
    return int(row[0]) if row else None


async def _upsert_version(session: AsyncSession, attachment_id: int, version: int) -> None:
    """Сохранить новую версию вложения (upsert).

    Args:
        session: Асинхронная сессия БД.
        attachment_id: Идентификатор вложения.
        version: Новая версия.
    """
    q = text(
        """
        INSERT INTO confluence_attachments (attachment_id, version)
        VALUES (:aid, :ver)
        ON CONFLICT (attachment_id) DO UPDATE SET version = EXCLUDED.version
        """
    )
    await session.execute(q, {"aid": attachment_id, "ver": version})
    await session.commit()


async def sync_once() -> None:
    """Один проход синхронизации: запрос JSON → цикл → новые/обновлённые → индексация.

    Шаги:
    1) Получить список вложений.
    2) Для каждого извлечь версию из `_links.download`.
    3) Если версия новее — скачать, отправить в `/ingest`, зафиксировать версию в БД.
    4) Удалить временные файлы.
    """
    logger.remove()
    logger.add(__import__("sys").stdout, level=os.environ.get("LOG_LEVEL", "INFO"))

    engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    await _ensure_table(engine)
    async_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    attachments = _fetch_attachments()
    if not attachments:
        logger.info("вложений не найдено: page_id={}", PAGE_ID)
        return

    async with async_factory() as session:
        processed = 0
        for raw in attachments:
            try:
                aid = int(raw["id"])
                title = str(raw["title"])
                dl_path = str(raw["_links"]["download"])
                ver = _version_from_download_path(dl_path)

                current = await _get_current_version(session, aid)
                if current is not None and ver <= current:
                    logger.info("пропуск id={} title='{}' ver={} <= текущая={}", aid, title, ver, current)
                    continue

                logger.info("скачивание id={} title='{}' ver={}", aid, title, ver)
                tmp: Optional[Path] = None
                try:
                    tmp = _download_to_temp(dl_path, title)
                    _ingest_file(tmp, title)
                    await _upsert_version(session, aid, ver)
                    processed += 1
                finally:
                    if tmp:
                        # Удаляем временную директорию целиком
                        try:
                            for p in tmp.parent.glob("*"):
                                p.unlink(missing_ok=True)
                            tmp.parent.rmdir()
                        except Exception:
                            pass
            except Exception as e:
                # Продолжаем обработку остальных элементов, фиксируем предупреждение
                logger.warning("ошибка обработки элемента: {} (attachment_id={})", repr(e), raw.get("id"))

        logger.info("синхронизация завершена: всего={}, обработано={}", len(attachments), processed)


if __name__ == "__main__":
    asyncio.run(sync_once())


