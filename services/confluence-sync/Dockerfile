FROM python:3.11-slim

# Установка cron и tzdata для запуска по расписанию и корректной таймзоны
RUN apt-get update && apt-get install -y --no-install-recommends cron tzdata ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Таймзона по умолчанию (можно переопределить переменной TZ)
ENV TZ=Europe/Moscow

WORKDIR /app

# Копируем приложение
COPY services/confluence-sync/app.py /app/app.py

# Python-зависимости через pip (как просили)
RUN pip install --no-cache-dir \
    requests~=2.32 \
    loguru~=0.7 \
    "SQLAlchemy[asyncio]~=2.0" \
    asyncpg~=0.29

# Крон: ежедневный запуск в 03:00, логи в /var/log/cron.log
RUN echo 'SHELL=/bin/bash' > /etc/cron.d/confluence_sync && \
    echo 'PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' >> /etc/cron.d/confluence_sync && \
    echo '0 3 * * * root /usr/local/bin/python -u /app/app.py >> /var/log/cron.log 2>&1' >> /etc/cron.d/confluence_sync && \
    chmod 0644 /etc/cron.d/confluence_sync && \
    touch /var/log/cron.log

# Запуск cron в foreground, чтобы контейнер не завершался
CMD ["bash", "-lc", "cron -f"]


