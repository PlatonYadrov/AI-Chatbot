# Highland AI Chat

Highland AI Chat — это каркас платформы корпоративного ассистента с RAG-контурами. Репозиторий уже включает структуру каталогов, заглушки модулей и базовую инфраструктурную обвязку, отражающие целевую архитектуру.

## Что уже сделано
- **Сервисы**: созданы директории и пустые модули для ingestion-пайплайна, online-RAG контура и административного портала, а также общих библиотек.
- **Инфраструктура**: добавлены шаблоны конфигураций для Qdrant, Kafka, Redis, Airflow, мониторинга и Postgres, а также общий `docker-compose.yml`.
- **Документация**: подготовлены файлы `docs/architecture.md` и `docs/runbook.md` как отправная точка для описания архитектуры и эксплуатационных процедур.
- **Конфигурации**: заведены файлы окружений и конфиг модели в каталоге `configs/`.
- **Тесты**: добавлены заглушки для будущих тестов извлечения и генерации вместе с `golden_set.json`.

## Что необходимо реализовать для запуска
- **Коннекторы и пайплайн**: реализовать логику подключения к источникам данных, парсинг, предобработку, чанкование, эмбеддинг и индексацию, а также оркестрацию задач.
- **Online RAG**: написать REST API (FastAPI), реализовать поиск (dense/sparse/hybrid), ре-ранжирование, сбор промпта, запросы к LLM и guardrails.
- **Админ-портал**: дописать backend-обработчики и frontend-интерфейс для загрузки документов, мониторинга задач и выгрузки отчетов.
- **Инфраструктура**: наполнить docker-compose и конфигурации конкретными параметрами, подготовить Helm-чарты/скрипты деплоя, прописать CI/CD.
- **Тестирование и наблюдаемость**: покрыть ключевые сценарии интеграционными и нагрузочными тестами, описать метрики/алерты в Prometheus/Grafana.
- **Документация**: детализировать архитектуру, эксплуатационные инструкции и API-спецификацию.

Подробнее см. `docs/architecture.md`.

## Локальный контур (E2E) — как это работает

```
Client
  │
  ├─► online-rag (FastAPI /query)
  │      │
  │      ├─► TEI embeddings (HTTP /embed)
  │      ├─► Qdrant (similarity search)
  │      └─► vLLM (OpenAI /v1/chat/completions)
  │
  └─► ingestion (FastAPI /ingest)
         │
         ├─► OCR (HTTP /ocr)
         ├─► TEI embeddings (HTTP /embed)
         └─► Qdrant (upsert)
```

### Сервисы и роли
- OCR (`services/ocr`, :9000): извлекает текст из PDF/изображений (Tesseract + pdf2image).
- Ingestion (`services/ingestion`, :6000): принимает файл, вызывает OCR, делит на чанки, векторизует и пишет в Qdrant.
- Embeddings (TEI, :8080 → :80 внутри сети): выдаёт эмбеддинги (e5-base/BGE‑M3) через `/embed`.
- Qdrant (:6333): векторная БД, коллекция `rag_chunks`.
- Online‑RAG (`services/online-rag`, :7000): принимает запрос, ищет в Qdrant, собирает промпт и зовёт LLM.
- LLM (vLLM, :8000): OpenAI‑совместимый эндпоинт `/v1/chat/completions` (Qwen3‑30B‑A3B‑AWQ).

### Каналы связи (HTTP внутри docker-compose)
- ingestion → OCR: `POST http://ocr:9000/ocr`
- ingestion/online‑rag → TEI: `POST http://embeddings:80/embed`
- ingestion/online‑rag → Qdrant: `http://qdrant:6333`
- online‑rag → vLLM: `POST http://llm:8000/v1/chat/completions`

### Конфигурация (configs/dev.env)
- `OCR_URL=http://ocr:9000/ocr`
- `EMBEDDINGS_BASE_URL=http://embeddings:80`
- `QDRANT_URL=http://qdrant:6333`
- `QDRANT_COLLECTION=rag_chunks`
- `LLM_BASE_URL=http://llm:8000/v1`
- `LLM_MODEL_NAME=cognitivecomputations/Qwen3-30B-A3B-AWQ`

### Запуск
1) Установите NVIDIA runtime (для vLLM/TEI) и при необходимости добавьте `gpus` к сервисам `llm` и `embeddings`.
2) Поднимите стек:
```bash
docker compose up -d qdrant llm embeddings ocr ingestion online_rag
```
3) Инжест документа:
```bash
curl -F "file=@sample.pdf" http://localhost:6000/ingest
```
4) Запрос в RAG:
```bash
curl -X POST http://localhost:7000/query -H "Content-Type: application/json" -d '{"query":"Ваш вопрос","k":4}'
```

### Выбор моделей и размерности
- По умолчанию TEI использует `intfloat/e5-base` (768). Для `BAAI/bge-m3` (1024) замените `--model-id` в compose и пересоздайте коллекцию Qdrant.

### Когда подключать брокер сообщений
- Текущий контур онлайн‑запросов синхронный (HTTP) ради низкой задержки.
- Для массового ingestion/инкрементов используйте Kafka/Redpanda/NATS/Redis Streams: топики `raw_docs → parsed_docs → ready_for_embedding → embeddings → index_updates`, DLQ, реплей.

## Дополнительные материалы
- Практическое пошаговое руководство по построению рабочей RAG-системы с примерами кода и
  рекомендациями по обработке DOCX/PDF/XLSX/PPTX/SCRAM доступно в `docs/practical_guide.md`.

## С чего начать разработку
- **Первым делом — ingestion.** Реализуйте один end-to-end путь (например, Confluence → Kafka → Qdrant), чтобы сформировать контракт обмена сообщениями (`raw_docs`, `parsed_docs`, `ready_for_embedding`, `embeddings`) и проверить схемы из `shared/models.py`.
- **Фокус на передачу данных между микросервисами.** Настройте публикацию/подписку в Kafka, убедитесь, что процессоры передают `ChunkReady` в эмбеддер, а индексатор подтверждает апдейты через Redis Pub/Sub (`index_updated`).
- **Далее — минимальный online RAG.** После появления данных в Qdrant поднимите `gateway` + `query_embedder` + `dense_search` + `llm_service`, чтобы ответить на базовые запросы и проверить фильтры ACL.
- **Затем развивайте сервисы.** Добавляйте hybrid-поиск, reranker, guardrails, подключайте дополнительные коннекторы и наращивайте админку.
