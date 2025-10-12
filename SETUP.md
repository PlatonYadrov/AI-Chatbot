# AI-Chatbot - Setup Guide

## 🚀 Быстрый старт

### Первый запуск (3 команды)

```bash
# 1. Сборка образов
make build

# 2. Загрузка моделей Docling (один раз, ~500MB)
make download-models

# 3. Запуск всех сервисов
make up
```

### Проверка работы

```bash
# Тест загрузки документа
make test-ingestion

# Проверка здоровья сервисов
make health

# Просмотр логов
make logs-ingestion
```

---

## 📦 Что запускается

| Сервис | Порт | Описание |
|--------|------|----------|
| **ingestion** | 6000 | Парсинг документов (Docling, локально) |
| **online-rag** | 7000 | RAG API |
| **llm** | 8000 | Qwen3-30B-A3B-AWQ (2x GPU) |
| **embeddings** | 8080 | e5-base embeddings (1x GPU) |
| **qdrant** | 6333 | Vector database |
| **ocr** | 9000 | OCR service |

---

## 🔧 Docling (локальные модели)

### Что это?

Docling - парсер документов от IBM Research, работающий **полностью локально**:
- ✅ PDF, DOCX, PPTX, XLSX, изображения
- ✅ Встроенный OCR (Tesseract)
- ✅ Извлечение таблиц
- ✅ Анализ структуры документа
- ✅ Никаких внешних API

### Где хранятся модели?

Модели (~500MB) хранятся в Docker volume `docling_models`:

```bash
# Посмотреть модели
docker volume inspect ai-chatbot_docling_models

# Удалить модели (понадобится перезагрузка)
docker volume rm ai-chatbot_docling_models
```

### Работа без интернета

После `make download-models` все работает офлайн:

```bash
# Отключите интернет и проверьте
make up
make test-ingestion  # Должно работать!
```

---

## 📋 Доступные команды

### Основные

```bash
make build              # Сборка образов
make up                 # Запуск всех сервисов
make down               # Остановка
make restart            # Перезапуск
make logs               # Логи всех сервисов
make logs-ingestion     # Логи ingestion
make status             # Статус сервисов
```

### Модели и тесты

```bash
make download-models    # Загрузка Docling моделей (один раз)
make test-ingestion     # Тест API
make test-pdf           # Тест PDF
make test-docx          # Тест DOCX
make test-pptx          # Тест PPTX
make health             # Проверка здоровья
```

### Очистка

```bash
make clean              # Удалить volumes (включая модели!)
make clean-cache        # Очистить Docker cache
```

### Разработка

```bash
make dev-ingestion      # Запуск ingestion локально
make dev-rag            # Запуск RAG локально
```

### Полная установка

```bash
make setup              # build + download-models + up
```

---

## 🔍 Примеры использования

### Загрузка PDF

```bash
curl -X POST http://localhost:6000/ingest \
  -F "file=@tests/pdf/Knoleges.pdf" \
  | jq
```

Ответ:
```json
{
  "status": "ok",
  "doc_id": "a1b2c3d4e5f6g7h8",
  "chunks_total": 123,
  "chunks_unique": 98
}
```

### Загрузка DOCX

```bash
curl -X POST http://localhost:6000/ingest \
  -F "file=@document.docx" \
  | jq
```

### RAG запрос

```bash
curl -X POST http://localhost:7000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Что такое машинное обучение?",
    "k": 4
  }' | jq
```

---

## 🐛 Устранение проблем

### Модели не загружены

```bash
# Загрузите заново
make download-models
```

### Ingestion не работает

```bash
# Проверьте логи
make logs-ingestion

# Перезапустите
docker-compose restart ingestion
```

### Нехватка памяти

Docling требует ~4GB RAM. Увеличьте в `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

### LLM не запускается

Проверьте GPU:
```bash
nvidia-smi
```

Убедитесь, что модель скачана:
```bash
ls -la /mnt/hf_cache/models--cognitivecomputations--Qwen3-30B-A3B-AWQ/
```

---

## 📂 Структура volumes

```
ai-chatbot_docling_models   # Модели Docling (~500MB)
ai-chatbot_hf_cache         # LLM и embeddings кеш
ai-chatbot_qdrant_storage   # Векторная БД
ai-chatbot_rag_data         # Загруженные документы
```

Посмотреть:
```bash
docker volume ls | grep ai-chatbot
```

---

## ⚙️ Конфигурация

### configs/dev.env

Основная конфигурация всех сервисов:

```bash
# LLM
LLM_BASE_URL=http://llm:8000/v1
LLM_MODEL_NAME=cognitivecomputations/Qwen3-30B-A3B-AWQ

# Embeddings
EMBEDDINGS_BASE_URL=http://embeddings:80
EMBEDDINGS_MODEL=intfloat/e5-base

# Docling (локальные модели)
DOCLING_OFFLINE_MODE=true
DOCLING_OCR_ENABLED=true
DOCLING_EXTRACT_TABLES=true

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=rag_chunks
```

---

## 🎯 Production deployment

### 1. Сборка с моделями

Раскомментируйте в `services/ingestion/Dockerfile`:

```dockerfile
RUN python download_models.py || echo "Models will download on first run"
```

Пересоберите:
```bash
make build
```

### 2. Air-gapped развертывание

```bash
# На машине с интернетом
make build
make download-models
docker save -o images.tar $(docker images -q)

# Экспорт volumes
docker run --rm -v ai-chatbot_docling_models:/data -v $(pwd):/backup \
  alpine tar czf /backup/docling_models.tar.gz -C /data .

# Перенести images.tar и docling_models.tar.gz на целевую машину

# На целевой машине
docker load -i images.tar
docker volume create ai-chatbot_docling_models
docker run --rm -v ai-chatbot_docling_models:/data -v $(pwd):/backup \
  alpine tar xzf /backup/docling_models.tar.gz -C /data
```

---

## 📚 Документация

- **DOCLING_INTEGRATION_SUMMARY.md** - Полная сводка интеграции Docling
- **services/ingestion/README_DOCLING.md** - Подробный гайд по Docling
- **services/ingestion/DOCLING_MIGRATION.md** - Миграция со старых парсеров
- **services/ingestion/DOCLING_LOCAL_DEPLOYMENT.md** - Локальное развертывание

---

## 🆘 Помощь

```bash
make help    # Список всех команд
make status  # Текущий статус
make health  # Проверка здоровья
make logs    # Все логи
```

---

**Готово!** Все модели локальные, никаких внешних API. 🚀

