# 🚀 Checklist для Запуска AI-Chatbot

## ✅ Готовность Системы

### 1. Код и Конфигурация
- [x] API интегрирован с DoclingChunker
- [x] Токенизатор автоопределяется по модели эмбеддингов
- [x] Конфигурация в `docker-compose.yml` и `configs/dev.env`
- [x] Все зависимости в `requirements.txt`
- [x] Dockerfile настроен

### 2. Локальные Модели (Все работают БЕЗ интернета!)

#### ✅ Docling Models (загружаются автоматически при первом запуске)
```
~/.cache/huggingface/hub/ или /app/.cache/huggingface/
├── DocLayNet Layout Model (~200MB) ✅ локально
├── TableFormer Model (~150MB)      ✅ локально
├── Tesseract OCR (~50MB)           ✅ локально (apt install)
└── Formula/Code detectors (~80MB)  ✅ локально
```

#### ✅ Embeddings Model
```
TEI контейнер с intfloat/e5-base (~500MB)
├── Запускается: docker-compose
├── GPU: L4 #1
└── Статус: ✅ локально
```

#### ✅ LLM Model
```
vLLM контейнер с Qwen3-30B-A3B-AWQ (~20GB)
├── Запускается: docker-compose
├── GPU: L4 #0 + L4 #1 (tensor parallel)
└── Статус: ✅ локально
```

### 3. Chunking Tokenizer
```
Токенизатор: bert-base-uncased (автоопределение)
├── Для модели: intfloat/e5-base
├── Размер: ~1MB
└── Статус: ✅ локально (tiktoken/transformers)
```

---

## 🎯 Что Нужно Сделать для Запуска

### Шаг 1: Установить Зависимости (если еще не установлены)

```bash
# В директории проекта
cd services/ingestion
pip install -r requirements.txt
```

**Критичные зависимости:**
- `docling>=2.0.0` - парсинг документов
- `docling-core[chunking]>=2.0.0` - HybridChunker
- `tiktoken` - токенизация

### Шаг 2: Загрузить Docling Модели (опционально, но рекомендуется)

```bash
# Предзагрузка моделей (чтобы не ждать при первом запуске)
python services/ingestion/download_models.py
```

Это загрузит:
- DocLayNet Layout Model
- TableFormer Model
- Formula/Code detectors

**Альтернатива:** Модели загрузятся автоматически при первом использовании.

### Шаг 3: Запустить Docker Compose

```bash
# В корне проекта
docker-compose up -d

# Или для ребилда
docker-compose up -d --build
```

Запустятся:
1. `ingestion` - Сервис обработки документов (:6000)
2. `embeddings` - TEI с e5-base (:8080)
3. `llm` - vLLM с Qwen3-30B (:8000)
4. `qdrant` - Векторная БД (:6333)
5. `online_rag` - RAG сервис (:7000)
6. `ocr` - OCR сервис (:9000)

### Шаг 4: Проверить Логи

```bash
# Проверить запуск ingestion
docker-compose logs ingestion

# Ожидаемый вывод:
# INFO:ingestion.api - Starting application
# INFO:docling - Models loading...
# INFO:ingestion.api - Server started on 0.0.0.0:6000
```

### Шаг 5: Проверить Здоровье Сервисов

```bash
# Embeddings
curl http://localhost:8080/health
# Ожидается: {"status":"ok"}

# LLM
curl http://localhost:8000/health
# Ожидается: {"status":"ok"}

# Qdrant
curl http://localhost:6333/collections
# Ожидается: {"collections":[...]}

# Ingestion (после первого запроса)
curl http://localhost:6000/
# Ожидается: FastAPI документация
```

---

## 🧪 Тест: Загрузка Документа

### С Traditional Chunking (по умолчанию)

```bash
curl -X POST http://localhost:6000/ingest \
  -F "file=@tests/pdf/Knoleges.pdf"
```

**Ожидаемый вывод:**
```json
{
  "status": "ok",
  "doc_id": "a3f5b2c1...",
  "chunks_total": 45,
  "chunks_unique": 42
}
```

**Логи (docker-compose logs ingestion):**
```
INFO docling_parse_start doc_id=a3f5b2c1 path=/data/tmp/Knoleges.pdf
INFO docling_conversion_complete doc_id=a3f5b2c1 num_pages=12
INFO traditional_chunking_complete chunks=45 strategy=traditional
INFO chunks_ready chunks_total=45 chunks_unique=42
INFO qdrant_upsert_ok collection=rag_chunks count=42
```

### С Docling Chunking (структурно-осознанный)

1. Включить в `configs/dev.env`:
```bash
USE_DOCLING_CHUNKING=true
```

2. Перезапустить:
```bash
docker-compose restart ingestion
```

3. Загрузить документ:
```bash
curl -X POST http://localhost:6000/ingest \
  -F "file=@tests/pdf/Knoleges.pdf"
```

**Ожидаемый вывод:**
```json
{
  "status": "ok",
  "doc_id": "a3f5b2c1...",
  "chunks_total": 38,
  "chunks_unique": 36
}
```

**Логи (с HybridChunker):**
```
INFO docling_chunking_enabled max_tokens=512 tokenizer=bert-base-uncased embeddings_model=intfloat/e5-base
INFO docling_parse_to_document doc_id=a3f5b2c1
INFO docling_chunker_initialized tokenizer=bert-base-uncased max_tokens=512
INFO docling_chunking_complete doc_id=a3f5b2c1 chunks=38 strategy=hybrid
INFO chunks_ready chunks_total=38 chunks_unique=36
INFO qdrant_upsert_ok collection=rag_chunks count=36
```

**Разница:**
- Traditional: 45 чанков (может разрывать контекст)
- Hybrid: 38 чанков (структурно-осознанные, лучше качество)

---

## 📊 Ожидаемые Метрики

### Время Обработки (PDF ~10 страниц)

```
Парсинг (Docling):           ~5-10 сек
Chunking (Traditional):      ~0.5 сек
Chunking (Hybrid):           ~1-2 сек
Embedding (e5-base):         ~2-3 сек
Qdrant Upsert:               ~0.5 сек
────────────────────────────────────
ИТОГО Traditional:           ~8-14 сек
ИТОГО Hybrid:                ~9-16 сек
```

### Использование Памяти

```
Docker Контейнеры:
├── ingestion:     ~2-4 GB RAM
├── embeddings:    ~3-4 GB VRAM (GPU L4 #1)
├── llm:           ~22-24 GB VRAM (GPU L4 #0+#1)
├── qdrant:        ~1-2 GB RAM
└── ocr:           ~500 MB RAM
────────────────────────────────────
ИТОГО:             ~6-8 GB RAM + ~26-28 GB VRAM
```

---

## ✅ Проверка: Все Ли Локально?

### Да, ВСЕ работает локально! ✅

```
┌─────────────────────────────────────────────────┐
│ Модель                  │ Локально │ Размер    │
├─────────────────────────────────────────────────┤
│ DocLayNet Layout        │    ✅    │ ~200 MB   │
│ TableFormer             │    ✅    │ ~150 MB   │
│ Tesseract OCR           │    ✅    │ ~50 MB    │
│ Formula/Code detectors  │    ✅    │ ~80 MB    │
│ BERT tokenizer          │    ✅    │ ~1 MB     │
│ intfloat/e5-base        │    ✅    │ ~500 MB   │
│ Qwen3-30B-A3B-AWQ       │    ✅    │ ~20 GB    │
└─────────────────────────────────────────────────┘

Внешние API: НЕТ! ❌
Интернет нужен: Только для первоначальной загрузки моделей
Offline работа: ✅ ДА (DOCLING_OFFLINE_MODE=true)
```

---

## 🎯 Итоговый Checklist

### Перед Запуском
- [ ] Docker и docker-compose установлены
- [ ] NVIDIA GPU доступен (для GPU режима)
- [ ] Порты свободны: 6000, 6333, 7000, 8000, 8080, 9000
- [ ] Достаточно места: ~30 GB для моделей

### Запуск
```bash
# 1. Установить зависимости (опционально, для локальной разработки)
cd services/ingestion && pip install -r requirements.txt

# 2. Предзагрузить Docling модели (опционально, но рекомендуется)
python services/ingestion/download_models.py

# 3. Запустить Docker Compose
docker-compose up -d --build

# 4. Проверить логи
docker-compose logs -f ingestion

# 5. Дождаться загрузки моделей (если не предзагрузили)
# Первый запуск: ~5-10 минут (загрузка моделей)
# Последующие запуски: ~30-60 секунд

# 6. Тест
curl -X POST http://localhost:6000/ingest \
  -F "file=@tests/pdf/Knoleges.pdf"
```

### После Запуска
- [ ] Все контейнеры запущены (`docker-compose ps`)
- [ ] Логи без критичных ошибок
- [ ] Health checks проходят
- [ ] Тестовая загрузка успешна

---

## 🐛 Troubleshooting

### Проблема 1: Import Error "HybridChunker not found"
```bash
# В контейнере ingestion:
docker-compose exec ingestion pip install 'docling-core[chunking]>=2.0.0'
docker-compose restart ingestion
```

### Проблема 2: Docling модели не загружаются
```bash
# Проверить интернет соединение при первом запуске
# Или загрузить вручную:
python services/ingestion/download_models.py
```

### Проблема 3: Out of Memory (GPU)
```bash
# В docker-compose.yml уменьшить:
--gpu-memory-utilization 0.92  →  0.85
--max-model-len 8000  →  6000
```

### Проблема 4: Chunking падает
```bash
# Автоматический fallback на traditional chunking
# Проверить логи:
docker-compose logs ingestion | grep chunking_fallback
```

---

## 🎉 Готово!

Система полностью готова к запуску:
- ✅ Код интегрирован
- ✅ Конфигурация настроена
- ✅ Все модели локальные
- ✅ Токенизатор соответствует embeddings
- ✅ Fallback механизмы работают

**Команда для запуска:**
```bash
docker-compose up -d --build
```

**Ожидаемое время:**
- Первый запуск: ~10-15 минут (загрузка моделей)
- Последующие: ~1-2 минуты

**Первый тест:**
```bash
curl -X POST http://localhost:6000/ingest \
  -F "file=@tests/pdf/Knoleges.pdf"
```

Успешного запуска! 🚀

