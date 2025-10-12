# AI-Chatbot - Quick Start Guide

## 🎯 Текущая конфигурация

### ✅ Что работает:

1. **vLLM (Qwen3-30B-AWQ)** - 2x L4 GPU
2. **Embeddings (e5-base)** - GPU 1
3. **Docling Parser** - CPU (все модели)
   - Layout Detection (структура)
   - TableFormer (таблицы 95% точность)
   - Tesseract OCR (русский + английский)
   - Formula Detection
   - Code Detection
4. **Qdrant** - векторная БД
5. **RAG API** - поиск и генерация

### ❌ Что отключено:

- **VLM** (нет VRAM на L4, требует +4GB)

---

## 🚀 Запуск за 4 шага

### Шаг 1: Проверка Qwen3-30B-AWQ весов

```bash
# Проверьте наличие модели
ls -lh /mnt/hf_cache/models--cognitivecomputations--Qwen3-30B-A3B-AWQ/

# Если нет - скачайте (ПЕРЕД docker-compose!)
huggingface-cli download cognitivecomputations/Qwen3-30B-A3B-AWQ \
  --local-dir /mnt/hf_cache/models--cognitivecomputations--Qwen3-30B-A3B-AWQ \
  --local-dir-use-symlinks False

# Проверьте размер (~18GB)
du -sh /mnt/hf_cache/models--cognitivecomputations--Qwen3-30B-A3B-AWQ/
```

### Шаг 2: Сборка образов

```bash
# Соберите все Docker образы
make build

# Или вручную:
docker-compose build
```

Займет: ~10-15 минут

### Шаг 3: Скачивание Docling моделей

```bash
# Скачайте модели Docling (~600MB, один раз)
make download-models

# Или вручную:
docker-compose run --rm ingestion python download_models_full.py
```

Займет: ~5-10 минут (зависит от интернета)

**Модели скачаются в:** `docling_models` Docker volume

### Шаг 4: Запуск всех сервисов

```bash
# Запустите все
make up

# Или вручную:
docker-compose up -d
```

---

## 📊 Проверка статуса

### Проверка GPU

```bash
# Смотрите использование VRAM
watch -n 1 nvidia-smi

# Ожидаемое:
# GPU 0: ~19-20GB / 24GB (vLLM)
# GPU 1: ~22-23GB / 24GB (vLLM + Embeddings)
```

### Проверка сервисов

```bash
# Статус контейнеров
docker-compose ps

# Логи
make logs-ingestion  # Ingestion
make logs-llm        # vLLM
docker-compose logs -f online_rag  # RAG

# Проверка здоровья
make health
```

### Проверка API

```bash
# 1. Проверка LLM
curl http://localhost:8000/v1/models

# 2. Проверка Embeddings
curl http://localhost:8080/health

# 3. Проверка Qdrant
curl http://localhost:6333/collections

# 4. Тест загрузки документа
make test-ingestion

# Или вручную:
curl -F "file=@tests/pdf/Knoleges.pdf" http://localhost:6000/ingest
```

---

## 🎯 Что произойдет при запуске:

### 1. vLLM загрузит модель

```
Loading model: Qwen3-30B-A3B-AWQ
Tensor parallel size: 2
Loading weights... (займет 2-3 минуты)
✓ Model loaded successfully
Server ready at http://0.0.0.0:8000
```

### 2. Embeddings загрузится

```
Loading e5-base model...
✓ Model loaded
Server ready at http://0.0.0.0:80
```

### 3. Ingestion использует кешированные модели

```
Loading Docling models from cache...
✓ Layout model loaded
✓ TableFormer loaded
✓ OCR ready (Tesseract)
✓ Formula detection ready
✓ Code detection ready
Server ready at http://0.0.0.0:6000
```

---

## ⚠️ Важные замечания:

### 1. Веса Qwen3 НЕ скачиваются автоматически!

Если при запуске vLLM ошибка:
```
Model not found: cognitivecomputations/Qwen3-30B-A3B-AWQ
```

**Решение:**
```bash
# Остановите
docker-compose down

# Скачайте модель
huggingface-cli download cognitivecomputations/Qwen3-30B-A3B-AWQ \
  --local-dir /mnt/hf_cache/models--cognitivecomputations--Qwen3-30B-A3B-AWQ

# Перезапустите
make up
```

### 2. Docling модели скачаются автоматически

Если забыли `make download-models`, модели скачаются при первом использовании.

**Но лучше скачать заранее!**

### 3. Первый запуск медленнее

- vLLM: загрузка весов ~2-3 минуты
- Первый inference: ~10-20 секунд (компиляция CUDA kernels)
- Последующие запросы: быстро (~40-60 tok/sec)

---

## 🧪 Тестирование

### Тест 1: Загрузка PDF

```bash
curl -X POST http://localhost:6000/ingest \
  -F "file=@tests/pdf/Knoleges.pdf" | jq

# Ожидаемый ответ:
{
  "status": "ok",
  "doc_id": "a1b2c3d4...",
  "chunks_total": 123,
  "chunks_unique": 98
}
```

### Тест 2: RAG запрос

```bash
curl -X POST http://localhost:7000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Что такое машинное обучение?",
    "k": 4
  }' | jq

# Получите ответ с источниками
```

### Тест 3: LLM напрямую

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cognitivecomputations/Qwen3-30B-A3B-AWQ",
    "messages": [{"role": "user", "content": "Привет!"}]
  }' | jq
```

---

## 📈 Мониторинг

### GPU

```bash
# Реальное время
watch -n 1 nvidia-smi

# Только VRAM
watch -n 1 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv'
```

### Docker

```bash
# Ресурсы контейнеров
docker stats

# Логи
docker-compose logs -f --tail=100
```

### Производительность

```bash
# vLLM метрики
curl http://localhost:8000/metrics

# Embeddings метрики
curl http://localhost:8080/metrics
```

---

## 🛠 Частые проблемы

### Проблема 1: OOM (Out of Memory) на GPU

**Симптомы:**
```
CUDA out of memory
```

**Решение:**
```yaml
# В docker-compose.yml уменьшите:
--gpu-memory-utilization 0.85  # Было 0.92
--max-model-len 6000           # Было 8000
```

### Проблема 2: vLLM не запускается

**Симптомы:**
```
Model not found
```

**Решение:**
```bash
# Проверьте путь к моделям
ls /mnt/hf_cache/

# Убедитесь что модель там есть
docker-compose run --rm llm ls /root/.cache/
```

### Проблема 3: Docling модели не загружаются

**Симптомы:**
```
Failed to download model from HuggingFace
```

**Решение:**
```bash
# Скачайте вручную
make download-models

# Проверьте volume
docker volume inspect ai-chatbot_docling_models
```

---

## 🎉 Готово!

После успешного запуска:

- ✅ **Ingestion API:** http://localhost:6000
- ✅ **RAG API:** http://localhost:7000
- ✅ **LLM API:** http://localhost:8000
- ✅ **Embeddings:** http://localhost:8080
- ✅ **Qdrant:** http://localhost:6333

---

## 📚 Дополнительная документация

- **SETUP.md** - Подробная установка
- **GPU_ALLOCATION.md** - Распределение GPU
- **MODELS_INFO.md** - Информация о моделях
- **services/ingestion/README_DOCLING.md** - Docling детали

---

## 💡 Полезные команды

```bash
make help              # Все команды
make status            # Статус сервисов
make restart           # Перезапуск
make clean             # Очистка (удалит модели!)
make logs-ingestion    # Логи ingestion
make test-pdf          # Тест PDF
make test-docx         # Тест DOCX
```

