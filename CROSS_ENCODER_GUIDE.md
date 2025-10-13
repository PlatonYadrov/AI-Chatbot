# Cross-Encoder Reranking - Руководство по использованию

## 🎯 Обзор

Cross-Encoder Reranking - это компонент для улучшения качества поиска в RAG пайплайне. Он использует предобученные модели для более точного ранжирования результатов поиска по релевантности к запросу.

## 🏗️ Архитектура

```
Query → Vector Search (Qdrant) → Cross-Encoder Reranking → LLM Generation
```

### Как это работает:
1. **Initial Retrieval**: Получаем больше кандидатов из векторного поиска (обычно k*3)
2. **Reranking**: Cross-Encoder оценивает релевантность каждого кандидата к запросу
3. **Final Selection**: Выбираем топ-k наиболее релевантных результатов
4. **Generation**: Передаем отранжированные результаты в LLM

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
cd services/online-rag
pip install sentence-transformers torch numpy
```

### 2. Конфигурация

#### В `configs/dev.env`:
```bash
# Cross-Encoder Reranking Configuration
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
CROSS_ENCODER_DEVICE=cuda
CROSS_ENCODER_CACHE_DIR=/app/.cache/huggingface
```

#### В `configs/model_config.yaml`:
```yaml
reranking:
  model_name: cross-encoder/ms-marco-MiniLM-L-12-v2
  device: cuda
  max_length: 512
  batch_size: 16
  enabled: true
```

### 3. Использование в API

```python
# POST /query
{
    "query": "What is machine learning?",
    "k": 4,
    "use_reranking": true,
    "rerank_top_k": 3
}
```

## 📊 Рекомендуемые модели

### 🥇 Основная рекомендация: `cross-encoder/ms-marco-MiniLM-L-12-v2`

**Преимущества:**
- ✅ Размер: ~50MB (быстрая загрузка)
- ✅ Производительность: SOTA на MS MARCO
- ✅ Мультиязычность: Хорошо работает с русским
- ✅ Скорость: ~100ms на пару query-document
- ✅ VRAM: ~1-2GB

**Использование:**
```python
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    device="cuda",
    max_length=512
)
```

### 🥈 Альтернатива: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Преимущества:**
- ✅ Размер: ~25MB (еще быстрее)
- ✅ Скорость: ~50ms на пару
- ✅ VRAM: ~500MB-1GB
- ✅ Хорошо для production нагрузки

### 🥉 Для русского языка: Fine-tuning

**Стратегия:**
1. Взять базовую модель `cross-encoder/ms-marco-MiniLM-L-12-v2`
2. Дообучить на русских данных
3. Использовать датасет: `ai-forever/rubq-reranking`

## 💻 Программный интерфейс

### Базовое использование

```python
from reranker.cross_encoder_service import CrossEncoderReranker

# Инициализация
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    device="cuda"
)

# Ранжирование
query = "What is machine learning?"
candidates = [
    {"text": "Machine learning is a subset of AI...", "metadata": {...}},
    {"text": "The weather today is sunny...", "metadata": {...}},
    # ... more candidates
]

result = reranker.rerank_candidates(query, candidates, top_k=3)
```

### Convenience функции

```python
from reranker.cross_encoder_service import rerank_candidates

# Простое использование
reranked = rerank_candidates(query, candidates, top_k=3)
```

### Результат

```python
# result.reranked_candidates - отсортированные кандидаты
# result.scores - оценки релевантности
# result.query - исходный запрос

for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
    print(f"{i+1}. Score: {score:.4f}")
    print(f"   Text: {candidate['text'][:100]}...")
```

## ⚙️ Конфигурация

### Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Модель для ранжирования |
| `CROSS_ENCODER_DEVICE` | `auto` | Устройство: `cuda`, `cpu`, `auto` |
| `CROSS_ENCODER_CACHE_DIR` | `None` | Директория для кэша моделей |

### Параметры модели

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `max_length` | `512` | Максимальная длина последовательности |
| `batch_size` | `16` | Размер батча для обработки |
| `device` | `auto` | Устройство для вычислений |

## 🔧 Интеграция в RAG пайплайн

### В Gateway API

```python
# services/online-rag/api/gateway.py

@app.post("/query")
def query_endpoint(req: QueryRequest):
    # 1. Получаем больше кандидатов для ранжирования
    initial_k = req.k * 3 if req.use_reranking else req.k
    docs = vs.similarity_search(req.query, k=initial_k)
    
    # 2. Применяем ранжирование
    if req.use_reranking and docs:
        candidates = [{"text": d.page_content, "metadata": d.metadata} for d in docs]
        reranked = rerank_candidates(req.query, candidates, top_k=req.k)
        # Обновляем порядок документов
    
    # 3. Генерируем ответ
    context = "\n\n".join(d.page_content for d in docs)
    answer = generate_response(context, req.query)
```

### Стратегия поиска

```python
# Рекомендуемая стратегия:
# 1. Получить k*3 кандидатов из векторного поиска
# 2. Применить Cross-Encoder ранжирование
# 3. Выбрать топ-k результатов
# 4. Передать в LLM для генерации ответа

initial_k = 12  # Получить больше кандидатов
final_k = 4     # Выбрать лучшие после ранжирования
```

## 📈 Производительность

### Ожидаемые метрики

| Модель | Размер | Скорость | VRAM | Качество |
|--------|--------|----------|------|----------|
| `ms-marco-MiniLM-L-12-v2` | 50MB | ~100ms | 1-2GB | ⭐⭐⭐⭐⭐ |
| `ms-marco-MiniLM-L-6-v2` | 25MB | ~50ms | 500MB-1GB | ⭐⭐⭐⭐ |

### Оптимизация

```python
# Для production нагрузки
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Быстрее
    batch_size=32,  # Больший батч
    device="cuda"
)

# Для максимального качества
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",  # Лучше качество
    batch_size=16,
    device="cuda"
)
```

## 🧪 Тестирование

### Запуск тестов

```bash
# Базовое тестирование
python test_cross_encoder.py

# Тестирование интеграции
python -m pytest tests/test_reranking.py

# Нагрузочное тестирование
python test_cross_encoder.py --performance
```

### Пример теста

```python
def test_reranking():
    query = "What is machine learning?"
    candidates = [
        {"text": "Machine learning is a subset of AI...", "metadata": {}},
        {"text": "The weather is sunny today...", "metadata": {}},
    ]
    
    result = rerank_candidates(query, candidates, top_k=1)
    
    # Проверяем, что ML-документ получил более высокий рейтинг
    assert result[0]['text'].startswith("Machine learning")
```

## 🐛 Troubleshooting

### Проблема: "CUDA out of memory"

**Решение:**
```python
# Используйте меньшую модель или CPU
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"  # Или уменьшите batch_size
)
```

### Проблема: "Model not found"

**Решение:**
```bash
# Проверьте подключение к интернету
# Модель загрузится автоматически при первом использовании
```

### Проблема: Медленная работа

**Решение:**
```python
# Оптимизируйте параметры
reranker = CrossEncoderReranker(
    batch_size=32,  # Увеличьте батч
    device="cuda"   # Используйте GPU
)
```

## 📚 Дополнительные ресурсы

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [Cross-Encoder Models on Hugging Face](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads)

## 🎯 Следующие шаги

1. **Тестирование**: Запустите тесты на ваших данных
2. **Настройка**: Подберите оптимальные параметры
3. **Мониторинг**: Отслеживайте метрики качества
4. **Fine-tuning**: Рассмотрите дообучение на русских данных

---

**Готово к production!** 🚀
