# Docling Models - Краткая справка

## Какие модели используются?

### ✅ Установлено и работает локально:

1. **DocLayNet Layout Detection** (~200MB)
   - Определяет структуру: заголовки, параграфы, списки, таблицы, формулы, код
   
2. **TableFormer** (~150MB)
   - Распознает структуру таблиц с точностью ~95%
   - Режим: ACCURATE (максимальное качество)
   
3. **Tesseract OCR** (~50-100MB)
   - Распознавание текста на сканах
   - Языки: русский + английский
   
4. **Formula Detection** (~50MB)
   - Автоматически определяет математические формулы
   
5. **Code Detection** (~30MB)
   - Автоматически находит блоки кода

**Итого: ~600MB** (все локально!)

---

## Как загрузить все модели?

### Вариант 1: Базовый (рекомендуется)

```bash
make download-models
```

Загрузит все модели кроме VLM (~600MB)

### Вариант 2: С VLM (для GPU серверов)

```bash
make download-models-with-vlm
```

Загрузит ВСЕ модели включая VLM (~5GB)  
**Требует GPU для инференса!**

---

## Где хранятся?

```bash
# В Docker
docker volume inspect ai-chatbot_docling_models

# Локально
~/.cache/huggingface/hub/
├── models--ds4sd--docling-layout-model/
├── models--ds4sd--docling-tableformer/
└── models--ds4sd--docling-formula-detection/
```

---

## Проверка

```bash
# Посмотреть загруженные модели
docker-compose exec ingestion ls -lh /app/.cache/huggingface/hub/

# Размер
docker-compose exec ingestion du -sh /app/.cache/huggingface/hub/

# Тест офлайн-режима
python test_offline_mode.py
```

---

## Производительность

| Модель | Время обработки | Точность |
|--------|----------------|----------|
| Layout Detection | 0.1-0.3 сек/страница | ~95% |
| TableFormer (ACCURATE) | 0.5-1.5 сек/таблица | ~95% |
| Tesseract OCR | 1-3 сек/страница | ~90% |
| Formula Detection | 0.1-0.2 сек/формула | ~92% |

**Общая скорость:** 2-5 секунд на страницу PDF (зависит от сложности)

---

## Offline работа

После `make download-models`:

✅ Все модели закешированы  
✅ Интернет НЕ нужен  
✅ Никаких внешних API  
✅ 100% приватность  

---

## Требования

- **CPU:** 4+ cores
- **RAM:** 4-8 GB
- **Диск:** 1 GB (для моделей)
- **GPU:** Не требуется (опционально для VLM)

---

Полная документация: `services/ingestion/DOCLING_MODELS.md`

