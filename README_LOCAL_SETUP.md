# Локальная установка Docling - Быстрый старт

## Что установлено

✅ **DoclingParser** - унифицированный парсер документов  
✅ **Локальные модели** - все работает офлайн  
✅ **Конфигурация** - переменные окружения  
✅ **Docker** - готовая конфигурация

---

## Быстрая установка

### 1. Установка зависимостей

```bash
# Установка Python пакетов
cd services/ingestion
pip install -r requirements.txt

# Установка Tesseract (для OCR)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng

# macOS:
brew install tesseract tesseract-lang
```

### 2. Загрузка моделей (один раз с интернетом)

```bash
# Из корня проекта
python services/ingestion/download_models.py
```

Это загрузит ~500MB моделей в `~/.cache/huggingface/hub/`

### 3. Тест офлайн-режима

```bash
python test_offline_mode.py
```

Ожидаемый вывод:
```
✓ Found 3 models in cache
✓ Tesseract OCR 5.x.x
✓ DoclingParser initialized in offline mode
✓ Successfully parsed 45 blocks
```

### 4. Запуск API

```bash
cd services/ingestion
uvicorn api:app --host 0.0.0.0 --port 8001
```

Тест:
```bash
curl -F "file=@tests/pdf/Knoleges.pdf" http://localhost:8001/ingest
```

---

## Docker развертывание

### Вариант 1: С предзагрузкой моделей

```bash
# Создать директорию для кеша
mkdir -p data/models/docling

# Собрать образ
docker-compose -f docker-compose.ingestion.yml build

# Загрузить модели (один раз)
docker-compose -f docker-compose.ingestion.yml run --rm ingestion python download_models.py

# Запустить (работает офлайн)
docker-compose -f docker-compose.ingestion.yml up -d
```

### Вариант 2: Using Makefile

```bash
# Полная установка
make -f Makefile.docling setup-docker

# Или пошагово:
make -f Makefile.docling docker-build
make -f Makefile.docling docker-download-models
make -f Makefile.docling docker-run

# Тест
make -f Makefile.docling test-api
```

---

## Конфигурация

Файл `services/ingestion/.env`:

```bash
# Локальная работа (по умолчанию)
DOCLING_OFFLINE_MODE=true
DOCLING_OCR_ENABLED=true
DOCLING_EXTRACT_TABLES=true
DOCLING_CACHE_DIR=/data/models/docling

# Tesseract
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
```

---

## Использование в коде

```python
from parsers.docling_parser import DoclingParser

# Создание парсера (все локально)
parser = DoclingParser(
    extract_tables=True,
    extract_images=True,
    ocr_enabled=True,
    offline_mode=True,  # Принудительно офлайн
    cache_dir="/custom/cache",  # Опционально
)

# Парсинг документа
blocks = list(parser.parse("document.pdf"))

for block in blocks:
    print(f"[{block.meta['label']}] {block.text[:50]}...")
```

---

## Проверка локальной работы

```bash
# Быстрый тест
python test_offline_mode.py

# Полные тесты
pytest tests/test_docling_integration.py -v

# Минимальный тест
python tests/test_docling_minimal.py
```

---

## Устранение проблем

### Tesseract не найден

```bash
# Проверка
tesseract --version

# Установка (Ubuntu)
sudo apt-get install tesseract-ocr

# Установка (macOS)
brew install tesseract
```

### Модели не найдены

```bash
# Проверить кеш
ls -la ~/.cache/huggingface/hub/

# Загрузить заново
python services/ingestion/download_models.py
```

### Медленная обработка

```python
# Отключить OCR для документов с текстом
parser = DoclingParser(ocr_enabled=False)

# Отключить таблицы
parser = DoclingParser(extract_tables=False)
```

---

## Air-gapped развертывание

Для серверов без интернета:

```bash
# На машине с интернетом
make -f Makefile.docling airgap-export

# Перенести airgap-package/ на целевую машину

# На целевой машине
make -f Makefile.docling airgap-install
```

---

## Команды Makefile

```bash
make -f Makefile.docling help                 # Помощь
make -f Makefile.docling install-local        # Установка локально
make -f Makefile.docling download-models      # Загрузка моделей
make -f Makefile.docling test-offline         # Тест офлайн
make -f Makefile.docling docker-build         # Сборка Docker
make -f Makefile.docling docker-run           # Запуск Docker
make -f Makefile.docling setup-local          # Полная локальная установка
make -f Makefile.docling setup-docker         # Полная Docker установка
```

---

## Требования

- **CPU**: 2+ cores (4+ рекомендуется)
- **RAM**: 4GB+ (8GB рекомендуется)
- **Диск**: 2GB (500MB модели + 1.5GB зависимости)
- **Tesseract**: Для OCR

---

## Производительность

| Документ | Размер | Время | Настройки |
|----------|--------|-------|-----------|
| PDF с текстом | 10 стр | 5-10 сек | OCR выкл |
| Скан PDF | 10 стр | 15-30 сек | OCR вкл |
| DOCX | 50 стр | 3-7 сек | - |
| Изображение | 1 стр | 2-5 сек | OCR вкл |

---

**Готово!** Все работает локально без внешних API. 🚀

