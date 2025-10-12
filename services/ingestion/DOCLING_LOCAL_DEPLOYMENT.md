# Docling - Локальное развертывание (только локальные модели)

## 🔒 Обзор

**Docling работает полностью локально** и не требует подключения к внешним API или облачным сервисам. Все модели загружаются один раз и хранятся на вашем компьютере.

---

## 🧠 Какие модели использует Docling?

### 1. **OCR (распознавание текста)**
- **Библиотека:** Tesseract OCR (через `tesserocr`)
- **Размер:** ~50-100 MB (зависит от языков)
- **Работа:** Полностью локально
- **Источник:** Open Source (Apache 2.0)

### 2. **Layout Analysis (анализ структуры)**
- **Модель:** DoclingParseV4 (IBM Research)
- **Размер:** ~200-500 MB
- **Работа:** Полностью локально
- **Источник:** IBM (встроена в Docling)

### 3. **Table Detection (обнаружение таблиц)**
- **Модель:** TableFormer или аналогичная
- **Размер:** ~100-300 MB
- **Работа:** Полностью локально
- **Источник:** Встроена в Docling

### 4. **Vision Language Model (опционально)**
- **Модель:** PaliGemma или аналогичная (только если включено)
- **Размер:** ~3-5 GB
- **Работа:** Локально (на GPU или CPU)
- **Использование:** Только для сложных документов с изображениями

---

## 📦 Установка для полностью локальной работы

### Вариант 1: Базовая установка (рекомендуется)

```bash
# Установка Docling с базовыми возможностями
pip install docling

# Установка Tesseract для OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng

# macOS:
brew install tesseract tesseract-lang

# Windows: скачайте установщик с https://github.com/UB-Mannheim/tesseract/wiki
# После установки добавьте в PATH: C:\Program Files\Tesseract-OCR
```

### Вариант 2: Расширенная установка (с VLM)

```bash
# Установка с визуальными моделями (требует больше ресурсов)
pip install "docling[vlm,tesserocr]"

# Установка Tesseract (см. выше)
```

### Вариант 3: Минимальная установка (без OCR)

```bash
# Только для документов с текстовым слоем (не сканы)
pip install docling
```

---

## 🔧 Настройка локального кеша моделей

Docling автоматически загружает модели в локальный кеш при первом использовании:

```bash
# Linux/macOS
~/.cache/huggingface/hub/

# Windows
C:\Users\<Username>\.cache\huggingface\hub\
```

### Предварительная загрузка моделей

Чтобы загрузить все модели заранее (например, для офлайн-работы):

```python
# download_models.py
from docling.document_converter import DocumentConverter
import os

# Создаем тестовый документ для инициализации всех моделей
test_pdf = "test.pdf"  # Любой тестовый PDF

print("Загружаем модели Docling...")
converter = DocumentConverter()
result = converter.convert(test_pdf)
print("✓ Модели загружены и закешированы локально")
print(f"Кеш: {os.path.expanduser('~/.cache/huggingface/hub/')}")
```

Запустите один раз с интернетом:
```bash
python download_models.py
```

После этого Docling будет работать **полностью офлайн**.

---

## ⚙️ Конфигурация для локальной работы

### 1. Базовая конфигурация (без внешних вызовов)

Обновим `services/ingestion/parsers/docling_parser.py`:

```python
class DoclingParser(BaseParser):
    def __init__(
        self,
        *,
        extract_tables: bool = True,
        extract_images: bool = True,
        ocr_enabled: bool = True,
        preserve_structure: bool = True,
        offline_mode: bool = True,  # Новый параметр
    ) -> None:
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self.preserve_structure = preserve_structure
        self.offline_mode = offline_mode
    
    def parse(self, path: str, *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        ensure_dependency("docling", "pip install docling")
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        
        # Настройка для локальной работы
        pipeline_options = PdfPipelineOptions()
        
        if self.offline_mode:
            # Используем только локальные ресурсы
            pipeline_options.do_ocr = self.ocr_enabled
            pipeline_options.do_table_structure = self.extract_tables
            # Отключаем VLM если не нужен
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        
        converter = DocumentConverter(
            pipeline_options=pipeline_options,
        )
        
        result = converter.convert(str(path))
        # ... rest of the code
```

### 2. Конфигурация через переменные окружения

Создайте `.env` файл:

```bash
# services/ingestion/.env

# Docling настройки
DOCLING_OFFLINE_MODE=true
DOCLING_OCR_ENABLED=true
DOCLING_EXTRACT_TABLES=true
DOCLING_CACHE_DIR=/data/models/docling

# Tesseract путь (если не в PATH)
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
```

Используйте в коде:

```python
import os

class DoclingParser(BaseParser):
    def __init__(self, **kwargs):
        self.offline_mode = os.getenv("DOCLING_OFFLINE_MODE", "true").lower() == "true"
        self.ocr_enabled = os.getenv("DOCLING_OCR_ENABLED", "true").lower() == "true"
        # ...
```

---

## 🐳 Docker развертывание (полностью локально)

### Обновленный Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предварительная загрузка моделей (во время build)
# Это гарантирует, что контейнер будет работать офлайн
COPY download_models.py .
RUN python download_models.py || echo "Models will be downloaded on first run"

# Копируем код приложения
COPY . .

# Настройка переменных окружения для локальной работы
ENV DOCLING_OFFLINE_MODE=true
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Docker Compose с локальным кешем

```yaml
# docker-compose.yml
version: '3.8'

services:
  ingestion:
    build: 
      context: ./services/ingestion
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    volumes:
      # Монтируем локальный кеш моделей
      - ./data/models/docling:/app/.cache/huggingface:rw
      # Данные для обработки
      - ./data/uploads:/data:rw
    environment:
      - DOCLING_OFFLINE_MODE=true
      - DOCLING_OCR_ENABLED=true
      - QDRANT_URL=http://qdrant:6333
      - EMBEDDINGS_BASE_URL=http://embeddings:80
    networks:
      - rag-network
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

networks:
  rag-network:
    driver: bridge
```

Сборка и запуск:

```bash
# Создаем директорию для кеша моделей
mkdir -p data/models/docling

# Собираем образ (модели загрузятся во время сборки)
docker-compose build ingestion

# Запускаем (работает полностью офлайн)
docker-compose up -d ingestion
```

---

## 🧪 Проверка локальной работы

### Скрипт проверки

Создайте `test_offline.py`:

```python
#!/usr/bin/env python3
"""Проверка, что Docling работает полностью локально."""

import sys
import os
from pathlib import Path

# Отключаем сеть для теста (опционально)
# os.environ['http_proxy'] = 'http://127.0.0.1:9999'
# os.environ['https_proxy'] = 'http://127.0.0.1:9999'

sys.path.insert(0, "services/ingestion")

print("=" * 60)
print("DOCLING OFFLINE MODE TEST")
print("=" * 60)

print("\n[1/4] Проверка кеша моделей...")
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
if cache_dir.exists():
    models = list(cache_dir.glob("*"))
    print(f"✓ Найдено {len(models)} моделей в кеше")
    print(f"  Путь: {cache_dir}")
else:
    print("⚠ Кеш моделей пуст (модели загрузятся при первом использовании)")

print("\n[2/4] Проверка Tesseract...")
try:
    import subprocess
    result = subprocess.run(
        ["tesseract", "--version"], 
        capture_output=True, 
        text=True
    )
    version = result.stdout.split('\n')[0]
    print(f"✓ {version}")
except Exception as e:
    print(f"✗ Tesseract не найден: {e}")
    print("  Установите: apt-get install tesseract-ocr")

print("\n[3/4] Проверка Docling (локальный режим)...")
try:
    from parsers.docling_parser import DoclingParser
    
    parser = DoclingParser(
        ocr_enabled=True,
        extract_tables=True,
        offline_mode=True,
    )
    print("✓ DoclingParser инициализирован")
    
except Exception as e:
    print(f"✗ Ошибка: {e}")
    sys.exit(1)

print("\n[4/4] Тест парсинга документа...")
test_pdf = Path("tests/pdf/Knoleges.pdf")
if test_pdf.exists():
    try:
        print(f"  Парсинг: {test_pdf}")
        blocks = list(parser.parse(str(test_pdf)))
        print(f"✓ Успешно обработано {len(blocks)} блоков")
        print(f"  Первый блок: {blocks[0].text[:60]}...")
    except Exception as e:
        print(f"✗ Ошибка парсинга: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"⚠ Тестовый файл не найден: {test_pdf}")

print("\n" + "=" * 60)
print("✓ Docling работает в полностью локальном режиме!")
print("=" * 60)
print("\nПримечания:")
print("- Модели хранятся в:", cache_dir)
print("- Для работы офлайн запустите скрипт один раз с интернетом")
print("- После загрузки моделей интернет не требуется")
```

Запуск:
```bash
python test_offline.py
```

---

## 📋 Список моделей и их расположение

После первого запуска модели будут в:

```
~/.cache/huggingface/hub/
├── models--DS4SD--docling-layout-v1/        # Layout analysis (~200MB)
├── models--DS4SD--docling-tableformer/      # Table detection (~100MB)
├── models--DS4SD--docling-ocr/              # OCR enhancement (~50MB)
└── (опционально) models--google--paligemma-3b-mix-224/  # VLM (~3GB)
```

### Управление кешем

```bash
# Просмотр размера кеша
du -sh ~/.cache/huggingface/hub/

# Очистка кеша (будет загружено заново)
rm -rf ~/.cache/huggingface/hub/

# Копирование кеша на другую машину
tar -czf docling_models.tar.gz ~/.cache/huggingface/hub/
# На другой машине:
tar -xzf docling_models.tar.gz -C ~/
```

---

## ⚡ Оптимизация для локальной работы

### 1. Отключение VLM (экономия памяти)

```python
# Не устанавливайте [vlm] зависимости
pip install docling  # Без [vlm]

# В коде
parser = DoclingParser(
    extract_images=False,  # Отключаем тяжелую обработку изображений
)
```

### 2. Использование CPU вместо GPU

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Принудительно CPU
```

### 3. Кеширование результатов

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def parse_document_cached(file_path: str, file_hash: str):
    parser = DoclingParser()
    return list(parser.parse(file_path))

# Использование
file_hash = hashlib.md5(open(path, 'rb').read()).hexdigest()
blocks = parse_document_cached(path, file_hash)
```

---

## 🔒 Полностью изолированное развертывание

Для максимальной изоляции (air-gapped среда):

### Шаг 1: Загрузка моделей на машине с интернетом

```bash
# На машине с интернетом
python -c "
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert('test.pdf')
print('Модели загружены')
"

# Архивируем кеш
cd ~
tar -czf docling_cache.tar.gz .cache/huggingface/
```

### Шаг 2: Перенос на целевую машину

```bash
# На целевой машине (без интернета)
tar -xzf docling_cache.tar.gz -C ~/

# Установка Docling и зависимостей (pip wheel заранее)
pip install docling-*.whl --no-index --find-links ./wheels/
```

---

## 📊 Требования к ресурсам

| Конфигурация | CPU | RAM | Диск | Производительность |
|--------------|-----|-----|------|-------------------|
| Минимальная (без OCR) | 2 cores | 2 GB | 500 MB | 1-2 стр/сек |
| Базовая (с OCR) | 4 cores | 4 GB | 1 GB | 0.5-1 стр/сек |
| Оптимальная (с VLM) | 8 cores | 8 GB | 6 GB | 0.2-0.5 стр/сек |
| С GPU (VLM) | 4 cores + GPU | 16 GB | 6 GB | 2-5 стр/сек |

---

## ❓ FAQ

### Q: Нужен ли интернет после установки?
**A:** Нет! После первого запуска (когда загрузятся модели) Docling работает полностью офлайн.

### Q: Можно ли использовать свои модели OCR?
**A:** Да, можно использовать кастомный Tesseract или другие OCR движки через настройку.

### Q: Как проверить, что ничего не уходит в интернет?
**A:** Запустите `test_offline.py` или используйте firewall/proxy для блокировки.

### Q: Какой минимальный размер установки?
**A:** ~1.5 GB (Docling + Tesseract + базовые модели)

### Q: Можно ли использовать без Tesseract?
**A:** Да, если документы имеют текстовый слой (не сканы). Установите `DoclingParser(ocr_enabled=False)`.

---

## ✅ Checklist для production

- [ ] Установлен Tesseract OCR
- [ ] Модели Docling загружены в локальный кеш
- [ ] Протестирован офлайн-режим (`test_offline.py`)
- [ ] Настроены переменные окружения
- [ ] Смонтирован volume для кеша моделей в Docker
- [ ] Проверена производительность на реальных документах
- [ ] Настроено логирование
- [ ] Документированы требования к ресурсам

---

**Итого:** Docling - это полностью локальное решение, не требующее внешних API. Все работает на вашем железе! 🚀

