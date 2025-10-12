# 🚀 Docling Integration - Быстрый старт

## Что сделано

Ваш ingestion сервис теперь использует **Docling** - современную библиотеку от IBM Research для парсинга документов любых форматов.

## ⚡ Быстрая проверка (без Docker)

### 1. Установите зависимости

```bash
cd services/ingestion
pip install -r requirements.txt
```

Это установит Docling и все необходимое (~2GB, может занять несколько минут).

### 2. Запустите быстрый тест

```bash
# Из корня репозитория
python test_docling_quick.py
```

**Ожидаемый вывод:**
```
======================================================================
DOCLING INTEGRATION - QUICK TEST
======================================================================

[1/5] Checking Docling installation...
✓ Docling 2.x.x is installed

[2/5] Checking DoclingParser...
✓ DoclingParser imported successfully

[3/5] Testing document parsing...
✓ Parsed 45 blocks from test PDF
  First block type: title
  First block text: Introduction to Machine Learning...

[4/5] Testing chunker integration...
✓ Chunker created 3 chunks from first block
  First chunk tokens: 187
  First chunk text: Introduction to Machine Learning...

[5/5] Checking metadata quality...
✓ Found 8 unique metadata fields:
  block_index, doc_id, label, level, page, path, type, headings
  Important fields present: 5/5

======================================================================
SUMMARY
======================================================================
✓ Docling is installed and working
✓ DoclingParser is functional
✓ Successfully parsed test document (45 blocks)
✓ Chunker integration working
✓ Rich metadata preserved
```

## 📝 Минимальный пример использования

Создайте файл `test_parse.py`:

```python
from pathlib import Path
import sys
sys.path.insert(0, "services/ingestion")

from parsers.docling_parser import DoclingParser
from processors.chunker import chunk_text

# Парсинг документа
parser = DoclingParser()
blocks = list(parser.parse("tests/pdf/Knoleges.pdf"))

print(f"Извлечено блоков: {len(blocks)}")

# Разбиение на чанки
total_chunks = 0
for block in blocks[:5]:  # Первые 5 блоков
    chunks = chunk_text(block.text, doc_metadata=block.meta)
    total_chunks += len(chunks)
    print(f"Блок '{block.meta.get('label')}' -> {len(chunks)} чанков")

print(f"\nВсего чанков: {total_chunks}")
```

Запустите:
```bash
python test_parse.py
```

## 🌐 Запуск API локально

```bash
cd services/ingestion
uvicorn api:app --host 0.0.0.0 --port 8001
```

Тестирование в другом терминале:
```bash
curl -F "file=@tests/pdf/Knoleges.pdf" http://localhost:8001/ingest
```

**Ожидаемый ответ:**
```json
{
  "status": "ok",
  "doc_id": "a1b2c3d4e5f6g7h8",
  "chunks_total": 123,
  "chunks_unique": 98
}
```

## 🎯 Что изменилось

### Старый подход (было)
```python
from parsers.pdf_parser import PdfParser
from parsers.docx_parser import DocxParser
from processors.normalizer import normalize_text

# Разные парсеры для разных форматов
if file.endswith('.pdf'):
    parser = PdfParser()
elif file.endswith('.docx'):
    parser = DocxParser()
# ...

blocks = list(parser.parse(file))
for block in blocks:
    text = normalize_text(block.text)  # Ручная нормализация
```

### Новый подход (стало)
```python
from parsers.docling_parser import DoclingParser

# Один парсер для всех форматов
parser = DoclingParser()
blocks = list(parser.parse(file))  # PDF, DOCX, PPTX, изображения...

for block in blocks:
    text = block.text  # Уже нормализован!
```

## 📋 Поддерживаемые форматы

| Формат | Расширения | Особенности |
|--------|-----------|-------------|
| PDF | `.pdf` | OCR, таблицы, структура |
| Word | `.docx`, `.doc` | Таблицы, структура |
| PowerPoint | `.pptx`, `.ppt` | Слайды, текст |
| Excel | `.xlsx`, `.xls` | Листы, таблицы |
| Изображения | `.png`, `.jpg`, `.tiff` | OCR |
| HTML | `.html`, `.htm` | Структура |
| Markdown | `.md` | Структура |

## 🧪 Полное тестирование

```bash
# Минимальная проверка
python tests/test_docling_minimal.py

# Полные тесты (требует pytest)
pytest tests/test_docling_integration.py -v -s
```

## 📚 Документация

1. **README_DOCLING.md** → Подробное руководство с примерами
2. **DOCLING_MIGRATION.md** → Руководство по миграции
3. **DOCLING_INTEGRATION_SUMMARY.md** → Полная сводка изменений

## 💡 Примеры использования

### Пример 1: Простой парсинг
```python
from parsers.docling_parser import DoclingParser

parser = DoclingParser()
blocks = list(parser.parse("document.pdf"))

for block in blocks:
    print(f"[{block.meta['label']}] {block.text[:80]}...")
```

### Пример 2: С метаданными
```python
parser = DoclingParser(preserve_structure=True)
blocks = list(parser.parse("book.pdf"))

for block in blocks:
    page = block.meta.get('page', '?')
    headings = block.meta.get('headings', [])
    context = " > ".join(headings) if headings else "Root"
    print(f"[Page {page}] {context}: {block.text[:50]}...")
```

### Пример 3: Только таблицы
```python
parser = DoclingParser(extract_tables=True)
blocks = list(parser.parse("report.pdf"))

tables = [b for b in blocks if b.meta.get('label') == 'table']
print(f"Найдено таблиц: {len(tables)}")

for table in tables:
    print(f"\nТаблица на странице {table.meta.get('page')}:")
    print(table.text)
```

## ⚠️ Частые вопросы

### Q: Первый запуск очень долгий?
**A:** Docling загружает ML-модели при первом использовании (1-3 минуты). Последующие запуски быстрые.

### Q: Нужен ли интернет?
**A:** Только при первом запуске для загрузки моделей. После - работает офлайн.

### Q: Где старые парсеры?
**A:** Они не удалены, помечены как "legacy" в requirements.txt. Можно использовать, но не рекомендуется.

### Q: Как ускорить обработку?
**A:** Отключите OCR для документов с текстовым слоем: `DoclingParser(ocr_enabled=False)`

## 🐛 Проблемы?

1. Проверьте установку: `pip show docling`
2. Запустите тест: `python test_docling_quick.py`
3. Посмотрите логи: `python tests/test_docling_minimal.py`
4. Читайте DOCLING_MIGRATION.md для деталей

## 🎉 Готово!

Ваш RAG пайплайн теперь использует современный unified parser от IBM Research!

**Следующие шаги:**
1. Протестируйте на своих документах
2. Настройте под ваши нужды (см. README_DOCLING.md)
3. Запустите в Docker (обновите Dockerfile если нужно)

---

**Документация:** [services/ingestion/README_DOCLING.md](services/ingestion/README_DOCLING.md)  
**Миграция:** [services/ingestion/DOCLING_MIGRATION.md](services/ingestion/DOCLING_MIGRATION.md)  
**Docling GitHub:** https://github.com/DS4SD/docling

