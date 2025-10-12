# Docling Models - Полный список

## Все модели, которые использует Docling (локально)

### 1. **Layout Analysis (Анализ структуры документа)**
- **Модель:** DocLayNet Layout Detection
- **Размер:** ~200-300 MB
- **Задача:** Определяет типы блоков: title, section_header, paragraph, list, table, figure, caption, formula, code
- **Hugging Face:** `ds4sd/docling-layout-model`
- **Работает:** Полностью локально

### 2. **Table Structure Recognition (TableFormer)**
- **Модель:** TableFormer
- **Размер:** ~100-150 MB
- **Задача:** Распознает структуру таблиц (строки, колонки, объединенные ячейки)
- **Hugging Face:** `ds4sd/docling-tableformer`
- **Режимы:**
  - `FAST` - быстрый (точность ~85%)
  - `ACCURATE` - точный (точность ~95%, медленнее)

### 3. **OCR (Оптическое распознавание текста)**
**Вариант A: Tesseract (по умолчанию)**
- **Размер:** ~50-100 MB (+ языковые пакеты)
- **Задача:** Распознавание текста на сканах
- **Установка:** Через apt-get
- **Языки:** eng, rus, deu, fra и др.

**Вариант B: EasyOCR (опционально)**
- **Размер:** ~50-80 MB
- **Задача:** Альтернативный OCR движок
- **Hugging Face:** Автоматически загружается

### 4. **Formula Detection (Распознавание формул)**
- **Модель:** Formula detector
- **Размер:** ~50 MB
- **Задача:** Находит и извлекает математические формулы
- **Формат:** LaTeX
- **Hugging Face:** `ds4sd/docling-formula-detection`

### 5. **Code Block Detection (Распознавание кода)**
- **Модель:** Code detector
- **Размер:** ~30 MB
- **Задача:** Находит блоки кода в документах
- **Поддержка:** Python, Java, C++, JavaScript и др.

### 6. **Vision Language Model (VLM) - ОПЦИОНАЛЬНО**
**Если включено `vlm` в зависимостях:**
- **Модель:** PaliGemma или Qwen-VL
- **Размер:** ~3-5 GB
- **Задача:**
  - Описание изображений в документе
  - Ответы на вопросы по изображениям
  - Распознавание сложных диаграмм
- **Требования:** GPU рекомендуется
- **Hugging Face:** `google/paligemma-3b-mix-224` или `Qwen/Qwen-VL`

### 7. **Page Image Rendering**
- **Библиотека:** PyMuPDF (встроена)
- **Задача:** Рендеринг страниц PDF в изображения
- **Размер:** ~10 MB

---

## Итоговые требования по памяти

| Конфигурация | Модели | Размер на диске | RAM |
|--------------|--------|-----------------|-----|
| **Минимальная** (без OCR) | Layout + TableFormer | ~400 MB | 2 GB |
| **Базовая** (с Tesseract) | Layout + TableFormer + Tesseract | ~600 MB | 4 GB |
| **Полная** (с формулами) | Layout + TableFormer + OCR + Formula + Code | ~800 MB | 4 GB |
| **Максимальная** (с VLM) | Все + VLM | ~5 GB | 8-16 GB |

---

## Где хранятся модели?

```bash
~/.cache/huggingface/hub/
├── models--ds4sd--docling-layout-model/
├── models--ds4sd--docling-tableformer/
├── models--ds4sd--docling-formula-detection/
└── models--google--paligemma-3b-mix-224/  # если VLM включен
```

В Docker:
```bash
/app/.cache/huggingface/hub/  # монтируется как volume
```

---

## Автоматическая загрузка

Все модели загружаются **автоматически** при первом использовании:

```python
from docling.document_converter import DocumentConverter

# При первом запуске загрузит все нужные модели (новый API)
from docling.datamodel.pipeline_options import PdfPipelineOptions

options = PdfPipelineOptions()
converter = DocumentConverter(format_options={"pdf": options})
result = converter.convert("document.pdf")
```

---

## Управление моделями

### Предзагрузка (рекомендуется)

```bash
# Загрузить все модели заранее
python services/ingestion/download_models.py
```

### Проверка кеша

```bash
# Посмотреть загруженные модели
ls -lh ~/.cache/huggingface/hub/

# Размер кеша
du -sh ~/.cache/huggingface/hub/
```

### Очистка

```bash
# Удалить все модели (будут загружены заново)
rm -rf ~/.cache/huggingface/hub/models--ds4sd*
```

---

## Конфигурация в коде

### Базовая (Layout + Tables)

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

options = PdfPipelineOptions()
options.do_ocr = False  # Без OCR (только текстовый слой)
options.do_table_structure = True
options.table_structure_options.mode = TableFormerMode.FAST

converter = DocumentConverter(format_options={"pdf": options})
```

### С OCR

```python
options = PdfPipelineOptions()
options.do_ocr = True  # Включить Tesseract OCR
options.ocr_engine = "tesseract"  # или "easyocr"
```

### С формулами и кодом

```python
options = PdfPipelineOptions()
options.do_ocr = True
options.do_table_structure = True
options.generate_page_images = True
options.generate_picture_images = True
```

### С VLM (требует GPU)

```python
# Установка
pip install "docling[vlm]"

# Использование
from docling.pipeline.vlm_pipeline import VlmPipeline

options = PdfPipelineOptions()
options.do_ocr = True
options.do_table_structure = True
options.use_vlm = True  # Включить Vision Language Model

converter = DocumentConverter(format_options={"pdf": options})
```

---

## Производительность

### Layout Detection
- **CPU:** 0.1-0.3 сек/страница
- **Точность:** ~95%

### TableFormer
- **FAST режим:** 0.2-0.5 сек/таблица
- **ACCURATE режим:** 0.5-1.5 сек/таблица
- **Точность FAST:** ~85%
- **Точность ACCURATE:** ~95%

### OCR (Tesseract)
- **CPU:** 1-3 сек/страница
- **Точность:** ~90-95% (зависит от качества скана)

### VLM
- **CPU:** 5-10 сек/изображение
- **GPU:** 0.5-2 сек/изображение
- **Точность:** ~90%

---

## Рекомендации

### Для production (без GPU)
```python
options = PdfPipelineOptions()
options.do_ocr = True
options.ocr_engine = "tesseract"
options.do_table_structure = True
options.table_structure_options.mode = TableFormerMode.ACCURATE
```

**Модели:** Layout + TableFormer + Tesseract  
**Размер:** ~600 MB  
**RAM:** 4 GB  

### Для максимального качества (с GPU)
```python
options = PdfPipelineOptions()
options.do_ocr = True
options.ocr_engine = "easyocr"
options.do_table_structure = True
options.table_structure_options.mode = TableFormerMode.ACCURATE
options.use_vlm = True
```

**Модели:** Все + VLM  
**Размер:** ~5 GB  
**RAM:** 8-16 GB  
**GPU:** 4+ GB VRAM  

---

## Offline работа

После загрузки моделей через `download_models.py`:

✅ Все модели закешированы локально  
✅ Интернет НЕ требуется  
✅ Никаких внешних API вызовов  
✅ Полная приватность данных  

```bash
# Отключите интернет и проверьте
sudo ifconfig eth0 down
python test_offline_mode.py  # Должно работать!
```

