# Архитектуры RAG для документов с поддержкой изображений

## Предпосылки

Этот документ описывает практические паттерны построения корпоративных RAG-систем, которые обрабатывают DOCX, PDF, PPTX, XLSX и кастомные форматы с вложенными изображениями. Основной фокус — офлайн-среда, где pytesseract, модели эмбеддингов и LLM подготавливаются заранее.

## Варианты архитектуры

### Вариант A — текст в первую очередь + OCR-инъекция
- Извлекаем текст и изображения.
- Делает OCR для каждого изображения и прикрепляем результат к близлежащему текстовому блоку (абзац/страница/слайд).
- Далее выполняем чанкинг → эмбеддинги (e5/bge) → dense/гибридный поиск → реранкер → ответ.
- **Плюсы:** минимальная сложность, покрытие большинства кейсов.
- **Минусы:** теряем чисто визуальные сигналы.

### Вариант B — двойной индекс: текст + изображения (CLIP)
- Реализует Вариант A.
- Дополнительно строит отдельный индекс изображений на CLIP-эмбеддингах.
- На поиске объединяет текстовые и визуальные кандидаты перед реранком.
- **Плюсы:** можно искать презентации по визуальным признакам.
- **Минусы:** усложнение архитектуры и рост хранилища.

### Вариант C — учёт лэйаута + Captioning/Structuring LLM
- Вариант A с сохранением bbox и иерархии блоков.
- OCR-текст пропускается через локальный LLM, который нормализует факты (JSON, буллеты).
- Индексируем «сырой» и нормализованный текст как две проекции.
- **Плюсы:** выше точность фактов, аккуратные цитаты.
- **Минусы:** большие требования к ресурсу ингеста.

### Вариант D — OCR по требованию
- На индексации сохраняем ссылки на изображения и подписи.
- OCR запускается на этапе ответа для top-N фрагментов.
- **Плюсы:** экономия времени/места при загрузке.
- **Минусы:** рост латентности при поиске.

### Вариант E — структурные данные (XLSX, PDF-таблицы) → двойной индекс
- Таблицы храним в Markdown, KV-формате и агрегированном JSON.
- Индексируем все представления и на реранке усиливаем Markdown/JSON.
- **Плюсы:** лучше ищется по значениям ячеек и колонок.
- **Минусы:** чуть больше места.

**Рекомендация:** начать с A + частично C: OCR-инъекция + лёгкая LLM-нормализация на ингесте, гибридный поиск (dense + TF-IDF), реранкер. Остальные варианты включаем при необходимости.

## Эталонный пайплайн

```
[DOCX/PDF/PPTX/XLSX]
    → [Парсеры по типам]
    → [Извлечь картинки] → [OCR (+кэш)]
    → [Смерджить OCR в текст + метаданные]
    → [Опциональная LLM-адаптация]
    → [Чанкинг: структурный + RC/семантический]
    → [Эмбеддинги: e5/bge; опционально CLIP]
    → [Векторный стор (FAISS/…); опционально TF-IDF]
    → [Гибридный ретрив + реранкер]
    → [LLM-ответ с цитатами]
```

## Общие сущности и OCR-кэш

```python
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import io
import re

from PIL import Image, ImageFilter, ImageOps
import pytesseract

@dataclass
class RawBlock:
    text: str
    meta: Dict[str, Any]

class OcrCache:
    def __init__(self) -> None:
        self._memo: Dict[str, str] = {}

    @staticmethod
    def _key(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def get(self, data: bytes) -> Optional[str]:
        return self._memo.get(self._key(data))

    def set(self, data: bytes, value: str) -> None:
        self._memo[self._key(data)] = value

OCR_CACHE = OcrCache()

def ocr_image_bytes(img_bytes: bytes, lang: str = "rus+eng") -> str:
    cached = OCR_CACHE.get(img_bytes)
    if cached is not None:
        return cached

    img = Image.open(io.BytesIO(img_bytes))
    if min(img.size) < 800:
        scale = max(1, 800 // max(1, min(img.size)))
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    img = ImageOps.grayscale(img).filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)

    text = pytesseract.image_to_string(img, lang=lang) or ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    OCR_CACHE.set(img_bytes, text)
    return text

def merge_text(base: str, ocr_text: str, caption: Optional[str] = None) -> str:
    parts: List[str] = []
    if base.strip():
        parts.append(base.strip())
    if caption and caption.strip():
        parts.append(f"[Подпись] {caption.strip()}")
    if ocr_text.strip():
        parts.append(f"[OCR] {ocr_text.strip()}")
    return "\n".join(parts)
```

## Парсеры с поддержкой OCR

### DOCX: абзацы + изображения

```python
from docx import Document


def parse_docx_with_ocr(path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
    document = Document(path)
    blocks: List[RawBlock] = []

    for paragraph in document.paragraphs:
        text = (paragraph.text or "").strip()
        if text:
            blocks.append(RawBlock(text, {"type": "docx", "path": path, "doc_id": doc_id}))

    last_block = len(blocks) - 1
    for rel in document.part.rels.values():
        if "image" not in getattr(rel, "target_ref", ""):
            continue
        blob = rel._target.blob
        ocr_text = ocr_image_bytes(blob)
        if not ocr_text:
            continue
        if last_block >= 0:
            block = blocks[last_block]
            block.text = merge_text(block.text, ocr_text)
            block.meta["has_ocr"] = True
            block.meta["image_count"] = block.meta.get("image_count", 0) + 1
        else:
            blocks.append(
                RawBlock(
                    merge_text("", ocr_text),
                    {"type": "docx", "path": path, "doc_id": doc_id, "has_ocr": True, "image_count": 1},
                )
            )

    yield from blocks
```

### PPTX: shapes + Notes + OCR

```python
from pptx import Presentation


def parse_pptx_with_ocr(path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
    presentation = Presentation(path)
    for index, slide in enumerate(presentation.slides, start=1):
        text_parts: List[str] = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text_parts.append(shape.text.strip())

        notes = ""
        if slide.has_notes_slide and slide.notes_slide and slide.notes_slide.notes_text_frame:
            notes = (slide.notes_slide.notes_text_frame.text or "").strip()
        if notes:
            text_parts.append(f"Notes: {notes}")

        ocr_chunks: List[str] = []
        for shape in slide.shapes:
            if hasattr(shape, "image") and shape.image:
                ocr = ocr_image_bytes(shape.image.blob)
                if ocr:
                    ocr_chunks.append(ocr)

        merged = merge_text("\n\n".join(text_parts), "\n\n".join(ocr_chunks)) if ocr_chunks else "\n\n".join(text_parts)
        if merged.strip():
            yield RawBlock(
                merged,
                {
                    "type": "pptx",
                    "path": path,
                    "doc_id": doc_id,
                    "slide": index,
                    "has_ocr": bool(ocr_chunks),
                    "image_count": len(ocr_chunks),
                },
            )
```

### PDF: layout-aware с PyMuPDF

```python
import fitz
import math


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_dist(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def parse_pdf_with_ocr(path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
    document = fitz.open(path)
    for page_number, page in enumerate(document, start=1):
        raw = page.get_text("rawdict")
        blocks: List[RawBlock] = []
        images: List[Tuple[bytes, Tuple[float, float, float, float]]] = []

        for block in raw["blocks"]:
            if block["type"] == 0:
                text = "".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
                if text:
                    blocks.append(
                        RawBlock(
                            text,
                            {
                                "type": "pdf",
                                "path": path,
                                "doc_id": doc_id,
                                "page": page_number,
                                "bbox": tuple(block["bbox"]),
                            },
                        )
                    )
            elif block["type"] == 1 and block.get("image"):
                pix = fitz.Pixmap(document, block["image"])
                try:
                    data = pix.tobytes("png")
                except Exception:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    data = pix.tobytes("png")
                images.append((data, tuple(block["bbox"])))

        for data, bbox in images:
            ocr_text = ocr_image_bytes(data)
            if not ocr_text:
                continue
            if blocks:
                target = min(blocks, key=lambda b: _bbox_dist(b.meta["bbox"], bbox))
                target.text = merge_text(target.text, ocr_text)
                target.meta["has_ocr"] = True
                target.meta["image_count"] = target.meta.get("image_count", 0) + 1
            else:
                blocks.append(
                    RawBlock(
                        merge_text("", ocr_text),
                        {
                            "type": "pdf",
                            "path": path,
                            "doc_id": doc_id,
                            "page": page_number,
                            "has_ocr": True,
                            "image_count": 1,
                            "bbox": bbox,
                        },
                    )
                )

        yield from blocks
```

### XLSX: листы, окна строк и опциональный OCR

```python
import pandas as pd
from openpyxl import load_workbook


def parse_xlsx_with_ocr(path: str, doc_id: Optional[str] = None, rows_per_chunk: int = 50) -> Iterable[RawBlock]:
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if df.empty:
            continue
        summary = f"Лист: {sheet}\nКолонки: {', '.join(map(str, df.columns))}\nСтрок: {len(df)}"
        yield RawBlock(summary, {"type": "xlsx_sheet", "path": path, "doc_id": doc_id, "sheet": sheet})

        for start in range(0, len(df), rows_per_chunk):
            part = df.iloc[start:start + rows_per_chunk]
            table_text = part.to_markdown(index=False)
            kv_view = "\n".join(
                " | ".join(f"{column}: {row[column]}" for column in part.columns)
                for _, row in part.iterrows()
            )
            yield RawBlock(
                f"{table_text}\n\n{kv_view}",
                {
                    "type": "xlsx_rows",
                    "path": path,
                    "doc_id": doc_id,
                    "sheet": sheet,
                    "row_range": (start, min(start + rows_per_chunk, len(df))),
                },
            )

    try:
        workbook = load_workbook(path)
        for worksheet in workbook.worksheets:
            for image in getattr(worksheet, "_images", []) or []:
                if hasattr(image, "_data"):
                    ocr = ocr_image_bytes(image._data())
                    if ocr:
                        yield RawBlock(
                            merge_text("", ocr),
                            {
                                "type": "xlsx_image",
                                "path": path,
                                "doc_id": doc_id,
                                "sheet": worksheet.title,
                                "has_ocr": True,
                            },
                        )
    except Exception:
        pass
```

## Адаптация текста локальным LLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_local_llm(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")


def adapt_text_with_llm(llm, text: str, max_new_tokens: int = 256) -> str:
    prompt = (
        "Ты помощник, который нормализует сырой фрагмент документа (включая текст OCR).\n"
        "Выдели факты, числа, единицы и названия сущностей.\n"
        "Если есть таблицы/диаграммы, кратко опиши ключевые выводы.\n\n"
        f"Фрагмент:\n{text}\n\nНормализованный фрагмент:"
    )
    result = llm(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
    return result.split("Нормализованный фрагмент:")[-1].strip()
```

## Чанкинг

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_blocks(blocks: Iterable[RawBlock], size: int = 1200, overlap: int = 150) -> List[RawBlock]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ".", " "],
    )
    chunks: List[RawBlock] = []
    for block in blocks:
        for part in splitter.split_text(block.text):
            meta = dict(block.meta)
            chunks.append(RawBlock(part, meta))
    return chunks
```

## Эмбеддинги и FAISS

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = "cpu") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        payload = [f"passage: {text}" for text in texts] if "e5" in self.model_name else texts
        embeddings = self.model.encode(payload, batch_size=64, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        payload = f"query: {query}" if "e5" in self.model_name else query
        embedding = self.model.encode([payload], normalize_embeddings=True)
        return np.asarray(embedding, dtype="float32")


class FaissStore:
    def __init__(self, dim: int, index_factory: str = "HNSW32") -> None:
        self.index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, texts: List[str], metas: List[Dict[str, Any]]) -> None:
        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query_vector: np.ndarray, top_k: int = 10):
        scores, ids = self.index.search(query_vector, top_k)
        return [
            (self.texts[idx], self.metas[idx], float(scores[0][offset]))
            for offset, idx in enumerate(ids[0])
            if idx != -1
        ]
```

## Гибридный ретрив и реранкинг

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers.cross_encoder import CrossEncoder


class HybridRetriever:
    def __init__(self, dense_store: FaissStore, texts: List[str]) -> None:
        self.dense = dense_store
        self.vectorizer = TfidfVectorizer(max_features=150_000)
        self.sparse_matrix = self.vectorizer.fit_transform(texts)
        self.texts = texts

    def tfidf_top(self, query: str, top_k: int = 100):
        query_vec = self.vectorizer.transform([query])
        similarities = linear_kernel(query_vec, self.sparse_matrix).ravel()
        top_indices = similarities.argsort()[::-1][:top_k]
        return [
            (self.texts[idx], self.dense.metas[idx], float(similarities[idx]))
            for idx in top_indices
        ]


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates, top_k: int = 10):
        pairs = [(query, text) for text, _, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        return [(text, meta, float(score)) for (text, meta, _), score in ranked[:top_k]]
```

## Ингест и поиск

```python
from pathlib import Path
import json
import time


def ingest_folder(
    root: str,
    embedder: Embedder,
    store: FaissStore,
    adapt_llm=None,
    use_adaptation: bool = False,
) -> int:
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        doc_id = f"{path.name}-{int(time.time())}"

        if ext == ".docx":
            blocks = list(parse_docx_with_ocr(str(path), doc_id))
        elif ext == ".pptx":
            blocks = list(parse_pptx_with_ocr(str(path), doc_id))
        elif ext == ".pdf":
            blocks = list(parse_pdf_with_ocr(str(path), doc_id))
        elif ext == ".xlsx":
            blocks = list(parse_xlsx_with_ocr(str(path), doc_id))
        else:
            continue

        if use_adaptation and adapt_llm is not None:
            for block in blocks:
                if block.meta.get("has_ocr"):
                    block.text = adapt_text_with_llm(adapt_llm, block.text)

        for chunk in split_blocks(blocks, size=1200, overlap=150):
            texts.append(chunk.text)
            metas.append(chunk.meta)

    embeddings = embedder.encode_passages(texts)
    store.add(embeddings, texts, metas)
    return len(texts)


def search_answer(
    query: str,
    embedder: Embedder,
    store: FaissStore,
    hybrid: Optional[HybridRetriever] = None,
    reranker: Optional[Reranker] = None,
    dense_top_k: int = 80,
    final_top_k: int = 10,
):
    query_vector = embedder.encode_query(query)
    dense_candidates = store.search(query_vector, dense_top_k)

    if hybrid:
        sparse_candidates = hybrid.tfidf_top(query, dense_top_k)
        pool = {}
        for text, meta, score in dense_candidates + sparse_candidates:
            key = (text, json.dumps(meta, ensure_ascii=False))
            pool[key] = max(pool.get(key, 0.0), score)
        candidates = [(text, json.loads(meta_json), score) for (text, meta_json), score in pool.items()]
    else:
        candidates = dense_candidates

    if reranker:
        ranked = reranker.rerank(query, candidates, top_k=final_top_k)
    else:
        ranked = sorted(candidates, key=lambda item: item[2], reverse=True)[:final_top_k]

    return ranked
```

## Рекомендованные параметры

| Тип | Размер чанка | Оверлап | Особенности |
| --- | --- | --- | --- |
| DOCX | 800–1200 токенов | 100–150 | OCR «к последнему абзацу», поиск подписей «Рис.» |
| PDF | 800–1200 | 100–150 | Привязка OCR по bbox к текстовым блокам |
| PPTX | 1 слайд | 0–50 | Сшиваем shapes, Notes, OCR |
| XLSX | 50 строк | 0 | Markdown + KV, опциональный OCR картинок листа |

Эмбеддинги: `intfloat/multilingual-e5-base` или `BAAI/bge-m3`. Реранкер: `cross-encoder/ms-marco-MiniLM-L-6-v2`, позже `bge-reranker-v2-m3`. Для FAISS — `HNSW32`, а при масштабировании `IVF4096,PQ64`.

## Практика индустрии

- **Диаграммы:** OCR-инъекция в соседние абзацы + адаптация LLM до «факто-фрагментов».
- **Презентации:** «Слайд = чанк», хранение конспектов от LLM в отдельном индексе.
- **PDF-отчёты:** удаление колонтитулов, сглаживание переносов, OCR таблиц через `pdfplumber`.
- **Excel:** индексируем окна строк и summary-блоки, чтобы отвечать на вопросы «где это посмотреть».
- **Гибридный поиск:** dense + TF-IDF + реранкер, иногда self-query retriever для метаданных.
- **Безопасность:** ACL в метаданных, шифрование индексов, PII-редактирование перед LLM.
- **Наблюдаемость:** метрики Recall@K, доля ответов с цитатами, latency, отчёт по пустым запросам.

## Оптимизации

- Кэшировать OCR и эмбеддинги по хэшу.
- Хранить raw/adapted варианты чанков и отдавать оба на реранк.
- Использовать FAISS IVF+PQ и шардирование по отделам.
- Сначала dense (K=100) + TF-IDF (K=100), затем реранкер на 200 → top-10.
- Поддерживать «золотой» набор Q/A менеджеров и регулярно обновлять пороги семантического чанкинга.

## Мини-пример запуска

```python
embedder = Embedder("intfloat/multilingual-e5-base", device="cpu")
dimension = embedder.encode_passages(["passage: тест"]).shape[1]
store = FaissStore(dimension, index_factory="HNSW32")

# llm = load_local_llm("/models/ru-instruct")
count = ingest_folder("./kb", embedder, store, adapt_llm=None, use_adaptation=False)
print("Indexed chunks:", count)

hybrid = HybridRetriever(store, store.texts)
reranker = Reranker()
results = search_answer(
    "Покажи KPI по регионам в 2024 и выводы по диаграммам",
    embedder,
    store,
    hybrid,
    reranker,
    dense_top_k=80,
    final_top_k=10,
)

for text, meta, score in results:
    location = meta.get("page") or meta.get("slide") or meta.get("sheet")
    print(round(score, 3), meta.get("path"), location)
    print(text[:200].replace("\n", " "), "...")
```

## Дальнейшее развитие

- Добавить семантическое сегментирование перед чанкингом (границы по провалу косинусной близости).
- Индексировать CLIP-эмбеддинги параллельно с текстом.
- Автоматически извлекать факты из диаграмм через LLM («перечисли пары (категория, значение)»).
- Включить self-query retriever и guardrails (обязательные цитаты, запрет галлюцинаций).
```
