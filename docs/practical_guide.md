# Практическое руководство по построению корпоративной RAG-системы

Данное руководство описывает, как на базе текущего каркаса репозитория собрать рабочую
корпоративную RAG-систему для менеджеров, умеющую работать с форматами **DOCX**, **PDF**,
**XLSX**, **PPTX** и кастомным **SCRAM**. Материал структурирован по ключевым этапам:

1. Архитектура пайплайна.
2. Парсинг и извлечение данных из файлов.
3. Стратегии разбиения на чанки.
4. Настройка эмбеддингов и хранилища векторов.
5. Сквозной ingestion-процесс.
6. Сборка RAG-контура и генерации ответа.
7. Обработка изображений, таблиц и мультимодальных данных.
8. Типовые проблемы и способы их решения.
9. Практический пример развёртывания.
10. Значения параметров по умолчанию.
11. Оффлайн-установка зависимостей.
12. Контроль качества, ACL и безопасность.

## 0. Быстрый план архитектуры

```
[Файловые источники]
      │ (watcher / batch)
      ▼
[Парсеры по типам] → [Нормализация] → [Чанкинг]
      │                               │
      └──────────────► [Метаданные] ◄─┘
                         (doc_id, тип, автор, лист/слайд/страница, путь заголовков, диапазоны ячеек)
      ▼
[Эмбеддинги] (батчами, кэш)
      ▼
[Векторный стор] (FAISS локально / Pinecone / Qdrant / Milvus)
      ▼
[Гибридный ретрив] (dense + BM25/TF‑IDF + реранкер)
      ▼
[LLM‑чейн: слияние контекстов, цитирование, контроль размера]
      ▼
[Ответ менеджеру + ссылки на источники]
```

## 1. Парсинг и извлечение содержимого по типам файлов

### Общие принципы

* Сохраняйте структуру и координаты: страницы (PDF), заголовки уровней (DOCX), номера
  слайдов и заметки (PPTX), имена листов и диапазоны ячеек (XLSX).
* Нормализуйте текст: убирайте двойные пробелы, мягкие переносы (`-\n`), повторяющиеся
  футеры/хедеры, приводите табличный текст к читаемому виду.
* Для изображений и сканов используйте OCR (например, `pytesseract`), а координаты
  сохраните в метаданные.

### Интерфейс парсеров

```python
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

@dataclass
class RawBlock:
    text: str
    meta: Dict[str, Any]

class BaseParser:
    def parse(self, path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
        raise NotImplementedError
```

### DOCX → абзацы + путь заголовков

```python
import re
from docx import Document

class DocxParser(BaseParser):
    def parse(self, path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
        doc = Document(path)
        headings_stack: list[str] = []

        def heading_level(style_name: str) -> Optional[int]:
            match = re.search(r"(\d+)", style_name or "")
            return int(match.group(1)) if match else None

        for paragraph in doc.paragraphs:
            style = getattr(paragraph.style, "name", "") or ""
            text = paragraph.text.strip()
            if not text:
                continue
            level = heading_level(style)
            if level:
                while len(headings_stack) >= level:
                    headings_stack.pop()
                headings_stack.append(text)
                yield RawBlock(text=text, meta=dict(
                    doc_id=doc_id,
                    type="docx",
                    path=path,
                    headings=headings_stack.copy(),
                    is_heading=True,
                ))
            else:
                yield RawBlock(text=text, meta=dict(
                    doc_id=doc_id,
                    type="docx",
                    path=path,
                    headings=headings_stack.copy(),
                ))
```

### PDF → страницы и блоки

```python
import re
import fitz  # PyMuPDF

class PdfParser(BaseParser):
    def parse(self, path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
        document = fitz.open(path)
        for page_num, page in enumerate(document, start=1):
            for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
                normalized = (text or "").strip()
                if not normalized:
                    continue
                normalized = re.sub(r"-\n", "", normalized)
                normalized = re.sub(r"\s+\n", "\n", normalized)
                yield RawBlock(
                    text=normalized,
                    meta=dict(
                        doc_id=doc_id,
                        type="pdf",
                        path=path,
                        page=page_num,
                        bbox=(x0, y0, x1, y1),
                    ),
                )
```

#### Извлечение таблиц из PDF

```python
import pdfplumber
import pandas as pd

def extract_pdf_tables(path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                dataframe = pd.DataFrame(table[1:], columns=table[0])
                text = dataframe.to_markdown(index=False)
                yield RawBlock(
                    text=text,
                    meta=dict(
                        doc_id=doc_id,
                        type="pdf_table",
                        path=path,
                        page=page_num,
                    ),
                )
```

### XLSX → листы и окна строк

```python
import pandas as pd

class XlsxParser(BaseParser):
    def parse(self, path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            frame = workbook.parse(sheet_name)
            if frame.empty:
                continue
            summary = (
                f"Лист: {sheet_name}\n"
                f"Колонки: {', '.join(map(str, frame.columns))}\n"
                f"Строк: {len(frame)}"
            )
            yield RawBlock(summary, meta=dict(
                doc_id=doc_id,
                type="xlsx_sheet",
                path=path,
                sheet=sheet_name,
            ))

            window = 50
            for start in range(0, len(frame), window):
                chunk = frame.iloc[start:start + window]
                text = chunk.to_markdown(index=False)
                yield RawBlock(text, meta=dict(
                    doc_id=doc_id,
                    type="xlsx_rows",
                    path=path,
                    sheet=sheet_name,
                    row_range=(start, min(start + window, len(frame))),
                ))
```

#### Строки как пары ключ-значение

```python
def row_as_kv(row: pd.Series) -> str:
    return " | ".join(f"{column}: {row[column]}" for column in row.index)
```

### PPTX → слайды и заметки

```python
from pptx import Presentation

class PptxParser(BaseParser):
    def parse(self, path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
        presentation = Presentation(path)
        for slide_index, slide in enumerate(presentation.slides, start=1):
            fragments: list[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    payload = (shape.text or "").strip()
                    if payload:
                        fragments.append(payload)
            if slide.has_notes_slide and slide.notes_slide and slide.notes_slide.notes_text_frame:
                notes = (slide.notes_slide.notes_text_frame.text or "").strip()
                if notes:
                    fragments.append(f"Notes: {notes}")
            if fragments:
                yield RawBlock("\n\n".join(fragments), meta=dict(
                    doc_id=doc_id,
                    type="pptx",
                    path=path,
                    slide=slide_index,
                ))
```

### SCRAM → кастомный формат

```python
import os

class ScramParser(BaseParser):
    def __init__(self, decrypt_fn, inner_parser_selector):
        self.decrypt_fn = decrypt_fn
        self.inner_parser_selector = inner_parser_selector

    def parse(self, path: str, doc_id: Optional[str] = None) -> Iterable[RawBlock]:
        payload = self.decrypt_fn(path)
        parser = self.inner_parser_selector(payload)
        tmp_path = f"{path}.decoded"
        with open(tmp_path, "wb") as handler:
            handler.write(payload)
        try:
            yield from parser.parse(tmp_path, doc_id=doc_id)
        finally:
            os.remove(tmp_path)
```

## 2. Чанкинг: стратегии

### 2.1 Структурный

* DOCX — по заголовкам H1/H2/H3, добавляя overlap 1–2 абзаца.
* PDF — по страницам и блокам, объединяя мелкие блоки до размера 600–1200 токенов.
* PPTX — один слайд + заметки = один чанк.
* XLSX — summary листа + окна строк (20–100) + KV-представление.

### 2.2 Семантический

```python
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def naive_ru_sent_split(text: str) -> list[str]:
    sentences = re.split(r"(?<=[\.\?\!])\s+(?=[А-ЯA-ZЁ])", text.strip())
    return [segment.strip() for segment in sentences if segment.strip()]

def semantic_chunk(
    text: str,
    model: SentenceTransformer,
    max_tokens: int = 512,
    sim_threshold: float = 0.55,
    overlap_sents: int = 1,
) -> list[str]:
    sentences = naive_ru_sent_split(text)
    if not sentences:
        return []
    embeddings = model.encode(sentences, batch_size=64, normalize_embeddings=True)
    similarities = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal()

    boundaries = [0]
    for index, similarity in enumerate(similarities, start=1):
        if similarity < sim_threshold:
            boundaries.append(index)
    boundaries.append(len(sentences))

    chunks: list[str] = []
    pointer = 0

    def approx_tokens(payload: str) -> int:
        return max(1, int(len(payload) / 3.5))

    while pointer < len(boundaries) - 1:
        start = boundaries[pointer]
        end = boundaries[pointer + 1]
        piece = " ".join(sentences[start:end])

        while (
            pointer + 1 < len(boundaries) - 1
            and approx_tokens(piece) < max_tokens
        ):
            next_start, next_end = boundaries[pointer + 1], boundaries[pointer + 2]
            candidate = piece + " " + " ".join(sentences[next_start:next_end])
            if approx_tokens(candidate) <= max_tokens:
                piece = candidate
                pointer += 1
            else:
                break

        if overlap_sents and end < len(sentences):
            back = max(start, end - overlap_sents)
            piece += " " + " ".join(sentences[back:end])

        chunks.append(piece)
        pointer += 1

    return chunks
```

### 2.3 RecursiveCharacterTextSplitter

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n## ", "\n### ", "\n\n", "\n", ".", " "],
)
chunks = splitter.split_text(big_text)
```

### 2.4 Табличные данные

* Формируйте чанки по ключевым колонкам и диапазонам строк.
* Добавляйте в метаданные `sheet`, `row_range`, `key_columns`.
* Сохраняйте одновременно Markdown, KV-пары и краткий summary.

## 3. Эмбеддинги и векторное хранилище

### Рекомендуемые модели

* **`intfloat/multilingual-e5-base` / `-large`** — сильная мультиязычная модель, требует
  префиксы `query:` / `passage:`.
* **`BAAI/bge-m3`** — гибридная модель с поддержкой sparse/multi-vector и реранкера.
* **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** — бюджетный вариант.

### Имплементация на sentence-transformers + FAISS

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if "e5" in self.model_name:
            texts = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        if "e5" in self.model_name:
            query = f"query: {query}"
        vector = self.model.encode([query], normalize_embeddings=True)
        return np.asarray(vector, dtype="float32")

class FaissStore:
    def __init__(self, dim: int, index_factory: str = "HNSW32", use_gpu: bool = False):
        index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
        if use_gpu:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        self.index = index
        self.texts: list[str] = []
        self.metas: list[dict] = []

    def add(self, vectors: np.ndarray, texts: list[str], metas: list[dict]):
        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query_vector: np.ndarray, top_k: int = 10):
        scores, ids = self.index.search(query_vector, top_k)
        results = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.texts[idx], self.metas[idx], float(score)))
        return results
```

### Гибридный поиск и реранкер

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers.cross_encoder import CrossEncoder

class HybridRetriever:
    def __init__(self, dense_store: FaissStore, texts: list[str]):
        self.dense = dense_store
        self.vectorizer = TfidfVectorizer(max_features=150_000)
        self.sparse_matrix = self.vectorizer.fit_transform(texts)
        self.texts = texts

    def bm25_like(self, query: str, top_k: int = 50):
        vector = self.vectorizer.transform([query])
        sims = linear_kernel(vector, self.sparse_matrix).ravel()
        top_indices = sims.argsort()[::-1][:top_k]
        return [(self.texts[i], self.dense.metas[i], float(sims[i])) for i in top_indices]

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[tuple[str, dict, float]], top_k: int = 10):
        pairs = [(query, candidate[0]) for candidate in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        return [(candidate[0], candidate[1], float(score)) for candidate, score in ranked[:top_k]]
```

## 4. Ингест-пайплайн: объединение компонентов

```python
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

class EmbeddingCache:
    def __init__(self, path: str = "emb_cache.json"):
        try:
            self.db = json.load(open(path, "r", encoding="utf-8"))
        except FileNotFoundError:
            self.db = {}
        self.path = path

    def get(self, key: str):
        return self.db.get(key)

    def set(self, key: str, vector):
        self.db[key] = vector
        json.dump(self.db, open(self.path, "w", encoding="utf-8"), ensure_ascii=False)

def ingest_directory(
    dir_path: str,
    embedder: Embedder,
    store: FaissStore,
    parsers_by_ext: dict[str, BaseParser],
    cache: EmbeddingCache | None = None,
):
    texts: list[str] = []
    metas: list[dict] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", ".", " "],
    )

    for path in Path(dir_path).rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower().lstrip(".")
        parser = parsers_by_ext.get(ext)
        if not parser:
            continue
        doc_id = f"{path.name}-{int(time.time())}"
        for block in parser.parse(str(path), doc_id=doc_id):
            payload = block.text.strip()
            if not payload:
                continue
            for chunk in splitter.split_text(payload):
                texts.append(chunk)
                metas.append(block.meta)

    cache = cache or EmbeddingCache()
    vectors: list[np.ndarray] = []
    for text in texts:
        key = hash_text(text)
        cached = cache.get(key)
        if cached is None:
            cached = embedder.encode_passages([text])[0].tolist()
            cache.set(key, cached)
        vectors.append(np.asarray(cached, dtype="float32"))

    embeddings = np.vstack(vectors)
    store.add(embeddings, texts, metas)
```

## 5. Минимальный RAG-конвейер

```python
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # замените на локальную LLM

class FaissRetriever:
    def __init__(self, embedder: Embedder, store: FaissStore):
        self.embedder = embedder
        self.store = store

    def get_relevant_documents(self, query: str, k: int = 10) -> list[Document]:
        vector = self.embedder.encode_query(query)
        results = self.store.search(vector, top_k=k)
        documents = []
        for text, meta, score in results:
            metadata = meta.copy()
            metadata["score"] = score
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

retriever = FaissRetriever(embedder, store)
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.1),
    chain_type="stuff",
    retriever=retriever,
)
print(qa_chain.run("Какие KPI у нас приняты для региональных менеджеров?"))
```

## 6. Мультимодальность и таблицы

* Для изображений и сканов используйте `pytesseract` с предварительной обработкой
  (бинаризация, поворот, 300 DPI).
* Сохраните `bbox` и идентификаторы изображений в метаданные для последующей отрисовки.
* Таблицы индексируйте в виде Markdown и KV-представления, а при необходимости выполняйте
  агрегации на лету (через `pandas`).

```python
import pytesseract
from PIL import Image

def ocr_image_to_text(path: str) -> str:
    image = Image.open(path)
    return pytesseract.image_to_string(image, lang="rus+eng")
```

## 7. Проблемы и решения

| Проблема | Решение |
| --- | --- |
| Сканированные PDF и шум | OCR + предобработка изображений, удаление повторяющихся футеров, дефисов и мусорных блоков |
| Очень большие файлы | Стриминговый парсинг, батч-эмбеддинг, FAISS IVF/PQ, шардирование по отделам |
| Повторы и дубликаты | MinHash/SimHash, SHA256-хэши чанков, кэш эмбеддингов |
| Конфиденциальность (ACL) | Метаданные `tenant`, `department`, `acl`, пост-фильтрация результатов, изолированные индексы |
| Шум в текстах | Регулярные выражения для удаления повторяющихся паттернов, лимиты на длину мусорных блоков |
| Потеря смысла при чанкинге | Структурный + семантический чанкинг с overlap, контроль coverage@k |
| Медленный ретрив | HNSW/IVF, ANN-поиск (K=200) + реранкер, кэширование запросов, батчинг LLM |
| Смешение языков | Мультиязычные модели, унификация форматов дат и валют |

## 8. Лучшие практики

* Гибридный поиск + реранкер обеспечивают качество ответов.
* Чанки 400–1200 токенов, overlap 10–20 % (или 100–150 токенов).
* Отдельные политики обработки по типу документа.
* Лёгкий реранкер можно дообучить на внутренних Q/A.
* Инфраструктура: батчи, шардирование, квантование (PQ), атомарные обновления индекса.
* Наблюдаемость: Recall@K, MRR@10, latency LLM, отчёты по пустым ответам.

## 9. Полный пример сквозного сценария

```python
parsers = {
    "docx": DocxParser(),
    "pdf": PdfParser(),
    "xlsx": XlsxParser(),
    "pptx": PptxParser(),
    # "scram": ScramParser(decrypt_fn=..., inner_parser_selector=...)
}

embedder = Embedder("intfloat/multilingual-e5-base", device="cpu")
dim = embedder.encode_passages(["passage: тест"]).shape[1]
store = FaissStore(dim=dim, index_factory="HNSW32")

ingest_directory("./knowledge_base", embedder, store, parsers)

retriever = HybridRetriever(store, store.texts)
dense_candidates = store.search(embedder.encode_query("Регламенты по KPI"), top_k=80)
sparse_candidates = retriever.bm25_like("Регламенты по KPI", top_k=80)

combined = {(text, json.dumps(meta, ensure_ascii=False)): score for text, meta, score in dense_candidates}
for text, meta, score in sparse_candidates:
    key = (text, json.dumps(meta, ensure_ascii=False))
    combined[key] = max(combined.get(key, 0.0), score)

candidates = [(text, json.loads(meta_json), score) for (text, meta_json), score in combined.items()]
reranker = Reranker()
relevant = reranker.rerank("Регламенты по KPI", candidates, top_k=10)

for text, meta, score in relevant:
    print(round(score, 3), meta.get("path"), meta.get("page") or meta.get("slide") or meta.get("sheet"))
```

## 10. Параметры по умолчанию

| Формат | Стратегия | Размер чанка | Overlap |
| --- | --- | --- | --- |
| DOCX | H1/H2/H3 → семантика | 600–1000 токенов | 100–150 |
| PDF | Страница/блок → объединение | 800–1200 токенов | 100 |
| PPTX | Слайд + notes | ~400 токенов | 0 |
| XLSX | Summary + окна строк | 20–100 строк | 0–10 % |

Эмбеддинги: `e5-base`/`e5-large`, альтернативно `bge-m3`.
Векторное хранилище: `FAISS` с `HNSW32`, для больших объёмов — `IVF + PQ`.
Реранкер: `ms-marco-MiniLM-L-6-v2` → затем `bge-reranker-v2-m3`.

## 11. Оффлайн-установка зависимостей

1. На машине с интернетом выполните `pip download -r requirements.txt -d wheels/`.
2. Перенесите каталог `wheels/` в защищённый контур.
3. Установите зависимости: `pip install --no-index --find-links wheels/ -r requirements.txt`.
4. Модели `sentence-transformers` скачайте заранее и укажите локальный путь в `SentenceTransformer`.

Минимальный `requirements.txt`:

```
pymupdf
python-docx
python-pptx
pandas
openpyxl
sentence-transformers
faiss-cpu
scikit-learn
pdfplumber
pytesseract
langchain
```

## 12. Контроль качества и безопасность

* Фильтруйте результаты по ACL (`department`, `role`, `tenant`) перед показом менеджеру.
* Реализуйте PII-редактор перед генерацией ответа:

```python
import re

def redact_pii(text: str) -> str:
    text = re.sub(r"\b\d{16}\b", "[CARD]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ID]", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[EMAIL]", text)
    text = re.sub(r"\+?\d[\d\-\s]{7,}\d", "[PHONE]", text)
    return text
```

* Если confidence низкая, верните менеджеру «Не найдено» + ссылки на релевантные документы.
* Отслеживайте метрики: `Recall@K`, `MRR@10`, `embed_seconds`, `retrieval_latency`, `llm_latency`,
  «пустые» ответы, частоту обновлений индекса.

---

Это руководство можно использовать как чек-лист при поэтапной реализации сервисов,
описанных в `services/ingestion` и `services/online-rag`, а также при наполнении модулей
из каталога `services/shared`.
