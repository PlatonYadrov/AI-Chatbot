# ===================== Token-aware Docling chunker (hard ≤ HARD_MAX) =====================
import re
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

# ---- параметры по умолчанию (под RAG) ----
TARGET_TOKENS = 400     # желаемый размер чанка
HARD_MAX      = 512     # потолок токенов на чанк (жёсткая гарантия)
MIN_TOKENS    = 120     # минимальный размер (коротыши будут склеиваться)
OVERLAP_SENTS = 1       # оверлап по предложениям между чанками

# Разделение на предложения (простое, EN/RU)
_sent_re = re.compile(r"(?<=[\.\!\?])\s+(?=[A-ZА-Я0-9])")

def split_sents(txt: str) -> List[str]:
    parts = [s.strip() for s in _sent_re.split(txt) if s.strip()]
    return parts or ([txt] if txt else [])

# ---- токенайзер (intfloat/multilingual-e5-large) должен быть заранее создан пользователем ----
# tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

def tlen(text: str) -> int:
    """Точный подсчёт токенов e5 без усечения и без спец-символов."""
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))

# ---- таблицы: поддержка и объектной и dict-структуры ----
def _cell_text(cell) -> str:
    """Достаёт человекочитаемый текст ячейки."""
    if isinstance(cell, str):
        return cell
    if isinstance(cell, dict):
        return str(cell.get("text", ""))
    return str(getattr(cell, "text", cell))

def _extract_grid(table_data):
    """Возвращает 2D-список ячеек (объект/словарь поддерживаются)."""
    if hasattr(table_data, "grid"):
        return table_data.grid
    if isinstance(table_data, dict) and "grid" in table_data:
        return table_data["grid"]
    return None

def _grid_to_rows(grid) -> List[List[str]]:
    """Преобразует сетку таблицы в список строк (списков ячеек)."""
    if not grid or not isinstance(grid, list) or not grid[0]:
        return []
    return [[_cell_text(c) for c in row] for row in grid]

def _rows_to_markdown(rows: List[List[str]]) -> str:
    """Markdown из строк (первая строка — заголовок)."""
    if not rows:
        return ""
    header = " | ".join(rows[0])
    sep    = " | ".join(["---"] * len(rows[0]))
    body   = [" | ".join(r) for r in rows[1:]]
    return "\n".join([header, sep, *body])

def _split_markdown_table(rows: List[List[str]], max_tokens: int) -> List[str]:
    """
    Режет markdown-таблицу по строкам так, чтобы каждый фрагмент ≤ max_tokens.
    Заголовок повторяется в каждом фрагменте.
    """
    if not rows:
        return []
    header = rows[0]
    sep = ["---"] * len(header)

    parts = []
    cur_rows = [header, sep]
    cur_txt = _rows_to_markdown(cur_rows)
    cur_len = tlen(cur_txt)

    for r in rows[1:]:
        # попробуем добавить строку
        candidate = "\n".join([_rows_to_markdown(cur_rows), " | ".join(r)])
        cand_len = tlen(candidate)
        if cand_len <= max_tokens:
            # обновим cur_rows/cur_len аккуратно
            cur_rows.append(r)
            cur_txt = candidate
            cur_len = cand_len
        else:
            # зафиксируем текущий фрагмент
            parts.append(_rows_to_markdown(cur_rows))
            # начнём новый с повтором заголовка
            cur_rows = [header, sep, r]
            cur_txt = _rows_to_markdown(cur_rows)
            cur_len = tlen(cur_txt)
            # если даже одна строка с заголовком не помещается (крайне редко):
            if cur_len > max_tokens:
                # жёстко усечём по ячейкам в этой строке
                safe_cells = []
                for cell in r:
                    probe = "\n".join([_rows_to_markdown([header, sep]), " | ".join(safe_cells + [cell])])
                    if tlen(probe) <= max_tokens:
                        safe_cells.append(cell)
                    else:
                        break
                cur_rows = [header, sep, safe_cells] if safe_cells else [header, sep]
                cur_txt = _rows_to_markdown(cur_rows)
                cur_len = tlen(cur_txt)

    if tlen(_rows_to_markdown(cur_rows)) > 0:
        parts.append(_rows_to_markdown(cur_rows))
    return parts

# ---- основной чанкер ----
def chunk_docling_token_packer(
    doc, *,
    target_tokens: int = TARGET_TOKENS,
    hard_max: int      = HARD_MAX,
    min_tokens: int    = MIN_TOKENS,
    overlap_sents: int = OVERLAP_SENTS,
    keep_tables_md: bool = True,
    table_mode: str = "inline",        # "inline" | "separate"
    doc_id: Optional[str] = None,
    src_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Упаковывает элементы DoclingDocument в токено-осознанные чанки (жёстко ≤ hard_max).
    - Заголовки открывают новую секцию и включаются в текст чанка.
    - Таблицы:
        * inline: превращаются в Markdown и идут вместе с текстом (режутся по строкам при необходимости)
        * separate: каждая таблица — отдельный чанк; если не влезает — режется по строкам на несколько чанков
    - Длинные абзацы режутся по предложениям.
    - Коротыши склеиваются постфактум (без превышения hard_max).

    Возвращает: List[{"text": str, "meta": dict}]
    meta: doc_id, path, section_path, page_start/end, block_start/end, types, table_index?
    """
    assert table_mode in ("inline", "separate"), "table_mode должен быть 'inline' или 'separate'"

    chunks: List[Dict[str, Any]] = []

    cur_text: List[str] = []
    cur_tokens = 0
    cur_meta = {
        "doc_id": doc_id, "path": src_path,
        "types": set(), "section_path": [],
        "page_start": None, "page_end": None,
        "block_start": None, "block_end": None,
    }
    last_tail: List[str] = []  # предложения для overlap
    table_index = 0            # сквозная нумерация таблиц
    block_idx = -1

    def flush():
        nonlocal cur_text, cur_tokens, last_tail
        if not cur_text:
            return
        text = "\n\n".join([x for x in cur_text if x.strip()])
        # контроль безопасности
        if tlen(text) > hard_max:
            # крайний случай: ещё раз дорежем по предложениям
            sents = split_sents(text)
            buf = []
            buf_len = 0
            safe_chunks = []
            for s in sents:
                sl = tlen(s)
                if buf and buf_len + sl > hard_max:
                    safe_chunks.append("\n".join(buf))
                    buf, buf_len = [], 0
                if sl > hard_max:
                    # очень длинное «предложение»: усечём по символам как fallback
                    piece = s
                    while tlen(piece) > hard_max:
                        # грубое деление пополам по символам
                        half = max(1, len(piece)//2)
                        left, piece = piece[:half], piece[half:]
                        # добить левую сторону до ≤ hard_max
                        while tlen(left) > hard_max and len(left) > 1:
                            left = left[:-1]
                        safe_chunks.append(left)
                    if piece:
                        buf.append(piece); buf_len += tlen(piece)
                else:
                    buf.append(s); buf_len += sl
            if buf:
                safe_chunks.append("\n".join(buf))
            # записываем безопасные куски как отдельные чанки
            for sc in safe_chunks:
                chunks.append({"text": sc, "meta": {**cur_meta, "types": sorted(cur_meta["types"])}})
        else:
            sents = split_sents(text)
            last_tail = sents[-overlap_sents:] if overlap_sents and sents else []
            chunks.append({"text": text, "meta": {**cur_meta, "types": sorted(cur_meta["types"])}})

        # сброс для нового чанка (кроме хлебных крошек section_path)
        cur_text.clear(); cur_tokens = 0
        cur_meta["page_start"] = None; cur_meta["page_end"] = None
        cur_meta["block_start"] = None; cur_meta["block_end"] = None
        cur_meta["types"] = set()

    def try_put_text(piece: str, typ: str, page_no, blk_idx, *, allow_overlap=True):
        """Безопасно положить текст в текущий чанк, не превышая hard_max."""
        nonlocal cur_tokens
        if not piece.strip():
            return
        tok = tlen(piece)

        if tok > hard_max:
            # режем по предложениям
            for sent in split_sents(piece):
                s_tok = tlen(sent)
                if s_tok > hard_max:
                    # emergency: дробим по символам
                    frag = sent
                    while tlen(frag) > hard_max:
                        half = max(1, len(frag)//2)
                        left, frag = frag[:half], frag[half:]
                        while tlen(left) > hard_max and len(left) > 1:
                            left = left[:-1]
                        try_put_text(left, typ, page_no, blk_idx, allow_overlap=False)
                    if frag:
                        try_put_text(frag, typ, page_no, blk_idx, allow_overlap=False)
                else:
                    if cur_text and cur_tokens + s_tok > hard_max:
                        flush()
                    cur_text.append(sent); cur_tokens += s_tok
                    cur_meta["types"].add(typ)
                    cur_meta["block_start"] = cur_meta["block_start"] or blk_idx
                    cur_meta["block_end"] = blk_idx
                    if cur_meta["page_start"] is None: cur_meta["page_start"] = page_no
                    cur_meta["page_end"] = page_no or cur_meta["page_end"]
            return

        # обычное добавление
        if cur_text and cur_tokens + tok > hard_max:
            flush()
        cur_text.append(piece); cur_tokens += tok
        cur_meta["types"].add(typ)
        cur_meta["block_start"] = cur_meta["block_start"] or blk_idx
        cur_meta["block_end"] = blk_idx
        if cur_meta["page_start"] is None: cur_meta["page_start"] = page_no
        cur_meta["page_end"] = page_no or cur_meta["page_end"]

    def maybe_new_chunk_for(tokens_needed: int) -> None:
        """Открывает новый чанк, если добавление переполнит target/hard_max."""
        nonlocal cur_tokens
        if not cur_text:
            return
        if cur_tokens + tokens_needed <= hard_max and cur_tokens + tokens_needed <= target_tokens:
            return
        # Если текущий чанк ещё слишком мал — лучше потерпеть, чтобы не плодить коротышей
        if cur_tokens < int(0.6 * target_tokens) and cur_tokens + tokens_needed <= hard_max:
            return
        flush()
        # переносим overlap из прошлого чанка (хвост предложений), если влезает
        if last_tail:
            pre = " ".join(last_tail)
            pre_len = tlen(pre)
            if pre_len < int(0.2 * target_tokens) and pre_len <= hard_max:
                cur_text.append(pre); cur_tokens += pre_len

    # -------- обходим элементы документа --------
    for item, level in doc.iterate_items():
        block_idx += 1
        d = item.model_dump() if hasattr(item, "model_dump") else {}
        label = (d.get("label") or "text").lower()

        # провенанс
        page_no, bbox = None, None
        prov = getattr(item, "prov", None)
        prov_list = prov if isinstance(prov, list) else ([prov] if prov else [])
        for p in prov_list:
            if page_no is None and hasattr(p, "page_no"):
                page_no = getattr(p, "page_no")
            if bbox is None and hasattr(p, "bbox") and getattr(p, "bbox", None):
                b = p.bbox; bbox = (b.l, b.t, b.r, b.b)

        # --- заголовки (обновляют section_path и включаются в текст) ---
        if label in {"title", "section_header", "heading"}:
            flush()  # заголовок открывает логический раздел
            text = (d.get("text") or "").strip()
            try: h = int(level) if level is not None else 2
            except: h = 2
            while len(cur_meta["section_path"]) >= h:
                cur_meta["section_path"].pop()
            if text:
                cur_meta["section_path"].append(text)
                header = f"{'#' * min(6, h+1)} {text}"
                h_len = tlen(header)
                maybe_new_chunk_for(h_len)
                try_put_text(header, "heading", page_no, block_idx)
            continue

        # --- таблицы ---
        if label == "table":
            data = d.get("data")
            grid = _extract_grid(data)
            if keep_tables_md and grid:
                rows = _grid_to_rows(grid)
                if not rows:
                    continue
                # если режим separate — каждый фрагмент таблицы становится отдельным чанком
                if table_mode == "separate":
                    # закроем текущий чанк (чтобы таблица шла отдельно)
                    flush()
                    table_index += 1
                    md_parts = _split_markdown_table(rows, hard_max)
                    for part in md_parts:
                        # гарантировано ≤ hard_max
                        chunks.append({
                            "text": part,
                            "meta": {
                                "doc_id": doc_id, "path": src_path,
                                "types": ["table"],
                                "section_path": list(cur_meta["section_path"]),
                                "page_start": page_no, "page_end": page_no,
                                "block_start": block_idx, "block_end": block_idx,
                                "table_index": table_index,
                            }
                        })
                else:
                    # inline: включаем таблицу в текущий поток, при необходимости дробя её
                    md_parts = _split_markdown_table(rows, hard_max)
                    for part in md_parts:
                        ptok = tlen(part)
                        maybe_new_chunk_for(ptok)
                        try_put_text(part, "table", page_no, block_idx)
                        cur_meta["table_index"] = table_index + 1  # предварительно присвоим
                    table_index += 1
            else:
                # нет grid — кладём текстовое представление, при необходимости режем
                md_table = str(data) if data is not None else ""
                if not md_table.strip():
                    continue
                if table_mode == "separate":
                    flush()
                    table_index += 1
                    # безопасно положим как текст (с разбиением)
                    try_put_text(md_table, "table", page_no, block_idx)
                    # переведём последний(е) добавленные куска в режим "separate"? — уже идут отдельными чанками из-за flush() + try_put_text()
                    # добавим table_index в последний чанк
                    if chunks:
                        chunks[-1]["meta"]["table_index"] = table_index
                        chunks[-1]["meta"]["types"] = sorted(set(chunks[-1]["meta"]["types"]) | {"table"})
                else:
                    maybe_new_chunk_for(tlen(md_table))
                    try_put_text(md_table, "table", page_no, block_idx)
                    table_index += 1
            continue

        # --- обычный текст/прочие элементы ---
        block_text = (d.get("text") or "").rstrip()
        if not block_text.strip():
            continue

        block_tokens = tlen(block_text)

        # Очень длинный абзац — режем по предложениям и упаковываем
        if block_tokens > hard_max:
            for sent in split_sents(block_text):
                try_put_text(sent, "text", page_no, block_idx)
            continue

        # обычная упаковка целого блока
        maybe_new_chunk_for(block_tokens)
        try_put_text(block_text, "text" if label != "list_item" else "text", page_no, block_idx)

    # доброс последнего
    flush()

    # ---- постобработка: склейка коротышей (с уважением к hard_max) ----
    def merge_shorts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not items:
            return items
        out = [items[0]]
        for c in items[1:]:
            left = out[-1]
            # можно ли слить и не превысить лимит?
            if tlen(left["text"]) < min_tokens and tlen(left["text"]) + tlen(c["text"]) <= hard_max:
                left["text"] = left["text"].rstrip() + "\n\n" + c["text"]
                left["meta"]["types"] = sorted(set(left["meta"]["types"]) | set(c["meta"]["types"]))
                left["meta"]["page_end"] = c["meta"]["page_end"] or left["meta"]["page_end"]
                left["meta"]["block_end"] = c["meta"]["block_end"]
                if "table_index" in c["meta"] and "table_index" not in left["meta"]:
                    left["meta"]["table_index"] = c["meta"]["table_index"]
            else:
                out.append(c)
        # если последний остался маленьким — аккуратно попробуем склеить с предпоследним
        if len(out) >= 2 and tlen(out[-1]["text"]) < min_tokens:
            prev = out[-2]; last = out[-1]
            if tlen(prev["text"]) + tlen(last["text"]) <= hard_max:
                prev["text"] = prev["text"].rstrip() + "\n\n" + last["text"]
                prev["meta"]["types"] = sorted(set(prev["meta"]["types"]) | set(last["meta"]["types"]))
                prev["meta"]["page_end"] = last["meta"]["page_end"] or prev["meta"]["page_end"]
                prev["meta"]["block_end"] = last["meta"]["block_end"]
                if "table_index" in last["meta"] and "table_index" not in prev["meta"]:
                    prev["meta"]["table_index"] = last["meta"]["table_index"]
                out.pop()
        return out

    return merge_shorts(chunks)

# ---- утилиты вывода/конвертации ----

def pretty_print_chunks(chunks: List[Dict[str, Any]]) -> None:
    for i, ch in enumerate(chunks):
        m = ch["meta"]
        pages = f"{m.get('page_start')}–{m.get('page_end')}" if m.get("page_start") is not None else "?"
        print(f"[Chunk {i}] path={m.get('path')} pages={pages} types={m.get('types')}")
        if "table" in m.get("types", []):
            ti = m.get("table_index")
            if ti:
                print(f"(Table #{ti})")
        print(ch["text"])
        print("-" * 80)

def to_langchain_docs(chunks: List[Dict[str, Any]]):
    try:
        from langchain.schema import Document
    except Exception:
        raise RuntimeError("Установи langchain: pip install langchain")
    docs = []
    for ch in chunks:
        meta = dict(ch["meta"])
        if isinstance(meta.get("types"), set):
            meta["types"] = sorted(meta["types"])
        docs.append(Document(page_content=ch["text"], metadata=meta))
    return docs

# === доп. утилита: отчёт по длинам чанков ===
def print_chunk_lengths(chunks):
    rows = []
    max_tokens = 1
    for i, ch in enumerate(chunks):
        txt = ch.get("text", "")
        meta = ch.get("meta", {}) or {}
        tok = tlen(txt)
        max_tokens = max(max_tokens, tok)
        rows.append({
            "index": i,
            "tokens": tok,
            "chars": len(txt),
            "types": meta.get("types"),
            "pages": (meta.get("page_start"), meta.get("page_end")),
            "table_index": meta.get("table_index"),
        })

    print("Chunk lengths (tokens / chars):\n")
    for r in rows:
        bar_len = max(1, int(30 * r["tokens"] / max_tokens))
        pages = f"{r['pages'][0]}–{r['pages'][1]}" if r["pages"][0] is not None else "?"
        tinfo = f"(Table #{r['table_index']})" if r.get("table_index") else ""
        over = "  [>HARD_MAX]" if r["tokens"] > HARD_MAX else ""
        print(f"[Chunk {r['index']:>3}] {r['tokens']:>4} tok / {r['chars']:>5} ch  "
              f"types={r['types']}  pages={pages} {tinfo}{over}\n"
              f"{'#' * bar_len}\n")

    toks = [r["tokens"] for r in rows]
    if toks:
        print(f"Total chunks: {len(rows)} | tokens: min={min(toks)}, max={max(toks)}, avg={sum(toks)//len(toks)}")
    return rows
# ===================== /Token-aware Docling chunker =====================
