from __future__ import annotations

import io
import os
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract

try:
    from pdf2image import convert_from_bytes
    _PDF_OK = True
except Exception:
    _PDF_OK = False


app = FastAPI(title="OCR Service")


def _ocr_image(img: Image.Image, lang: str = "eng") -> str:
    return pytesseract.image_to_string(img, lang=lang)


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...), lang: str = "eng") -> JSONResponse:
    try:
        content = await file.read()
        filename = (file.filename or "").lower()
        texts: List[str] = []
        if filename.endswith(".pdf"):
            if not _PDF_OK:
                raise HTTPException(status_code=500, detail="pdf2image not available")
            images = convert_from_bytes(content)
            for img in images:
                texts.append(_ocr_image(img, lang=lang))
        else:
            # try image first; if fails, treat as text
            try:
                img = Image.open(io.BytesIO(content))
                texts.append(_ocr_image(img, lang=lang))
            except Exception:
                try:
                    texts.append(content.decode("utf-8", errors="ignore"))
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Unsupported file: {exc}")

        joined = "\n".join(texts)
        return JSONResponse({"text": joined})
    finally:
        await file.close()


