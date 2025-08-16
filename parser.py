# parser.py
import io
from typing import Dict, List, Tuple
import os

import fitz  # PyMuPDF
from PIL import Image
import pandas as pd

try:
    import camelot  # type: ignore
except Exception:
    camelot = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None


def _maybe_configure_tesseract_from_env():
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and pytesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = cmd


def extract_pdf(
    file_path: str, ocr_on_fail: bool = True
) -> Tuple[
    Dict[int, str], Dict[int, List[Image.Image]], Dict[int, List[pd.DataFrame]]
]:
    """Return (page_texts, images, tables) indexed by 1-based page number."""
    _maybe_configure_tesseract_from_env()

    doc = fitz.open(file_path)
    page_texts: Dict[int, str] = {}
    images: Dict[int, List[Image.Image]] = {}
    tables: Dict[int, List[pd.DataFrame]] = {}

    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        if (not text.strip()) and ocr_on_fail and (pytesseract is not None):
            try:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes(output="png"))).convert("RGB")
                text = pytesseract.image_to_string(img)
            except Exception:
                text = text or ""
        page_texts[i + 1] = text

        # Images
        page_imgs: List[Image.Image] = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                img_bytes = base.get("image")
                if img_bytes:
                    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    page_imgs.append(pil)
            except Exception:
                continue
        if page_imgs:
            images[i + 1] = page_imgs

    # Tables with Camelot (optional)
    if camelot is not None:
        try:
            cts = camelot.read_pdf(file_path, pages="all", flavor="stream")
            for t in cts:
                p = int(t.page)
                tables.setdefault(p, []).append(t.df)
        except Exception:
            pass

    return page_texts, images, tables