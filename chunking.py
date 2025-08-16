# chunking.py
import re
from functools import lru_cache
from typing import List, Dict

import spacy


@lru_cache(maxsize=1)
def _load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        # If model missing, caller should ensure it's downloaded.
        return spacy.blank("en")


def _split_paragraphs(text: str) -> List[str]:
    # Split on double newlines first (paragraphs)
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts if parts else [text]


def _split_lines(text: str) -> List[str]:
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts if parts else [text]


def _split_sentences(text: str) -> List[str]:
    nlp = _load_nlp()
    doc = nlp(text)
    parts = [s.text.strip() for s in doc.sents if s.text.strip()]
    return parts if parts else [text]


def _hard_wrap(text: str, max_chars: int) -> List[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def recursive_like_chunks(
    text: str, max_chars: int = 1200, overlap: int = 300
) -> List[str]:
    """Recursive-style splitter WITHOUT using LangChain's RecursiveCharacterTextSplitter.
    Strategy: paragraphs → lines → sentences → hard wrap, keeping overlaps between chunks.
    """
    if not text or not text.strip():
        return []

    def explode(block: str) -> List[str]:
        if len(block) <= max_chars:
            return [block]
        # Try paragraph split
        paras = _split_paragraphs(block)
        if len(paras) > 1:
            return _explode_list(paras)
        # Try line split
        lines = _split_lines(block)
        if len(lines) > 1:
            return _explode_list(lines)
        # Try sentence split
        sents = _split_sentences(block)
        if len(sents) > 1:
            return _explode_list(sents)
        # Fallback: hard wrap
        return _hard_wrap(block, max_chars)

    def _explode_list(items: List[str]) -> List[str]:
        out: List[str] = []
        cur = ""
        for it in items:
            if not cur:
                cur = it
                continue
            if len(cur) + 2 + len(it) <= max_chars:
                cur = cur + "\n\n" + it
            else:
                out.extend(explode(cur))
                cur = it
        if cur:
            out.extend(explode(cur))
        return out

    # First explosion
    raw_chunks = explode(text)

    # Add overlaps
    final: List[str] = []
    for i, ch in enumerate(raw_chunks):
        if i == 0:
            final.append(ch)
        else:
            tail = final[-1][-overlap:] if overlap > 0 else ""
            merged = (tail + "\n\n" + ch).strip()
            # Re-trim if overlap pushed it over a lot; we allow slight overflow
            if len(merged) > max_chars + overlap:
                merged = merged[: max_chars + overlap]
            final.append(merged)
    return final


def get_text_chunks(page_texts: Dict[int, str]) -> List[Dict]:
    """
    Splits text from all pages into chunks using the custom splitter.
    """
    all_chunks: List[Dict] = []
    for page_num, text in page_texts.items():
        if not text or not text.strip():
            continue

        chunks = recursive_like_chunks(text)

        for chunk in chunks:
            all_chunks.append({"page": page_num, "text": chunk})

    return all_chunks