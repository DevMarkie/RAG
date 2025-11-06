
from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any, Optional

try:  # PDF dependency is optional at import time; raise when used if missing.
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

# -----------------------------
# Text utilities
# -----------------------------

def normalize_whitespace(text: str) -> str:
   
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# PDF / TXT extraction
# -----------------------------

def extract_pdf_pages(pdf_path: str, preserve_newlines: bool = False) -> List[Tuple[int, str]]:
    
    if PdfReader is None:
        raise RuntimeError("Missing dependency 'pypdf'. Install requirements.txt.")
    reader = PdfReader(pdf_path)
    pages: List[Tuple[int, str]] = []
    for i, p in enumerate(reader.pages):
        try:
            raw = p.extract_text() or ""
        except Exception:
            raw = ""
        if preserve_newlines:
            text = raw.replace("\u00a0", " ").strip()
        else:
            text = normalize_whitespace(raw)
        pages.append((i + 1, text))
    return pages

def extract_text_file_pages(txt_path: str, preserve_newlines: bool = False) -> List[Tuple[int, str]]:
    """Treat a .txt file as pseudo-pages: split on form-feed (\f) if present else single page."""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = re.split(r"\f+", content)
    pages: List[Tuple[int, str]] = []
    for i, part in enumerate(parts):
        part_out = part.strip() if preserve_newlines else normalize_whitespace(part)
        pages.append((i + 1, part_out))
    return pages

def extract_document_pages(path: str, preserve_newlines: bool = False) -> List[Tuple[int, str]]:
    """Generic extractor for .pdf or .txt"""
    lower = path.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_pages(path, preserve_newlines=preserve_newlines)
    if lower.endswith(".txt"):
        return extract_text_file_pages(path, preserve_newlines=preserve_newlines)
    raise ValueError(f"Unsupported file extension for extraction: {path}")

# -----------------------------
# Chunking
# -----------------------------

def chunk_by_chars(text: str, max_chars: int = 1000, overlap_chars: int = 100) -> List[str]:
    
    if max_chars <= 0:
        return [text]
    # Guard against pathological overlaps
    overlap = max(0, min(overlap_chars, max_chars - 1))
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # Prefer splitting on whitespace if we have a decent candidate
            ws = text.rfind(" ", start, end)
            if ws != -1 and ws > start + int(max_chars * 0.6):
                end = ws
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        # Move start forward, respecting overlap guard
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks

def build_chunks_from_pdf(pdf_path: str, max_chars: int, overlap_chars: int) -> List[Dict[str, Any]]:
    
    pages = extract_document_pages(pdf_path, preserve_newlines=True)
    parts: List[str] = []
    for page_num, text in pages:
        parts.append(f"\n[PAGE:{page_num}]\n")
        parts.append(text)
    full = "".join(parts)
    raw_chunks = chunk_by_chars(full, max_chars=max_chars, overlap_chars=overlap_chars)
    results: List[Dict[str, Any]] = []
    for i, ch in enumerate(raw_chunks):
        m = re.search(r"\[PAGE:(\d+)\]", ch)
        page = int(m.group(1)) if m else None
        clean = re.sub(r"\[PAGE:\d+\]", "", ch).strip()
        results.append({"id": f"c{i}", "page": page, "text": clean})
    return results

# -----------------------------
# Article extraction
# -----------------------------

def extract_article_text(pdf_path: str, number: int) -> Optional[str]:
    """Extract text for 'Điều <number>' until the next heading or end."""
    pages = extract_document_pages(pdf_path, preserve_newlines=True)
    combined: List[str] = []
    for page_num, text in pages:
        combined.append(f"\n[PAGE:{page_num}]\n")
        combined.append(text)
        combined.append("\n")
    full = "".join(combined)
    pattern = re.compile(rf"(?is)(\bĐiều\s+{number}\s*[\.:]?\s*.*?)(?=\bĐiều\s+\d+\b|\Z)")
    m = pattern.search(full)
    if not m:
        return None
    segment = m.group(1)
    segment = re.sub(r"\[PAGE:\d+\]", "", segment)
    segment = re.sub(r"\n{3,}", "\n\n", segment)
    return segment.strip()

__all__ = [
    "normalize_whitespace",
    "extract_pdf_pages",
    "extract_text_file_pages",
    "extract_document_pages",
    "chunk_by_chars",
    "build_chunks_from_pdf",
    "extract_article_text",
]
