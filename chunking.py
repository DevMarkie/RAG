"""Utility functions for chunking and extracting structured text from documents."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:  # PDF dependency is optional at import time; raise when used if missing.
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace (including non-breaking space) into single spaces."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str, preserve_newlines: bool = False) -> List[Tuple[int, str]]:
    """Read a PDF file and return tuples of (page_number, text)."""
    if PdfReader is None:
        raise RuntimeError("Missing dependency 'pypdf'. Install requirements.txt.")
    reader = PdfReader(pdf_path)
    pages: List[Tuple[int, str]] = []
    for index, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        if preserve_newlines:
            text = raw_text.replace("\u00a0", " ").strip()
        else:
            text = normalize_whitespace(raw_text)
        pages.append((index + 1, text))
    return pages


def extract_text_file_pages(txt_path: str, preserve_newlines: bool = False) -> List[Tuple[int, str]]:
    """Treat a text file as pseudo-pages by splitting on form-feed markers when present."""
    with open(txt_path, "r", encoding="utf-8") as handle:
        content = handle.read()
    segments = re.split(r"\f+", content)
    pages: List[Tuple[int, str]] = []
    for index, segment in enumerate(segments):
        output = segment.strip() if preserve_newlines else normalize_whitespace(segment)
        pages.append((index + 1, output))
    return pages


def extract_document_pages(path: str, preserve_newlines: bool = False) -> List[Tuple[int, str]]:
    """Dispatch extraction based on extension (.pdf or .txt)."""
    lower = path.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_pages(path, preserve_newlines=preserve_newlines)
    if lower.endswith(".txt"):
        return extract_text_file_pages(path, preserve_newlines=preserve_newlines)
    raise ValueError(f"Unsupported file extension for extraction: {path}")


def chunk_by_chars(text: str, max_chars: int = 1000, overlap_chars: int = 100) -> List[str]:
    """Split text by character count while keeping a configurable overlap."""
    if max_chars <= 0:
        return [text]
    overlap = max(0, min(overlap_chars, max_chars - 1))
    chunks: List[str] = []
    start = 0
    total_length = len(text)
    while start < total_length:
        end = min(start + max_chars, total_length)
        if end < total_length:
            whitespace_idx = text.rfind(" ", start, end)
            if whitespace_idx != -1 and whitespace_idx > start + int(max_chars * 0.6):
                end = whitespace_idx
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == total_length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


def build_chunks_from_pdf(pdf_path: str, max_chars: int, overlap_chars: int) -> List[Dict[str, Any]]:
    """Return structured chunks enriched with optional page markers."""
    pages = extract_document_pages(pdf_path, preserve_newlines=True)
    fragments: List[str] = []
    for page_number, text in pages:
        fragments.append(f"\n[PAGE:{page_number}]\n")
        fragments.append(text)
    full_text = "".join(fragments)
    raw_chunks = chunk_by_chars(full_text, max_chars=max_chars, overlap_chars=overlap_chars)
    results: List[Dict[str, Any]] = []
    for index, chunk in enumerate(raw_chunks):
        match = re.search(r"\[PAGE:(\d+)\]", chunk)
        page = int(match.group(1)) if match else None
        clean_text = re.sub(r"\[PAGE:\d+\]", "", chunk).strip()
        results.append({"id": f"c{index}", "page": page, "text": clean_text})
    return results


def extract_article_text(pdf_path: str, number: int) -> Optional[str]:
    """Extract the text for the requested Điều, keeping the raw formatting."""
    pages = extract_document_pages(pdf_path, preserve_newlines=True)
    combined: List[str] = []
    for page_number, text in pages:
        combined.append(f"\n[PAGE:{page_number}]\n")
        combined.append(text)
        combined.append("\n")
    full_text = "".join(combined)
    pattern = re.compile(rf"(?is)(\bĐiều\s+{number}\s*[\.:]?\s*.*?)(?=\bĐiều\s+\d+\b|\Z)")
    match = pattern.search(full_text)
    if not match:
        return None
    segment = match.group(1)
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
