"""Helpers for generating embeddings with Google Gemini models."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import google.generativeai as genai
from tqdm import tqdm

from chunking import build_chunks_from_pdf


def _normalize_model_name(name: str) -> str:
    """Ensure the Gemini model name has the required "models/" prefix."""
    return name if name.startswith("models/") else f"models/{name}"


@dataclass
class GeminiEmbedder:
    """Wrapper around the Gemini embedding endpoint."""

    api_key: str
    model: str = "text-embedding-004"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        genai.configure(api_key=self.api_key)
        self.model = _normalize_model_name(self.model)

    def embed_documents(self, texts: Sequence[str], batch_size: int = 16) -> List[List[float]]:
        """Embed document chunks for retrieval."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=batch_size)

    def embed_query(self, text: str) -> List[float]:
        """Embed a user query."""
        embeddings = self._embed([text], task_type="RETRIEVAL_QUERY", batch_size=1)
        return embeddings[0]

    def _embed(self, texts: Sequence[str], task_type: str, batch_size: int = 16) -> List[List[float]]:
        outputs: List[List[float]] = []
        iterator: Iterable[str] = texts
        disable_bar = len(texts) < 5
        with tqdm(total=len(texts), disable=disable_bar, unit="chunk", desc="Embedding") as bar:
            for text in iterator:
                cleaned = text.strip()
                if not cleaned:
                    raise ValueError("Encountered empty chunk during embedding. Check chunking step.")
                response = genai.embed_content(
                    model=self.model,
                    content=cleaned,
                    task_type=task_type,
                )
                embedding = response.get("embedding") if isinstance(response, dict) else None
                if embedding is None:
                    raise RuntimeError("Gemini embedding response missing 'embedding' field")
                outputs.append(embedding)
                bar.update(1)
                # Gemini rate limits are generous, but a small sleep helps avoid spikes.
                if batch_size > 0:
                    time.sleep(0.05)
        return outputs


def embed_and_save(
    *,
    pdf_path: str,
    output_path: str,
    model: str,
    gemini_key: str,
    max_chars: int,
    overlap_chars: int,
    batch_size: int,
) -> None:
    """Chunk a document and persist embeddings to a JSONL file."""
    chunks = build_chunks_from_pdf(
        pdf_path=pdf_path,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    filtered = [c for c in chunks if c["text"].strip()]
    if not filtered:
        raise RuntimeError("No non-empty chunks produced from document")
    embedder = GeminiEmbedder(api_key=gemini_key, model=model)
    vectors = embedder.embed_documents([c["text"] for c in filtered], batch_size=batch_size)
    with open(output_path, "w", encoding="utf-8") as fp:
        for chunk, vector in zip(filtered, vectors):
            record = {
                "id": chunk["id"],
                "page": chunk.get("page"),
                "text": chunk["text"],
                "embedding": vector,
                "model": _normalize_model_name(model),
                "provider": "gemini",
                "source": os.path.basename(pdf_path),
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
