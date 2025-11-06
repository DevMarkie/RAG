"""Simple RAG pipeline for the CAC_BO_LUAT document corpus."""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import uuid
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import chromadb
import google.generativeai as genai

from chunking import build_chunks_from_pdf, extract_article_text, extract_document_pages
from embedding import GeminiEmbedder

COLLECTION_DEFAULT = "cac_bo_luat"
SUPPORTED_EXTENSIONS = (".pdf", ".txt")


def resolve_api_key(explicit: str | None) -> str:
    """Return the first available Gemini API key from CLI or environment."""
    for key in (explicit, os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY"), os.environ.get("GENAI_API_KEY")):
        if key:
            return key
    raise RuntimeError("Gemini API key not found. Provide --api-key or set GEMINI_API_KEY.")


def iter_documents(data_dir: Path) -> Iterable[Path]:
    """Yield supported documents under the provided directory, depth-first."""
    for path in sorted(data_dir.glob("**/*")):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            yield path


def ingest_corpus(
    *,
    data_dir: Path,
    db_dir: Path,
    api_key: str,
    embed_model: str,
    max_chars: int,
    overlap_chars: int,
    batch_size: int,
    collection_name: str,
    reset: bool,
) -> None:
    """Chunk every supported document and upsert its embeddings into ChromaDB."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = list(iter_documents(data_dir))
    if not files:
        raise RuntimeError(f"No PDF/TXT files found under {data_dir}")

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "source": "gemini"},
    )

    embedder = GeminiEmbedder(api_key=api_key, model=embed_model)
    total_chunks = 0
    for file_path in files:
        try:
            chunks = build_chunks_from_pdf(
                pdf_path=str(file_path),
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        except Exception as exc:  # pragma: no cover - best effort ingest
            print(f"[WARN] Failed to chunk {file_path.name}: {exc}", file=sys.stderr)
            continue
        filtered = [c for c in chunks if c["text"].strip()]
        if not filtered:
            print(f"[INFO] No usable chunks for {file_path.name}")
            continue
        embeddings = embedder.embed_documents([c["text"] for c in filtered], batch_size=batch_size)
        rel_path = file_path.relative_to(data_dir)
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, object]] = []
        for idx, chunk in enumerate(filtered):
            chunk_id = f"{rel_path.stem}-{idx}-{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            meta: Dict[str, object] = {
                "source": str(rel_path),
                "chunk": idx,
            }
            page = chunk.get("page")
            if page is not None:
                meta["page"] = int(page)
            metadatas.append(meta)
        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        total_chunks += len(filtered)
        print(f"[INGEST] {file_path.name}: added {len(filtered)} chunks")
    print(f"[DONE] Ingested {total_chunks} chunks into collection '{collection_name}'")


def format_context_block(index: int, text: str, metadata: Dict[str, object]) -> str:
    """Return a readable context block label for prompt construction."""
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    page_label = f", trang {page}" if page else ""
    return f"[Đoạn {index} | {source}{page_label}]\n{text.strip()}"


def build_prompt(question: str, contexts: List[Dict[str, object]]) -> str:
    """Compose a grounded prompt with explicit instructions and retrieved contexts."""
    prompt_parts: List[str] = [
        "Bạn là trợ lý pháp lý tiếng Việt.",
        "Sử dụng nguyên văn các đoạn trích dẫn bên dưới để trả lời câu hỏi.",
        "Nếu câu hỏi yêu cầu nội dung của một Điều luật, hãy ghép đầy đủ tất cả các khoản, điểm liên quan từ những đoạn được cung cấp, giữ nguyên số thứ tự và không tóm tắt.",
        "Nếu không tìm thấy thông tin phù hợp, hãy trả lời rằng bạn chưa có dữ liệu để trả lời.",
        "Hãy trích dẫn nguồn bằng cách ghi rõ tên file (và trang nếu có).",
        "\nCác đoạn trích:",
    ]
    for index, context in enumerate(contexts, start=1):
        prompt_parts.append(format_context_block(index, context["text"], context["metadata"]))
    prompt_parts.append("\nCâu hỏi: " + question.strip())
    prompt_parts.append("\nCâu trả lời:")
    return "\n\n".join(prompt_parts)


def retrieve_context(
    *,
    collection,
    embedder: GeminiEmbedder,
    question: str,
    top_k: int,
) -> List[Dict[str, object]]:
    """Retrieve the nearest chunks for a question and narrow by source when possible."""
    query_embedding = embedder.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents: Sequence[Sequence[str]] = results.get("documents", [])  # type: ignore[assignment]
    metadatas: Sequence[Sequence[Dict[str, object]]] = results.get("metadatas", [])  # type: ignore[assignment]
    distances: Sequence[Sequence[float]] = results.get("distances", [])  # type: ignore[assignment]
    if not documents:
        return []
    contexts: List[Dict[str, object]] = []
    for doc, meta, dist in zip(documents[0], metadatas[0], distances[0]):
        if not doc:
            continue
        contexts.append({"text": doc, "metadata": meta, "distance": dist})
    if not contexts:
        return []
    primary_source = contexts[0]["metadata"].get("source") if contexts else None
    if primary_source:
        same_source = [ctx for ctx in contexts if ctx["metadata"].get("source") == primary_source]
        if same_source:
            contexts = same_source
    return contexts


def generate_answer(
    *,
    api_key: str,
    llm_model: str,
    question: str,
    contexts: List[Dict[str, object]],
) -> str:
    """Send a grounded prompt to Gemini and return the resulting answer text."""
    genai.configure(api_key=api_key)
    clean_model = llm_model.strip()
    if clean_model.startswith("models/"):
        clean_model = clean_model.split("/", 1)[1]
    legacy_map = {
        "gemini-pro": "gemini-pro-latest",
        "gemini-1.0-pro": "gemini-pro-latest",
        "gemini-1.5-pro": "gemini-pro-latest",
        "gemini-1.5-flash": "gemini-flash-latest",
        "gemini-flash": "gemini-flash-latest",
    }
    clean_model = legacy_map.get(clean_model, clean_model)
    prompt = build_prompt(question, contexts)
    model = genai.GenerativeModel(clean_model)
    response = model.generate_content(prompt)
    answer = (response.text or "").strip() if hasattr(response, "text") else ""
    if not answer:
        answer = "Xin lỗi, tôi chưa tìm được câu trả lời phù hợp từ dữ liệu."  # graceful fallback
    return answer


def display_answer(answer: str, contexts: List[Dict[str, object]], show_context: bool, suppress_sources: bool = False) -> None:
    """Print the final answer and any supporting citations."""
    print("\n=== Trả lời ===")
    print(answer)
    if suppress_sources:
        return
    if not contexts:
        print("\nNguồn: (Không có đoạn trích phù hợp)")
        return
    source_order: List[str] = []
    source_pages: Dict[str, List[int]] = {}
    source_plain: Dict[str, bool] = {}
    for ctx in contexts:
        meta = ctx["metadata"]
        source_value = str(meta.get("source", "unknown")).strip()
        if source_value not in source_order:
            source_order.append(source_value)
            source_pages[source_value] = []
            source_plain[source_value] = False
        page = meta.get("page")
        if isinstance(page, int):
            pages = source_pages[source_value]
            if page not in pages:
                pages.append(page)
        else:
            source_plain[source_value] = True
    label_list: List[str] = []
    for src in source_order:
        pages = source_pages[src]
        if pages:
            for page in sorted(pages):
                label_list.append(f"{src} (trang {page})")
        elif source_plain.get(src, False) or not pages:
            label_list.append(src)
    print("\nNguồn: " + ", ".join(label_list))
    if show_context:
        print("\nChi tiết nguồn:")
        seen_detail: set[tuple[str, int | None]] = set()
        seen_source_with_page: set[str] = set()
        for idx, ctx in enumerate(contexts, start=1):
            meta = ctx["metadata"]
            source = str(meta.get("source", "unknown")).strip()
            page_val = meta.get("page")
            page: int | None = page_val if isinstance(page_val, int) else None
            if page is None and source in seen_source_with_page:
                continue
            key = (source, page)
            if key in seen_detail:
                continue
            seen_detail.add(key)
            if page is not None:
                seen_source_with_page.add(source)
            label = f"{source} (trang {page})" if page is not None else source
            print(f"  • Đoạn {idx}: {label}")


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface for ingesting and querying the corpus."""
    parser = argparse.ArgumentParser(description="RAG đơn giản cho bộ luật trong thư mục CAC_BO_LUAT")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Chunk + embed toàn bộ tài liệu trong thư mục")
    p_ingest.add_argument("--data-dir", type=Path, default=Path("CAC_BO_LUAT"), help="Thư mục chứa dữ liệu")
    p_ingest.add_argument("--db-dir", type=Path, default=Path("chroma_store"), help="Thư mục lưu vector store")
    p_ingest.add_argument("--chunk-size", type=int, default=1200, help="Số ký tự tối đa mỗi chunk")
    p_ingest.add_argument("--chunk-overlap", type=int, default=150, help="Số ký tự overlap")
    p_ingest.add_argument("--batch-size", type=int, default=8, help="Batch size logic (tạm dùng cho sleep)")
    p_ingest.add_argument("--embed-model", type=str, default="text-embedding-004", help="Model embedding Gemini")
    p_ingest.add_argument("--collection", type=str, default=COLLECTION_DEFAULT, help="Tên collection Chroma")
    p_ingest.add_argument("--api-key", type=str, default=None, help="Gemini API key (ưu tiên). Nếu thiếu sẽ lấy từ biến môi trường")
    p_ingest.add_argument("--reset", action="store_true", help="Xoá collection cũ trước khi ingest")

    p_query = sub.add_parser("ask", help="Đặt câu hỏi về một điều luật")
    p_query.add_argument("question", type=str, help="Câu hỏi cần trả lời")
    p_query.add_argument("--db-dir", type=Path, default=Path("chroma_store"), help="Thư mục vector store")
    p_query.add_argument("--collection", type=str, default=COLLECTION_DEFAULT, help="Tên collection Chroma")
    p_query.add_argument("--top-k", type=int, default=4, help="Số đoạn trích lấy ra")
    p_query.add_argument("--embed-model", type=str, default="text-embedding-004", help="Model embedding Gemini")
    p_query.add_argument("--llm-model", type=str, default="gemini-pro-latest", help="Model sinh trả lời")
    p_query.add_argument("--api-key", type=str, default=None, help="Gemini API key (ưu tiên). Nếu thiếu sẽ lấy từ biến môi trường")
    p_query.add_argument("--show-context", action="store_true", help="Hiện luôn nội dung các đoạn trích")
    p_query.add_argument("--data-dir", type=Path, default=Path("CAC_BO_LUAT"), help="Thư mục chứa tài liệu gốc để trích nguyên văn Điều")

    return parser


def locate_source_file(source: str, search_dirs: Sequence[Path]) -> Path | None:
    """Best-effort search for a relative source path within known directories."""
    candidates: List[Path] = []
    source_name = Path(source).name
    for base in search_dirs:
        if not base:
            continue
        candidate = base / source
        if candidate.exists():
            return candidate
        candidates.extend(list(base.glob(source)))
    if candidates:
        return candidates[0]
    direct = Path(source)
    if direct.exists():
        return direct
    for base in search_dirs:
        if not base:
            continue
        hits = list(base.rglob(source_name))
        if hits:
            return hits[0]
    return None


ARTICLE_NUMBER_PATTERN = re.compile(r"(?i)\b(?:Điều|Dieu|Di.{0,1}u|DIEU)\s+(\d+)\b")


def extract_article_verbatim(path: Path, number: int) -> tuple[str, List[int]] | None:
    """Pull the raw text for a Điều directly from the source document."""
    pages = extract_document_pages(str(path), preserve_newlines=True)
    combined_parts: List[str] = []
    for page_num, text in pages:
        combined_parts.append(f"\n[PAGE:{page_num}]\n")
        combined_parts.append(text)
    full = "".join(combined_parts)
    article_matches = list(ARTICLE_NUMBER_PATTERN.finditer(full))
    start_idx: int | None = None
    end_idx: int | None = None
    for idx, match in enumerate(article_matches):
        try:
            current = int(match.group(1))
        except ValueError:
            continue
        if current == number:
            start_idx = match.start()
            end_idx = article_matches[idx + 1].start() if idx + 1 < len(article_matches) else len(full)
            break
    if start_idx is None:
        article = extract_article_text(str(path), number)
        if not article:
            return None
        return article, []
    segment = full[start_idx:end_idx]
    page_numbers: List[int] = []
    last_page_before = None
    for marker in re.finditer(r"\[PAGE:(\d+)\]", full):
        page_idx = marker.start()
        marker_page = int(marker.group(1))
        if page_idx <= start_idx:
            last_page_before = marker_page
        elif start_idx < page_idx < end_idx:
            if marker_page not in page_numbers:
                page_numbers.append(marker_page)
    if last_page_before is not None and last_page_before not in page_numbers:
        page_numbers.insert(0, last_page_before)
    for marker in re.finditer(r"\[PAGE:(\d+)\]", segment):
        marker_page = int(marker.group(1))
        if marker_page not in page_numbers:
            page_numbers.append(marker_page)
    clean = re.sub(r"\[PAGE:\d+\]", "", segment)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    clean = clean.strip()
    return clean, page_numbers


def try_direct_article_answer(
    question: str,
    contexts: List[Dict[str, object]],
    search_dirs: Sequence[Path],
) -> tuple[str, List[Dict[str, object]], bool] | None:
    """Return the full Điều text when all retrieved snippets point to the same source."""
    match = ARTICLE_NUMBER_PATTERN.search(question)
    if not match or not contexts:
        return None
    number = int(match.group(1))
    source = contexts[0]["metadata"].get("source")
    if not source or not isinstance(source, str):
        return None
    doc_path = locate_source_file(source, search_dirs)
    if not doc_path:
        return None
    extracted = extract_article_verbatim(doc_path, number)
    if not extracted:
        return None
    article_text, pages = extracted
    answer = article_text
    context_override: List[Dict[str, object]] = []
    if pages:
        for page in pages:
            context_override.append({"text": f"Điều {number}", "metadata": {"source": source, "page": page}, "distance": 0.0})
    return answer, context_override, True


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        api_key = resolve_api_key(getattr(args, "api_key", None))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.cmd == "ingest":
        try:
            ingest_corpus(
                data_dir=args.data_dir,
                db_dir=args.db_dir,
                api_key=api_key,
                embed_model=args.embed_model,
                max_chars=args.chunk_size,
                overlap_chars=args.chunk_overlap,
                batch_size=args.batch_size,
                collection_name=args.collection,
                reset=args.reset,
            )
        except Exception as exc:
            print(f"Ingest failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "ask":
        client = chromadb.PersistentClient(path=str(args.db_dir))
        try:
            collection = client.get_collection(args.collection)
        except Exception:
            print("Không tìm thấy collection. Hãy chạy lệnh ingest trước.", file=sys.stderr)
            return 1
        embedder = GeminiEmbedder(api_key=api_key, model=args.embed_model)
        contexts = retrieve_context(
            collection=collection,
            embedder=embedder,
            question=args.question,
            top_k=args.top_k,
        )
        if not contexts:
            print("Không tìm thấy đoạn trích phù hợp trong vector store.")
            return 0
        search_dirs = [args.data_dir, Path.cwd()]
        direct_article = try_direct_article_answer(args.question, contexts, search_dirs)
        if direct_article:
            answer_text, override_contexts, suppress_sources = direct_article
            display_answer(answer_text, override_contexts, args.show_context, suppress_sources=suppress_sources)
            return 0
        answer = generate_answer(
            api_key=api_key,
            llm_model=args.llm_model,
            question=args.question,
            contexts=contexts,
        )
        display_answer(answer, contexts, args.show_context)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
