"""Command-line helpers for quick article lookup and embedding creation."""

import argparse
import json
import os
import re
import shutil
import sys
import textwrap
from typing import List, Optional, Tuple

from chunking import build_chunks_from_pdf, extract_article_text


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	"""Configure CLI arguments and parse the provided argv list."""
	parser = argparse.ArgumentParser(description="Luật Lao Động: hỏi Điều và tạo embeddings")
	sub = parser.add_subparsers(dest="cmd", required=True)

	# ask
	p_ask = sub.add_parser("ask", help="Hỏi Điều N: in đúng nội dung Điều N")
	p_ask.add_argument("question", type=str, help="Ví dụ: 'Điều 2 là gì?'")
	p_ask.add_argument("--pdf", type=str, default="luat_lao_dong.pdf", help="Đường dẫn PDF")
	p_ask.add_argument("--width", type=int, default=0, help="Độ rộng wrap (0 = tự động theo terminal)")
	p_ask.add_argument("--no-wrap", action="store_true", help="Không wrap dòng, in thẳng")
	p_ask.add_argument("--json", action="store_true", help="Xuất JSON thay vì văn bản (không wrap)")

	# embed
	p_emb = sub.add_parser("embed", help="Chunk văn bản/PDF và sinh embeddings (Gemini)")
	p_emb.add_argument("--pdf", type=str, default="luat_lao_dong.pdf", help="Đường dẫn PDF hoặc TXT")
	p_emb.add_argument("--out", type=str, default="embeddings.jsonl", help="File JSONL đầu ra")
	p_emb.add_argument("--model", type=str, default="text-embedding-004", help="Model Gemini embedding (mặc định text-embedding-004)")
	p_emb.add_argument("--max-chars", type=int, default=1000, help="Kích thước chunk (ký tự)")
	p_emb.add_argument("--overlap-chars", type=int, default=100, help="Overlap (ký tự)")
	p_emb.add_argument("--batch-size", type=int, default=16, help="Batch size xử lý (loop nội bộ)")

	# chunk-only
	p_chunk = sub.add_parser("chunk", help="Chia nhỏ văn bản/PDF thành các chunk và lưu JSONL (không embed)")
	p_chunk.add_argument("--pdf", type=str, default="luat_lao_dong.pdf", help="Đường dẫn PDF hoặc TXT")
	p_chunk.add_argument("--out", type=str, default="chunks.jsonl", help="File JSONL đầu ra cho chunks")
	p_chunk.add_argument("--max-chars", type=int, default=1000, help="Kích thước chunk (ký tự)")
	p_chunk.add_argument("--overlap-chars", type=int, default=100, help="Overlap (ký tự)")

	return parser.parse_args(argv)


def resolve_document_path(path: str, *, fallback_notice: Optional[str] = None) -> Tuple[Optional[str], bool]:
	"""Return an existing document path, optionally printing a fallback notice."""
	if os.path.exists(path):
		return path, False
	fallback: Optional[str] = None
	if path.lower().endswith(".pdf"):
		candidate = path + ".txt"
		if os.path.exists(candidate):
			fallback = candidate
	elif not path.lower().endswith(".txt"):
		candidate = path + ".txt"
		if os.path.exists(candidate):
			fallback = candidate
	if fallback:
		if fallback_notice:
			print(fallback_notice.format(path=fallback))
		return fallback, True
	return None, False


def determine_wrap_width(requested_width: int) -> int:
	"""Clamp the desired wrap width within sensible bounds."""
	if requested_width > 0:
		return max(40, min(requested_width, 200))
	try:
		terminal_width = shutil.get_terminal_size((100, 20)).columns
	except Exception:
		terminal_width = 100
	return max(40, min(terminal_width, 200))


def wrap_text(text: str, width: int) -> str:
	"""Wrap paragraphs while preserving indentation markers."""
	wrapped_blocks: List[str] = []
	for block in text.split("\n\n"):
		paragraph_lines: List[str] = []
		for line in block.splitlines():
			if not line.strip():
				paragraph_lines.append("")
				continue
			leading_ws = len(line) - len(line.lstrip())
			indent = " " * leading_ws
			fill_width = max(10, width - leading_ws)
			wrapped = textwrap.fill(
				line.strip(),
				width=fill_width,
				subsequent_indent=indent,
			)
			if not wrapped.startswith(indent):
				wrapped = indent + wrapped
			paragraph_lines.append(wrapped)
		wrapped_blocks.append("\n".join(paragraph_lines))
	return "\n\n".join(wrapped_blocks)


def handle_ask(args: argparse.Namespace) -> int:
	"""Answer an article-specific question by extracting the raw text."""
	pdf_path, _ = resolve_document_path(
		args.pdf,
		fallback_notice="[Info] Dùng file văn bản thay thế: {path}",
	)
	if not pdf_path:
		msg = "Không tìm thấy file văn bản" if args.pdf.lower().endswith(".txt") else "Không tìm thấy PDF"
		print(f"{msg}: {args.pdf}", file=sys.stderr)
		return 2
	match = re.search(r"(?i)\\bĐiều\\s+(\\d+)\\b", args.question)
	if not match:
		print("Không nhận diện được số Điều trong câu hỏi.", file=sys.stderr)
		return 0
	number = int(match.group(1))
	segment = extract_article_text(pdf_path, number)
	if not segment:
		if getattr(args, "json", False):
			payload = {"article_number": number, "found": False, "text": None}
			print(json.dumps(payload, ensure_ascii=False))
		else:
			print(f"Không tìm thấy Điều {number}.")
		return 0
	if getattr(args, "json", False):
		payload = {
			"article_number": number,
			"found": True,
			"text": segment,
			"length": len(segment),
			"line_count": segment.count("\n") + 1,
		}
		print(json.dumps(payload, ensure_ascii=False))
		return 0
	if args.no_wrap:
		print(segment)
		return 0
	wrap_width = determine_wrap_width(args.width)
	print(wrap_text(segment, wrap_width))
	return 0


def handle_embed(args: argparse.Namespace) -> int:
	"""Chunk input document and persist embeddings using Gemini."""
	pdf_path, _ = resolve_document_path(
		args.pdf,
		fallback_notice="[Info] Using fallback text file: {path}",
	)
	if not pdf_path:
		print(f"File not found: {args.pdf}", file=sys.stderr)
		return 2
	gemini_key = (
		os.environ.get("GEMINI_API_KEY")
		or os.environ.get("GOOGLE_API_KEY")
		or os.environ.get("GENAI_API_KEY")
	)
	if not gemini_key:
		print("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
		return 3
	try:
		from embedding import embed_and_save
	except Exception as exc:  # pragma: no cover - defensive import guard
		print(f"Embedding module not available: {exc}", file=sys.stderr)
		return 4
	try:
		embed_and_save(
			pdf_path=pdf_path,
			output_path=args.out,
			model=args.model,
			gemini_key=gemini_key,
			max_chars=args.max_chars,
			overlap_chars=args.overlap_chars,
			batch_size=args.batch_size,
		)
	except Exception as exc:
		print(f"Embedding failed: {exc}", file=sys.stderr)
		return 4
	print(f"Saved embeddings to {args.out}")
	return 0


def handle_chunk(args: argparse.Namespace) -> int:
	"""Chunk the document and dump the chunks into a JSONL file."""
	pdf_path, _ = resolve_document_path(
		args.pdf,
		fallback_notice="[Info] Using fallback text file: {path}",
	)
	if not pdf_path:
		print(f"File not found: {args.pdf}", file=sys.stderr)
		return 2
	try:
		chunks = build_chunks_from_pdf(
			pdf_path=pdf_path,
			max_chars=args.max_chars,
			overlap_chars=args.overlap_chars,
		)
	except Exception as exc:
		print(f"Chunking failed: {exc}", file=sys.stderr)
		return 4
	source_label = os.path.basename(pdf_path)
	written = 0
	with open(args.out, "w", encoding="utf-8") as handle:
		for chunk in chunks:
			record = {
				"id": chunk["id"],
				"page": chunk.get("page"),
				"text": chunk["text"],
				"source": source_label,
			}
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")
			written += 1
	print(f"Saved {written} chunks to {args.out}")
	return 0



def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	if args.cmd == "ask":
		return handle_ask(args)
	if args.cmd == "embed":
		return handle_embed(args)
	if args.cmd == "chunk":
		return handle_chunk(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

