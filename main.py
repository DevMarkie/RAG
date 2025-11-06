import argparse
import os
import re
import sys
import shutil
import textwrap
import json
from typing import List, Optional

from chunking import extract_article_text, build_chunks_from_pdf  # local module


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
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



def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)

	if args.cmd == "ask":
		pdf = args.pdf
		if not os.path.exists(pdf):
			# Fallback: if user actually has a .pdf.txt (OCR exported) try that
			alt_txt = None
			if pdf.lower().endswith(".pdf") and os.path.exists(pdf + ".txt"):
				alt_txt = pdf + ".txt"
			elif os.path.exists(pdf + ".txt"):
				alt_txt = pdf + ".txt"
			if alt_txt:
				print(f"[Info] Dùng file văn bản thay thế: {alt_txt}")
				pdf = alt_txt
			else:
				if pdf.lower().endswith(".txt"):
					print(f"Không tìm thấy file văn bản: {pdf}", file=sys.stderr)
				else:
					print(f"Không tìm thấy PDF: {pdf}", file=sys.stderr)
				return 2
		q = args.question
		m = re.search(r"(?i)\bĐiều\s+(\d+)\b", q)
		if not m:
			print("Không nhận diện được số Điều trong câu hỏi.", file=sys.stderr)
			return 0
		num = int(m.group(1))
		seg = extract_article_text(pdf, num)
		if not seg:
			if getattr(args, "json", False):
				print(json.dumps({"article_number": num, "found": False, "text": None}, ensure_ascii=False))
			else:
				print(f"Không tìm thấy Điều {num}.")
			return 0
		if getattr(args, "json", False):
			print(json.dumps({
				"article_number": num,
				"found": True,
				"text": seg,
				"length": len(seg),
				"line_count": seg.count("\n") + 1
			}, ensure_ascii=False))
			return 0
		if args.no_wrap:
			print(seg)
			return 0
		# xác định độ rộng terminal
		if args.width > 0:
			wrap_width = args.width
		else:
			try:
				wrap_width = shutil.get_terminal_size((100, 20)).columns
			except Exception:
				wrap_width = 100
		wrap_width = max(40, min(wrap_width, 200))
		wrapped_blocks: List[str] = []
		for block in seg.split("\n\n"):
			para_lines: List[str] = []
			for line in block.splitlines():
				if not line.strip():
					para_lines.append("")
					continue
				leading_ws = len(line) - len(line.lstrip())
				indent = " " * leading_ws
				fill_width = max(10, wrap_width - leading_ws)
				wrapped = textwrap.fill(line.strip(), width=fill_width, subsequent_indent=indent)
				if not wrapped.startswith(indent):
					wrapped = indent + wrapped
				para_lines.append(wrapped)
			wrapped_blocks.append("\n".join(para_lines))
		print("\n\n".join(wrapped_blocks))
		return 0

	if args.cmd == "embed":
		pdf = args.pdf
		if not os.path.exists(pdf):
			alt_txt = None
			if pdf.lower().endswith(".pdf") and os.path.exists(pdf + ".txt"):
				alt_txt = pdf + ".txt"
			elif os.path.exists(pdf + ".txt"):
				alt_txt = pdf + ".txt"
			if alt_txt:
				print(f"[Info] Using fallback text file: {alt_txt}")
				pdf = alt_txt
			else:
				print(f"File not found: {pdf}", file=sys.stderr)
				return 2
		gemini_key = (
			os.environ.get("GEMINI_API_KEY")
			or os.environ.get("GOOGLE_API_KEY")
			or os.environ.get("GENAI_API_KEY")
		)
		if not gemini_key:
			print("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
			return 3
		# Lazy import to avoid hard dependency when only chunking/asking
		try:
			from embedding import embed_and_save  # local import
		except Exception as e:
			print(f"Embedding module not available: {e}", file=sys.stderr)
			return 4
		try:
			embed_and_save(
				pdf_path=pdf,
				output_path=args.out,
				model=args.model,
				gemini_key=gemini_key,
				max_chars=args.max_chars,
				overlap_chars=args.overlap_chars,
				batch_size=args.batch_size,
			)
		except Exception as e:
			print(f"Embedding failed: {e}", file=sys.stderr)
			return 4
		print(f"Saved embeddings to {args.out}")
		return 0

	if args.cmd == "chunk":
		pdf = args.pdf
		if not os.path.exists(pdf):
			alt_txt = None
			if pdf.lower().endswith(".pdf") and os.path.exists(pdf + ".txt"):
				alt_txt = pdf + ".txt"
			elif os.path.exists(pdf + ".txt"):
				alt_txt = pdf + ".txt"
			if alt_txt:
				print(f"[Info] Using fallback text file: {alt_txt}")
				pdf = alt_txt
			else:
				print(f"File not found: {pdf}", file=sys.stderr)
				return 2
		try:
			chunks = build_chunks_from_pdf(
				pdf_path=pdf,
				max_chars=args.max_chars,
				overlap_chars=args.overlap_chars,
			)
		except Exception as e:
			print(f"Chunking failed: {e}", file=sys.stderr)
			return 4
		src = os.path.basename(pdf)
		written = 0
		with open(args.out, "w", encoding="utf-8") as f:
			for c in chunks:
				record = {
					"id": c["id"],
					"page": c.get("page"),
					"text": c["text"],
					"source": src,
				}
				f.write(json.dumps(record, ensure_ascii=False) + "\n")
				written += 1
		print(f"Saved {written} chunks to {args.out}")
		return 0

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

