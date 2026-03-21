from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import fitz


def detect_text_pdf(pdf_path: str | Path) -> bool:
    doc = fitz.open(pdf_path)
    try:
        total_chars = 0
        pages_checked = min(3, doc.page_count)
        for page_index in range(pages_checked):
            total_chars += len(doc.load_page(page_index).get_text("text").strip())
        return total_chars > 30
    finally:
        doc.close()


def render_page_to_image(pdf_path: str | Path, page_number: int, output_path: str | Path, zoom: float = 1.5) -> str:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(str(output_path))
        return str(output_path)
    finally:
        doc.close()


def extract_page_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    blocks = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, block_no, block_type = block[:7]
        text = (text or "").strip()
        if not text:
            continue
        blocks.append(
            {
                "rect": [float(x0), float(y0), float(x1), float(y1)],
                "text": text,
                "block_no": int(block_no),
                "block_type": int(block_type),
            }
        )
    return blocks


def _context_around_anchor(blocks: List[Dict[str, Any]], anchor_rect: fitz.Rect) -> tuple[str, str]:
    before_texts: list[tuple[float, str]] = []
    after_texts: list[tuple[float, str]] = []
    for block in blocks:
        rect = fitz.Rect(*block["rect"])
        text = block["text"]
        if rect.y1 <= anchor_rect.y0 + 5:
            before_texts.append((anchor_rect.y0 - rect.y1, text))
        elif rect.y0 >= anchor_rect.y1 - 5:
            after_texts.append((rect.y0 - anchor_rect.y1, text))
    before_texts.sort(key=lambda x: x[0])
    after_texts.sort(key=lambda x: x[0])
    context_before = "\n".join(t for _, t in before_texts[:2])
    context_after = "\n".join(t for _, t in after_texts[:3])
    return context_before[:800], context_after[:1200]


def find_anchor_candidates(pdf_path: str | Path, anchors: List[str]) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    candidates: List[Dict[str, Any]] = []
    try:
        normalized_anchors = [a.strip() for a in anchors if a.strip()]
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            blocks = extract_page_blocks(page)
            page_rect = page.rect
            for anchor_text in normalized_anchors:
                matches = page.search_for(anchor_text)
                for idx, rect in enumerate(matches, start=1):
                    context_before, context_after = _context_around_anchor(blocks, rect)
                    candidates.append(
                        {
                            "anchor_id": f"p{page_index + 1}_{anchor_text}_{idx}",
                            "page_number": page_index + 1,
                            "anchor_text": anchor_text,
                            "anchor_rect": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                            "context_before": context_before,
                            "context_after": context_after,
                            "page_width": float(page_rect.width),
                            "page_height": float(page_rect.height),
                        }
                    )
        return candidates
    finally:
        doc.close()


def get_page_occupied_rects(pdf_path: str | Path, page_number: int) -> List[List[float]]:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        rects = []
        for block in extract_page_blocks(page):
            rects.append(block["rect"])
        for img_info in page.get_image_info(xrefs=True):
            bbox = img_info.get("bbox")
            if bbox:
                rects.append([float(v) for v in bbox])
        return rects
    finally:
        doc.close()
