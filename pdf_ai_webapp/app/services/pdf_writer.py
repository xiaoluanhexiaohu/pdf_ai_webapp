from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import fitz


def add_images_to_pdf(
    src_pdf_path: str | Path,
    output_pdf_path: str | Path,
    placement_instructions: List[Dict],
    appendix_title: str = "AI 自动补充图片页",
) -> None:
    doc = fitz.open(src_pdf_path)
    try:
        for item in placement_instructions:
            image_path = item["image_path"]
            mode = item["mode"]
            if mode == "appendix_page":
                appendix_page = doc.new_page(width=item["page_width"], height=item["page_height"])
                appendix_page.insert_text(
                    fitz.Point(36, 36),
                    appendix_title,
                    fontsize=16,
                )
                appendix_page.insert_text(
                    fitz.Point(36, 58),
                    f"关联页码：第 {item['page_number']} 页 | 锚点：{item['anchor_text']}",
                    fontsize=10,
                )
                rect = fitz.Rect(36, 78, item["page_width"] - 36, item["page_height"] - 36)
                appendix_page.insert_image(rect, filename=str(image_path), keep_proportion=True)
                continue

            page = doc.load_page(item["page_number"] - 1)
            rect = fitz.Rect(*item["draw_rect"])
            page.insert_image(rect, filename=str(image_path), keep_proportion=True)
        doc.save(str(output_pdf_path))
    finally:
        doc.close()
