from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import get_settings, BASE_DIR
from app.schemas.placement import ImageAsset, PlacementDecision, ProcessResult
from app.services.ai_matcher import AIMatcher
from app.services.layout_engine import LayoutConfig, LayoutEngine
from app.services.pdf_parser import (
    detect_text_pdf,
    find_anchor_candidates,
    get_page_occupied_rects,
    render_page_to_image,
)
from app.services.pdf_writer import add_images_to_pdf
from app.utils.file_utils import ensure_dir, get_image_size, make_job_id, save_upload_file


def parse_anchors(raw_text: str) -> List[str]:
    parts = [line.strip() for line in raw_text.replace("；", "\n").replace(";", "\n").splitlines()]
    return [p for p in parts if p]


def process_pdf_project(pdf_upload, image_uploads, anchors_text: str) -> ProcessResult:
    settings = get_settings()
    job_id = make_job_id()

    job_upload_dir = ensure_dir(BASE_DIR / settings.upload_dir / job_id)
    job_output_dir = ensure_dir(BASE_DIR / settings.output_dir / job_id)
    job_temp_dir = ensure_dir(BASE_DIR / settings.temp_dir / job_id)

    pdf_path = save_upload_file(pdf_upload, job_upload_dir / pdf_upload.filename)
    image_assets: List[Dict[str, Any]] = []
    for idx, image_upload in enumerate(image_uploads, start=1):
        image_path = save_upload_file(image_upload, job_upload_dir / image_upload.filename)
        width, height = get_image_size(image_path)
        image_assets.append(
            ImageAsset(
                image_id=f"img_{idx}",
                filename=image_upload.filename,
                path=str(image_path),
                width=width,
                height=height,
            ).model_dump()
        )

    anchors = parse_anchors(anchors_text)
    if not anchors:
        raise ValueError("至少要提供一个锚点文字。")

    anchor_candidates = find_anchor_candidates(pdf_path, anchors)
    if not anchor_candidates:
        raise ValueError("在 PDF 中没有找到任何锚点文字，请检查文字是否和 PDF 中完全一致。")

    page_preview_map: Dict[int, str] = {}
    pages_to_render = sorted({item["page_number"] for item in anchor_candidates})[:8]
    for page_number in pages_to_render:
        preview_path = job_temp_dir / f"page_{page_number}.png"
        render_page_to_image(pdf_path, page_number, preview_path)
        page_preview_map[page_number] = str(preview_path)

    matcher = AIMatcher(api_key=settings.openai_api_key, model=settings.openai_model)
    placements = matcher.match(image_assets, anchor_candidates, page_preview_map)

    layout_engine = LayoutEngine(
        LayoutConfig(
            max_image_width_ratio=settings.max_image_width_ratio,
            max_image_height_ratio=settings.max_image_height_ratio,
            gap=settings.default_insert_gap,
        )
    )

    anchor_map = {item["anchor_id"]: item for item in anchor_candidates}
    writer_payload: List[Dict[str, Any]] = []
    notes: List[str] = []

    for placement in placements:
        anchor = anchor_map.get(placement.anchor_id)
        image_asset = next((img for img in image_assets if img["image_id"] == placement.image_id), None)
        if not anchor or not image_asset:
            continue

        occupied_rects = get_page_occupied_rects(pdf_path, anchor["page_number"])
        mode, draw_rect = layout_engine.choose_rect(
            image_path=image_asset["path"],
            page_width=anchor["page_width"],
            page_height=anchor["page_height"],
            anchor_rect=anchor["anchor_rect"],
            occupied_rects=occupied_rects,
            preferred_mode=placement.mode,
        )

        writer_payload.append(
            {
                "page_number": anchor["page_number"],
                "page_width": anchor["page_width"],
                "page_height": anchor["page_height"],
                "anchor_text": anchor["anchor_text"],
                "mode": mode,
                "draw_rect": draw_rect,
                "image_path": image_asset["path"],
            }
        )

        if mode == "appendix_page":
            notes.append(f"{image_asset['filename']} 在原页放不下，已追加补充页。")
        elif mode != placement.mode:
            notes.append(f"{image_asset['filename']} 原建议为 {placement.mode}，实际因空间原因改为 {mode}。")

    output_pdf_name = f"result_{job_id}.pdf"
    output_pdf_path = job_output_dir / output_pdf_name
    add_images_to_pdf(pdf_path, output_pdf_path, writer_payload)

    summary_data = {
        "job_id": job_id,
        "pdf_path": str(pdf_path),
        "image_assets": image_assets,
        "anchor_candidates": anchor_candidates,
        "placements": [item.model_dump() for item in placements],
        "notes": notes,
        "output_pdf": str(output_pdf_path),
    }
    with (job_output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    return ProcessResult(
        job_id=job_id,
        output_pdf_name=output_pdf_name,
        output_pdf_path=str(output_pdf_path),
        placements=placements,
        notes=notes,
    )
