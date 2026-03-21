from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import BASE_DIR, get_settings
from app.schemas.placement import ImageAsset, ProcessResult
from app.services.ai_matcher import AIMatcher
from app.services.docx_service import add_images_to_docx, find_anchor_candidates_docx
from app.services.layout_engine import LayoutConfig, LayoutEngine
from app.services.pdf_parser import find_anchor_candidates, get_page_occupied_rects, render_page_to_image
from app.services.pdf_writer import add_images_to_pdf
from app.services.training_store import TrainingStore, parse_training_rules
from app.utils.file_utils import ensure_dir, get_image_size, make_job_id, save_upload_file


def parse_anchors(raw_text: str) -> List[str]:
    parts = [line.strip() for line in raw_text.replace("；", "\n").replace(";", "\n").splitlines()]
    return [p for p in parts if p]


def _build_image_assets(image_uploads, job_upload_dir: Path) -> List[Dict[str, Any]]:
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
    return image_assets


def process_document_project(document_upload, image_uploads, anchors_text: str, training_rules_text: str = "") -> ProcessResult:
    settings = get_settings()
    job_id = make_job_id()

    job_upload_dir = ensure_dir(BASE_DIR / settings.upload_dir / job_id)
    job_output_dir = ensure_dir(BASE_DIR / settings.output_dir / job_id)
    job_temp_dir = ensure_dir(BASE_DIR / settings.temp_dir / job_id)

    source_path = save_upload_file(document_upload, job_upload_dir / document_upload.filename)
    source_suffix = Path(document_upload.filename).suffix.lower()

    image_assets = _build_image_assets(image_uploads, job_upload_dir)

    anchors = parse_anchors(anchors_text)
    if not anchors:
        raise ValueError("至少要提供一个锚点文字。")

    training_store = TrainingStore(BASE_DIR / settings.output_dir / "training_rules.json")
    training_rules = parse_training_rules(training_rules_text)
    added_rules = training_store.append_rules(training_rules)
    all_training_rules = training_store.load_rules()

    if source_suffix == ".pdf":
        result = _process_pdf(source_path, image_assets, anchors, all_training_rules, job_temp_dir, job_output_dir, job_id)
    elif source_suffix == ".docx":
        result = _process_docx(source_path, image_assets, anchors, all_training_rules, job_output_dir, job_id)
    else:
        raise ValueError("仅支持 PDF 或 Word(.docx) 文件。")

    if added_rules:
        result.notes.append(f"已写入 {added_rules} 条训练规则，后续任务将自动复用。")

    return result


def _process_pdf(
    pdf_path: Path,
    image_assets: List[Dict[str, Any]],
    anchors: List[str],
    training_rules: List[Dict[str, Any]],
    job_temp_dir: Path,
    job_output_dir: Path,
    job_id: str,
) -> ProcessResult:
    anchor_candidates = find_anchor_candidates(pdf_path, anchors)
    if not anchor_candidates:
        raise ValueError("在 PDF 中没有找到任何锚点文字，请检查文字是否和文档中完全一致。")

    page_preview_map: Dict[int, str] = {}
    pages_to_render = sorted({item["page_number"] for item in anchor_candidates})[:8]
    for page_number in pages_to_render:
        preview_path = job_temp_dir / f"page_{page_number}.png"
        render_page_to_image(pdf_path, page_number, preview_path)
        page_preview_map[page_number] = str(preview_path)

    settings = get_settings()
    matcher = AIMatcher(api_key=settings.openai_api_key, model=settings.openai_model, training_rules=training_rules)
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

    output_name = f"result_{job_id}.pdf"
    output_path = job_output_dir / output_name
    add_images_to_pdf(pdf_path, output_path, writer_payload)

    _save_summary(job_output_dir, job_id, str(pdf_path), image_assets, anchor_candidates, placements, notes, str(output_path))

    return ProcessResult(
        job_id=job_id,
        output_file_name=output_name,
        output_file_path=str(output_path),
        output_type="pdf",
        placements=placements,
        notes=notes,
    )


def _process_docx(
    docx_path: Path,
    image_assets: List[Dict[str, Any]],
    anchors: List[str],
    training_rules: List[Dict[str, Any]],
    job_output_dir: Path,
    job_id: str,
) -> ProcessResult:
    anchor_candidates = find_anchor_candidates_docx(docx_path, anchors)
    if not anchor_candidates:
        raise ValueError("在 Word 中没有找到任何锚点文字，请检查文字是否和文档中完全一致。")

    settings = get_settings()
    matcher = AIMatcher(api_key=settings.openai_api_key, model=settings.openai_model, training_rules=training_rules)
    placements = matcher.match(image_assets, anchor_candidates, {})

    anchor_map = {item["anchor_id"]: item for item in anchor_candidates}
    writer_payload: List[Dict[str, Any]] = []
    notes: List[str] = []

    for placement in placements:
        anchor = anchor_map.get(placement.anchor_id)
        image_asset = next((img for img in image_assets if img["image_id"] == placement.image_id), None)
        if not anchor or not image_asset:
            continue

        if placement.mode != "below":
            notes.append(f"Word 当前只实现段落插图，{image_asset['filename']} 的模式从 {placement.mode} 调整为 below。")

        writer_payload.append(
            {
                "anchor_text": anchor["anchor_text"],
                "mode": "below",
                "image_path": image_asset["path"],
            }
        )

    output_name = f"result_{job_id}.docx"
    output_path = job_output_dir / output_name
    add_images_to_docx(docx_path, output_path, writer_payload)

    _save_summary(job_output_dir, job_id, str(docx_path), image_assets, anchor_candidates, placements, notes, str(output_path))

    return ProcessResult(
        job_id=job_id,
        output_file_name=output_name,
        output_file_path=str(output_path),
        output_type="docx",
        placements=placements,
        notes=notes,
    )


def _save_summary(
    job_output_dir: Path,
    job_id: str,
    source_path: str,
    image_assets: List[Dict[str, Any]],
    anchor_candidates: List[Dict[str, Any]],
    placements,
    notes: List[str],
    output_file: str,
):
    summary_data = {
        "job_id": job_id,
        "source_path": source_path,
        "image_assets": image_assets,
        "anchor_candidates": anchor_candidates,
        "placements": [item.model_dump() for item in placements],
        "notes": notes,
        "output_file": output_file,
    }
    with (job_output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
