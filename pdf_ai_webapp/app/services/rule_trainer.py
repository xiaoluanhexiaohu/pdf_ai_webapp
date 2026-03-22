from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List

import fitz
from docx import Document

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

from app.config import BASE_DIR, get_settings
from app.services.training_store import TrainingStore
from app.utils.file_utils import ensure_dir, make_job_id, save_upload_file


ALLOWED_MODES = {"below", "right", "appendix_page"}


def _read_document_excerpt(doc_path: Path, max_chars: int = 5000) -> str:
    suffix = doc_path.suffix.lower()
    if suffix == ".pdf":
        parts: List[str] = []
        with fitz.open(doc_path) as doc:
            for page in doc:
                text = page.get_text("text").strip()
                if text:
                    parts.append(text)
                if sum(len(p) for p in parts) >= max_chars:
                    break
        return "\n".join(parts)[:max_chars]
    if suffix == ".docx":
        doc = Document(str(doc_path))
        text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        return text[:max_chars]
    return ""


def _keyword_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    parts = re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", stem)
    parts = [p for p in parts if len(p) >= 2]
    if not parts:
        return stem[:8] or "img"
    return parts[0]


def _normalize_rules(items: List[Dict[str, Any]], anchors: List[str]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in items:
        keyword = str(item.get("keyword", "")).strip()
        anchor_text = str(item.get("anchor_text", "")).strip()
        mode = str(item.get("mode", "below")).strip().lower()
        if not keyword or not anchor_text:
            continue
        if anchors and anchor_text not in anchors:
            continue
        if mode not in ALLOWED_MODES:
            mode = "below"
        normalized.append({"keyword": keyword, "anchor_text": anchor_text, "mode": mode})
    return normalized


def _heuristic_train_rules(images: List[Dict[str, str]], anchors: List[str]) -> List[Dict[str, str]]:
    if not anchors:
        return []
    rules: List[Dict[str, str]] = []
    for idx, image in enumerate(images):
        rules.append(
            {
                "keyword": _keyword_from_filename(image["filename"]),
                "anchor_text": anchors[idx % len(anchors)],
                "mode": "below",
            }
        )
    return rules


def _to_b64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def _train_with_openai(
    api_key: str,
    model: str,
    draft_excerpt: str,
    final_excerpt: str,
    images: List[Dict[str, str]],
    anchors: List[str],
) -> List[Dict[str, str]]:
    if not api_key or OpenAI is None:
        return []

    client = OpenAI(api_key=api_key)
    payload = {
        "task": "根据图片文件名和前后文变化，总结图片插入规则。",
        "anchors": anchors,
        "images": images,
        "draft_excerpt": draft_excerpt,
        "final_excerpt": final_excerpt,
        "output_schema": {
            "rules": [{"keyword": "string", "anchor_text": "string", "mode": "below|right|appendix_page"}]
        },
    }
    user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
    for image in images[:10]:
        user_content.append(
            {
                "type": "input_text",
                "text": f"训练图片: {image['filename']}",
            }
        )
        user_content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{_to_b64(Path(image['path']))}",
            }
        )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "你是训练规则生成器，只返回 JSON，不要额外解释。"}],
            },
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=1600,
    )
    data = _extract_json(getattr(response, "output_text", "") or "")
    return data.get("rules", [])


def _train_with_gemini(
    api_key: str,
    model: str,
    draft_excerpt: str,
    final_excerpt: str,
    images: List[Dict[str, str]],
    anchors: List[str],
) -> List[Dict[str, str]]:
    if not api_key or genai is None:
        return []
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)
    payload = {
        "task": "根据图片文件名和前后文变化，总结图片插入规则。",
        "anchors": anchors,
        "images": images,
        "draft_excerpt": draft_excerpt,
        "final_excerpt": final_excerpt,
        "output_schema": {
            "rules": [{"keyword": "string", "anchor_text": "string", "mode": "below|right|appendix_page"}]
        },
    }
    contents: List[Any] = [
        "你是训练规则生成器，只返回 JSON，不要额外解释。",
        json.dumps(payload, ensure_ascii=False),
    ]
    for image in images[:10]:
        mime = mimetypes.guess_type(image["path"])[0] or "image/png"
        with open(image["path"], "rb") as f:
            contents.append({"mime_type": mime, "data": f.read()})
    response = client.generate_content(contents)
    data = _extract_json(response.text or "")
    return data.get("rules", [])


def train_rules_from_examples(
    draft_doc_upload,
    final_doc_upload,
    image_uploads,
    anchors: List[str],
    provider: str = "openai",
) -> Dict[str, Any]:
    settings = get_settings()
    job_id = make_job_id()
    train_dir = ensure_dir(BASE_DIR / settings.temp_dir / f"train_{job_id}")

    draft_path = save_upload_file(draft_doc_upload, train_dir / f"draft{Path(draft_doc_upload.filename).suffix.lower()}")
    final_path = save_upload_file(final_doc_upload, train_dir / f"final{Path(final_doc_upload.filename).suffix.lower()}")

    images: List[Dict[str, str]] = []
    for image in image_uploads:
        path = save_upload_file(image, train_dir / image.filename)
        images.append({"filename": image.filename, "path": str(path)})

    draft_excerpt = _read_document_excerpt(draft_path)
    final_excerpt = _read_document_excerpt(final_path)

    raw_rules: List[Dict[str, Any]] = []
    selected_provider = (provider or settings.ai_provider or "openai").lower()
    try:
        if selected_provider == "gemini":
            raw_rules = _train_with_gemini(settings.gemini_api_key, settings.gemini_model, draft_excerpt, final_excerpt, images, anchors)
        else:
            raw_rules = _train_with_openai(settings.openai_api_key, settings.openai_model, draft_excerpt, final_excerpt, images, anchors)
    except Exception:
        raw_rules = []

    rules = _normalize_rules(raw_rules, anchors)
    if not rules:
        rules = _heuristic_train_rules(images, anchors)

    lines = [f"{item['keyword']}|{item['anchor_text']}|{item['mode']}" for item in rules]
    rules_text = "\n".join(lines)

    store = TrainingStore(BASE_DIR / settings.output_dir / "training_rules.json")
    added_count = store.append_rules(rules)

    return {
        "rules": rules,
        "rules_text": rules_text,
        "provider": selected_provider,
        "added_count": added_count,
        "notes": [
            f"训练已完成，共生成 {len(rules)} 条规则。",
            f"已写入规则库 {added_count} 条，可直接在“训练规则”输入框复用。",
        ],
    }
