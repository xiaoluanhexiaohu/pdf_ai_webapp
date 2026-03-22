from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR, get_settings
from app.services.project_service import process_document_project
from app.services.rule_trainer import train_rules_from_examples

router = APIRouter()
settings = get_settings()
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def _build_index_context(request: Request, **extra):
    context = {
        "request": request,
        "app_name": settings.app_name,
        "default_anchors": "现场照片：\n产品说明：\n附件图：",
        "default_training_rules": "# 每行: 关键词|锚点文字|模式(可选)\nsite|现场照片：|below\nproduct|产品说明：|right",
        "default_provider": settings.ai_provider,
        "generated_training_rules": "",
        "training_notes": [],
    }
    context.update(extra)
    return context


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", _build_index_context(request))


@router.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    doc_file: UploadFile = File(...),
    images: list[UploadFile] = File(...),
    anchors_text: str = Form(...),
    training_rules_text: str = Form(""),
    provider: str = Form("openai"),
):
    suffix = Path(doc_file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="请上传 PDF 或 Word(docx) 文件。")
    try:
        result = process_document_project(doc_file, images, anchors_text, training_rules_text, provider)
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "app_name": settings.app_name,
                "result": result,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "app_name": settings.app_name,
                "error": str(exc),
            },
            status_code=400,
        )


@router.post("/train", response_class=HTMLResponse)
async def train_rules(
    request: Request,
    draft_doc_file: UploadFile = File(...),
    final_doc_file: UploadFile = File(...),
    train_images: list[UploadFile] = File(...),
    anchors_text: str = Form(...),
    provider: str = Form("openai"),
):
    draft_suffix = Path(draft_doc_file.filename).suffix.lower()
    final_suffix = Path(final_doc_file.filename).suffix.lower()
    if draft_suffix not in {".pdf", ".docx"} or final_suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="训练前后文档仅支持 PDF 或 Word(docx)。")

    anchors = [line.strip() for line in anchors_text.splitlines() if line.strip()]
    if not anchors:
        raise HTTPException(status_code=400, detail="训练时至少要输入一个锚点。")

    try:
        train_result = train_rules_from_examples(
            draft_doc_upload=draft_doc_file,
            final_doc_upload=final_doc_file,
            image_uploads=train_images,
            anchors=anchors,
            provider=provider,
        )
        return templates.TemplateResponse(
            "index.html",
            _build_index_context(
                request,
                generated_training_rules=train_result["rules_text"],
                training_notes=train_result["notes"],
                default_anchors=anchors_text,
                default_provider=provider,
            ),
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            _build_index_context(
                request,
                generated_training_rules="",
                training_notes=[f"训练失败：{exc}"],
                default_anchors=anchors_text,
                default_provider=provider,
            ),
            status_code=400,
        )


@router.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    file_path = BASE_DIR / settings.output_dir / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    media_type = "application/pdf"
    if filename.lower().endswith(".docx"):
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    return FileResponse(path=file_path, filename=filename, media_type=media_type)
