from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR, get_settings
from app.services.project_service import process_pdf_project

router = APIRouter()
settings = get_settings()
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "default_anchors": "现场照片：\n产品说明：\n附件图：",
        },
    )


@router.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    pdf_file: UploadFile = File(...),
    images: list[UploadFile] = File(...),
    anchors_text: str = Form(...),
):
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件。")
    try:
        result = process_pdf_project(pdf_file, images, anchors_text)
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


@router.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    file_path = BASE_DIR / settings.output_dir / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path=file_path, filename=filename, media_type="application/pdf")
