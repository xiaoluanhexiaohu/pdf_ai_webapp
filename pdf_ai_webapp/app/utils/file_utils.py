import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile
from PIL import Image


def make_job_id() -> str:
    return uuid.uuid4().hex[:12]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination


def get_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height
