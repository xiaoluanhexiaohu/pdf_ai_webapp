from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, get_settings
from app.routers.web import router as web_router

settings = get_settings()
app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
app.include_router(web_router)
