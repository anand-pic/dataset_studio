from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from dataset_studio.web.routers.datasets import router as datasets_router

app = FastAPI(title="Dataset Studio", version="0.1.0")
app.include_router(datasets_router)

APP_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = APP_ROOT / "web" / "static"
TEMPLATE_PATH = APP_ROOT / "web" / "templates" / "index.html"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")
