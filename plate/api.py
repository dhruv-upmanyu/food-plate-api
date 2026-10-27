"""HTTP API for the meal-photo analyzer: upload a photo, get every dish with calories, protein, carbs, fat, fibre.

    uvicorn plate.api:app --host 0.0.0.0 --port 8000          # interactive docs at http://localhost:8000/docs
    curl -F "image=@thali.jpg" http://localhost:8000/v1/analyze

Endpoints
  GET  /health        liveness / readiness (200 once the models are loaded)
  GET  /v1/dishes     every dish with its standard serving and nutrition (per 100 g and per serving)
  POST /v1/analyze    multipart form field "image" (JPEG / PNG / WebP / HEIC) -> items + meal totals

Environment variables (all optional)
  PLATE_API_KEY         if set, /v1/* requests must send this value in the X-API-Key header
  PLATE_CORS_ORIGINS    comma-separated browser origins allowed to call the API ("*" = any); default none
  PLATE_MAX_UPLOAD_MB   largest accepted upload in MB, default 15
  PLATE_DET_CONF        detector confidence threshold, default 0.25 (chosen on validation photos)
  PLATE_TORCH_THREADS   CPU threads per worker, default = CPUs available to the process (with N workers on one
                        machine use CPUs / N; in a container limited with --cpus use that number)
Uploaded photos are processed in memory and never written to disk or logged.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Ultralytics otherwise checks for internet, sends anonymous usage events and pip-installs missing packages at
# run time (e.g. pi-heif the first time an upload is not a recognised image); a server should do none of these.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from plate.pipeline import CLS_DIR, DET_PATH, FIELDS, NUTRITION, PlateAnalyzer, load_image  # noqa: E402

API_VERSION = "1.0.0"
MAX_PIXELS = 50_000_000   # larger photos are rejected before decoding (decompression-bomb guard)

try:                      # iPhone HEIC photos, if pi-heif is installed (optional, see requirements-serve.txt)
    from pi_heif import register_heif_opener
    register_heif_opener()
    HEIC = True
except ImportError:
    HEIC = False


class Nutrients(BaseModel):
    kcal: float
    protein_g: float
    carb_g: float = Field(description="available carbohydrate (fibre not included)")
    fat_g: float
    fibre_g: float


class Alternative(BaseModel):
    name: str
    confidence: float


class DishItem(BaseModel):
    dish: str = Field(description="stable dish key, see /v1/dishes")
    name: str
    confidence: float = Field(description="classifier probability of this dish, 0-1")
    unsure: bool = Field(description="true when confidence < 0.4: show the alternatives or ask the user")
    alternatives: list[Alternative] = Field(description="next most likely dishes")
    count: float = Field(description="pieces for countable dishes (roti, idli, egg, ...), servings otherwise")
    unit: str = Field(description="what one count is, e.g. '1 medium roti'")
    grams: float = Field(description="count x standard serving weight; portions are not measured from the photo")
    kcal: float
    protein_g: float
    carb_g: float = Field(description="available carbohydrate (fibre not included)")
    fat_g: float
    fibre_g: float
    boxes: list[list[float]] = Field(description="[x1, y1, x2, y2] as fractions (0-1) of the upright image, "
                                                 "one per piece / bowl")


class Totals(Nutrients):
    grams: float


class AnalyzeResponse(BaseModel):
    items: list[DishItem] = Field(description="one entry per dish found, highest calories first")
    totals: Totals
    image_width: int = Field(description="after applying the photo's EXIF rotation")
    image_height: int
    processing_ms: int
    model_version: str


class DishInfo(BaseModel):
    dish: str
    name: str
    group: str
    recognised: bool = Field(description="false: in the nutrition table but not (yet) recognised from photos")
    serving: str
    serving_g: float
    countable: bool
    per_100g: Nutrients
    per_serving: Nutrients
    source: str


class Health(BaseModel):
    status: str
    api_version: str
    model_version: str
    dishes_recognised: int
    cpu_threads: int = Field(description="torch threads a request runs with (PLATE_TORCH_THREADS)")


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))   # Linux: respects CPU pinning (not --cpus quotas)
    except AttributeError:
        return os.cpu_count() or 1


def model_version(det_path: Path = DET_PATH, cls_dir: Path = CLS_DIR) -> str:
    """Short fingerprint of the model files, so every response says exactly which models produced it."""
    h = hashlib.sha256()
    for p in (det_path, cls_dir / "head.pt", cls_dir / "meta.json"):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def dish_table(labels: list[str]) -> list[DishInfo]:
    out = []
    for r in pd.read_csv(NUTRITION).to_dict("records"):
        per100 = {f: round(float(r[f]), 1) for f in FIELDS}
        out.append(DishInfo(dish=r["key"], name=r["name"], group=r["group"], recognised=r["key"] in labels,
                            serving=r["unit"], serving_g=float(r["unit_g"]), countable=bool(r["countable"]),
                            per_100g=Nutrients(**per100),
                            per_serving=Nutrients(**{f: round(float(r[f]) * r["unit_g"] / 100, 1) for f in FIELDS}),
                            source=r["source"]))
    return out


def create_app(analyzer: PlateAnalyzer | None = None, api_key: str | None = None,
               cors_origins: list[str] | None = None, max_upload_mb: float | None = None) -> FastAPI:
    """analyzer: pass one to reuse already-loaded models (tests); otherwise they load at startup."""
    api_key = api_key if api_key is not None else os.environ.get("PLATE_API_KEY") or None
    if cors_origins is None:
        cors_origins = [o.strip() for o in os.environ.get("PLATE_CORS_ORIGINS", "").split(",") if o.strip()]
    max_bytes = int((max_upload_mb or float(os.environ.get("PLATE_MAX_UPLOAD_MB", 15))) * 1024 * 1024)
    lock = threading.Lock()   # one inference at a time per process; run more workers to serve more in parallel
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        an = analyzer or PlateAnalyzer(det_conf=float(os.environ.get("PLATE_DET_CONF", 0.25)))
        # The first prediction sets up Ultralytics' predictor, which resets torch to min(8, CPUs - 1) threads
        # (1 thread on a 2-CPU server, half the speed). Warm up first, then choose the thread count.
        an.analyze(Image.new("RGB", (640, 480), (128, 128, 128)))
        torch.set_num_threads(int(os.environ.get("PLATE_TORCH_THREADS") or available_cpus()))
        state["analyzer"] = an
        state["version"] = model_version()
        state["dishes"] = dish_table(an.labels)
        yield
        state.clear()

    app = FastAPI(title="Meal nutrition from photo", version=API_VERSION, lifespan=lifespan,
                  description="Finds every dish on a meal photo and returns calories, protein, carbohydrate, fat and "
                              "fibre per dish and for the meal. Portions are standard household servings.")
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["*"])

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and not hmac.compare_digest(x_api_key or "", api_key):
            raise HTTPException(401, "missing or wrong X-API-Key header")

    @app.get("/health", response_model=Health)
    def health():
        # read in a request thread: torch's thread count is per thread, set from the global value on first use
        return Health(status="ok", api_version=API_VERSION, model_version=state["version"],
                      dishes_recognised=len(state["analyzer"].labels), cpu_threads=torch.get_num_threads())

    @app.get("/v1/dishes", response_model=list[DishInfo], dependencies=[Depends(check_key)])
    def dishes():
        return state["dishes"]

    @app.post("/v1/analyze", response_model=AnalyzeResponse, dependencies=[Depends(check_key)])
    def analyze(image: UploadFile = File(..., description="meal photo: JPEG, PNG, WebP (HEIC if pi-heif is installed)")):
        data = image.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"photo is larger than {max_bytes // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(400, "empty upload")
        try:
            raw = Image.open(io.BytesIO(data))
            if raw.width * raw.height > MAX_PIXELS:
                raise HTTPException(413, f"photo has more than {MAX_PIXELS // 1_000_000} megapixels")
            img = load_image(raw)
        # ImportError: in case Ultralytics' Image.open patch is active without pi-heif (see restore_pil_open)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError, ValueError, ImportError):
            raise HTTPException(400, f"not a readable image (send a JPEG, PNG{', HEIC' if HEIC else ''} or WebP photo)")
        start = time.perf_counter()
        with lock:
            meal = state["analyzer"].analyze(img)
        ms = round((time.perf_counter() - start) * 1000)
        items = [DishItem(**{**vars(it), "alternatives": [Alternative(name=n, confidence=c) for n, c in it.alternatives]})
                 for it in meal.items]
        return AnalyzeResponse(items=items, totals=Totals(**meal.totals), image_width=img.width,
                               image_height=img.height, processing_ms=ms, model_version=state["version"])

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": app.title, "docs": "/docs", "health": "/health"}

    return app


app = create_app()
