import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .models.autocomplete import load_autocomplete_model, run_autocomplete
from .models.classification import load_classification_model, run_classification
from .models.translation import load_translation_model, run_translation
from .schemas import (
    AutocompleteRequest,
    AutocompleteResponse,
    ClassificationRequest,
    ClassificationResponse,
    TranslationRequest,
    TranslationResponse,
)


model_store = {}


def _ensure_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in [
        config.AUTOCOMPLETE_MODEL,
        config.AUTOCOMPLETE_PREPROCESSING,
        config.TRANSLATION_MODEL,
        config.TRANSLATION_PREPROCESSING,
        config.CLASSIFICATION_MODEL,
        config.CLASSIFICATION_PREPROCESSING,
    ]:
        _ensure_file(path)

    model_store["autocomplete"] = load_autocomplete_model(
        config.AUTOCOMPLETE_MODEL,
        config.AUTOCOMPLETE_PREPROCESSING,
    )
    model_store["translation"] = load_translation_model(
        config.TRANSLATION_MODEL,
        config.TRANSLATION_PREPROCESSING,
    )
    model_store["classification"] = load_classification_model(
        config.CLASSIFICATION_MODEL,
        config.CLASSIFICATION_PREPROCESSING,
    )
    yield
    model_store.clear()


app = FastAPI(
    title="Deep Learning Suite API",
    description="FastAPI backend for autocomplete, translation, and news classification LSTM models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(model_store.keys())}


@app.post("/api/autocomplete", response_model=AutocompleteResponse)
def autocomplete(req: AutocompleteRequest):
    try:
        return run_autocomplete(
            model_store["autocomplete"],
            req.input,
            req.method,
            req.length,
            req.temperature,
            req.beam_width,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
    try:
        return run_translation(model_store["translation"], req.input)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/classify", response_model=ClassificationResponse)
def classify(req: ClassificationRequest):
    try:
        return run_classification(model_store["classification"], req.input)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

