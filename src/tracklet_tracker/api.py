
import logging
import os

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from tracklet_tracker.schemas import (
    AssociationRequest,
    AssociationResponse,
    HealthResponse,
)

from tracklet_tracker.inference import run_association_inference
from tracklet_tracker.helper_functions import load_saved_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Project root: tracklet-tracker/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(
    os.getenv(
        "TRACKLET_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "xgb_model.json"),
    )
)


# Load the model only once when the application starts.
@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model = None

    try:
        app.state.model = load_saved_model(MODEL_PATH)
        logger.info("Model loaded successfully: %s", MODEL_PATH)

    except Exception:
        logger.exception("Unable to load the model artifact.")

    yield

    app.state.model = None


app = FastAPI(
    title="Tracklet Tracker API",
    description="Aircraft tracklet association using rule-based and ML models.",
    version="1.0.0",
    lifespan=lifespan,
)


# Consistent validation error response.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed.",
                "details": jsonable_encoder(exc.errors()),
            }
        },
    )


# Consistent HTTP error response.
@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    detail = exc.detail

    if isinstance(detail, dict):
        error = detail
    else:
        error = {
            "code": "http_error",
            "message": str(detail),
        }

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error},
    )


# ---------------------------------------------------------
# GET /health
# ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health(request: Request):

    model = request.app.state.model

    if model is None or not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "The required model artifact is unavailable.",
            },
        )

    return {
        "status": "ok",
        "model_loaded": True,
    }


# ---------------------------------------------------------
# POST /tracklets/associate
# ---------------------------------------------------------

@app.post(
    "/tracklets/associate",
    response_model=AssociationResponse,
)
def associate_tracklets(
    payload: AssociationRequest,
    request: Request,
):

    model = request.app.state.model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "The model is not loaded. Inference is unavailable.",
            },
        )

    try:
        results = run_association_inference(
            position_reports=[
                report.model_dump()
                for report in payload.position_reports
            ],
            supporting_records=[
                record.model_dump()
                for record in payload.supporting_records
            ],
            model=model,
            threshold=payload.threshold,
        )

        return AssociationResponse(
            threshold=payload.threshold,
            **results,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_input",
                "message": str(exc),
            },
        )

    except Exception:
        logger.exception("Tracklet association inference failed.")

        raise HTTPException(
            status_code=500,
            detail={
                "code": "inference_failed",
                "message": "Unable to complete tracklet association.",
            },
        )

