import json
import logging
import os

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from tracklet_tracker.schemas import (
    APIModel,
    PositionReport,
    SupportingRecord,
    LabelRecord,
    AssociationResponse,
    HealthResponse,
)

from tracklet_tracker.inference import run_association_inference
from tracklet_tracker.helper_functions import load_saved_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(
    os.getenv(
        "TRACKLET_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "xgb_model.json"),
    )
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model = None

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}"
            )

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
        headers=exc.headers,
    )


def parse_jsonl(
    content: bytes,
    record_schema: type[APIModel],
    file_name: str,
) -> pd.DataFrame:

    try:
        text = content.decode("utf-8")

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_encoding",
                "message": f"{file_name} must use UTF-8 encoding.",
            },
        )

        records = []

    try:
        # First, try standard JSONL: one JSON object per line.
        try:
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]

        except json.JSONDecodeError:
            # Fallback: standard JSON array or a single JSON object.
            parsed = json.loads(text)

            if isinstance(parsed, list):
                records = parsed
            elif isinstance(parsed, dict):
                records = [parsed]
            else:
                raise ValueError("Expected a JSON object or array.")

        validated_records = [
            record_schema.model_validate(record).model_dump(mode="python")
            for record in records
        ]

    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_input",
                "message": f"Invalid data in {file_name}.",
                "details": str(exc),
            },
        )

    if not validated_records:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "empty_file",
                "message": f"{file_name} does not contain any valid records.",
            },
        )

    return pd.DataFrame(validated_records)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(request: Request):

    model = request.app.state.model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "The model is not loaded. Inference is unavailable.",
            },
        )

    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post(
    "/tracklets/associate",
    response_model=AssociationResponse,
)
async def associate_tracklets(
    request: Request,
    observations_file: UploadFile = File(...),
    supporting_file: UploadFile = File(...),
    labels_file: UploadFile | None = File(None),
    threshold: float = Form(default=0.90, ge=0.0, le=1.0),
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

    # Parse and validate the mandatory files.
    observations_df = parse_jsonl(
        await observations_file.read(),
        PositionReport,
        observations_file.filename or "observations.jsonl",
    )

    supporting_df = parse_jsonl(
        await supporting_file.read(),
        SupportingRecord,
        supporting_file.filename or "supporting.jsonl",
    )

    # Labels are optional and used only for evaluation.
    labels_df = None

    if labels_file is not None:
        labels_df = parse_jsonl(
            await labels_file.read(),
            LabelRecord,
            labels_file.filename or "labels.jsonl",
        )

    try:

        results = await run_in_threadpool(
            run_association_inference,
            observations_df=observations_df,
            supporting_df=supporting_df,
            model=model,
            threshold=threshold,
            labels_df=labels_df,
        )

        return AssociationResponse(
            threshold=threshold,
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

    