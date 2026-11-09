from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class InputRecord(APIModel):
    model_config = ConfigDict(extra="allow")


class PositionReport(InputRecord):
    observation_id: str = Field(min_length=1)
    tracklet_id: str = Field(min_length=1)

    observed_at: datetime
    received_at: datetime

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    altitude_ft: float | None = None

    source: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)


class SupportingRecord(InputRecord):
    record_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)

    received_at: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class LabelRecord(InputRecord):
    tracklet_id: str = Field(min_length=1)
    successor_tracklet_id: str | None = None


class SupportingMeasurements(APIModel):
    gap_distance_meters: float
    gap_time_seconds: float
    required_speed_mps: float


class RankedCandidate(APIModel):
    successor_tracklet_id: str
    score: float
    supporting_measurements: SupportingMeasurements


class TrackletAssociation(APIModel):
    tracklet_id: str
    status: Literal["associated", "unresolved"]
    accepted_successor_id: str | None
    ranked_candidates: list[RankedCandidate]


class EvaluationMetrics(APIModel):
    auc_roc: float
    gini: float
    precision: float
    recall: float
    accuracy: float

    total_events: int
    event_rate: float

    capture_rate_10: float
    capture_rate_25: float
    capture_rate_40: float
    capture_rate_50: float


class EvaluationSummary(APIModel):
    baseline: EvaluationMetrics
    learned: EvaluationMetrics


class AssociationResponse(APIModel):
    threshold: float
    baseline: list[TrackletAssociation]
    learned: list[TrackletAssociation]
    evaluation: EvaluationSummary | None = None


class HealthResponse(APIModel):
    status: Literal["ok"]
    model_loaded: bool
    