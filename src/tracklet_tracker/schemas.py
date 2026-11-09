
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PositionReport(APIModel):
    observation_id: str = Field(min_length=1)
    tracklet_id: str = Field(min_length=1)

    observed_at: datetime
    received_at: datetime

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    source: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)


class SupportingRecord(APIModel):
    record_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)

    received_at: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class AssociationRequest(APIModel):
    position_reports: list[PositionReport] = Field(min_length=1)
    supporting_records: list[SupportingRecord] = Field(default_factory=list)

    threshold: float = Field(default=0.90, ge=0.0, le=1.0)


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


class AssociationResponse(APIModel):
    threshold: float
    baseline: list[TrackletAssociation]
    learned: list[TrackletAssociation]


class HealthResponse(APIModel):
    status: Literal["ok"]
    model_loaded: bool
