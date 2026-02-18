from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisSpec(BaseModel):
    """Route-agnostic analysis request."""

    route: str = Field(default="fiber_bundle", description="Route plugin name.")
    analysis_type: str = Field(
        default="unified_conscious_field",
        description="Logical analysis type handled by route plugin.",
    )
    model: Optional[str] = Field(default=None, description="Target model identifier.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Plugin parameters.")
    input_payload: Dict[str, Any] = Field(
        default_factory=dict, description="Input data for the run."
    )


class Metric(BaseModel):
    key: str
    value: float
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: Optional[str] = None


class ConclusionCard(BaseModel):
    objective: str
    method: str
    evidence: List[str] = Field(default_factory=list)
    result: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    limitations: List[str] = Field(default_factory=list)
    next_action: str = ""


class EventEnvelope(BaseModel):
    event_type: str
    run_id: str
    step: int = 0
    timestamp: float
    payload: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    metrics: List[Metric] = Field(default_factory=list)
    conclusion: Optional[ConclusionCard] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)


class RunRecord(BaseModel):
    run_id: str
    spec: AnalysisSpec
    status: str
    created_at: float
    updated_at: float
    error: Optional[str] = None
    summary: Optional[RunSummary] = None
    event_count: int = 0

