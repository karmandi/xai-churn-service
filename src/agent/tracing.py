from typing import Any
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    step_num: int = Field(..., ge=1)
    action: str
    tool_name: str | None = None
    latency_ms: float = Field(..., ge=0.0)
    tokens_used: int = Field(..., ge=0)
    status: str
    error: str | None = None


class AgentTrace(BaseModel):
    trace_id: str
    total_latency_ms: float = Field(..., ge=0.0)
    total_tokens: int = Field(..., ge=0)
    status: str
    steps: list[TraceStep] = Field(default_factory=list)
    final_output: dict[str, Any] | str | None = None
