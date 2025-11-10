from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, root_validator, validator


class PlanTaskSchema(BaseModel):
    """Schema describing a single plan task."""

    id: str = Field(..., min_length=1)
    title: str = Field(default="")
    kind: Optional[str] = None
    content_type: Optional[str] = None
    order: int | None = Field(default=None, ge=0)
    layout_block_id: Optional[str] = None
    block_id: Optional[str] = None
    asset_path: Optional[str] = None
    notes: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class PlanSchema(BaseModel):
    """Schema describing the full planner output consumed downstream."""

    doc_class: Optional[str] = None
    doc_class_hint: Dict[str, Any] | None = None
    frontmatter: Dict[str, Any] = Field(default_factory=dict)
    content_flags: Dict[str, bool] = Field(default_factory=dict)
    tasks: List[PlanTaskSchema] = Field(default_factory=list)

    class Config:
        extra = "allow"


class ConsensusBlockSchema(BaseModel):
    """Schema describing post-judge consensus blocks."""

    block_id: str
    text: str = ""
    block_type: str = "text"
    page_index: int = Field(default=0, ge=0)
    flagged: bool = False
    ocr_outputs: Dict[str, str] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class LayoutBlueprintSchema(BaseModel):
    """Schema for layout planner blueprints."""

    version: str
    model_name: str
    created_at: str
    plan: Dict[str, Any]
    raw_response: Optional[str] = None
    source: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    class Config:
        extra = "allow"

    @root_validator(pre=True)
    def _coerce_model_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "model_name" not in values and "model" in values:
            values["model_name"] = values["model"]
        return values

    @validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError("created_at must be an ISO 8601 string") from exc
        return value


__all__ = [
    "ConsensusBlockSchema",
    "LayoutBlueprintSchema",
    "PlanSchema",
    "PlanTaskSchema",
    "ValidationError",
]
