from typing import List, Literal
from pydantic import BaseModel, Field


class AnchorCandidate(BaseModel):
    anchor_id: str
    page_number: int
    anchor_text: str
    anchor_rect: List[float]
    context_before: str = ""
    context_after: str = ""
    page_width: float
    page_height: float


class ImageAsset(BaseModel):
    image_id: str
    filename: str
    path: str
    width: int
    height: int


class PlacementDecision(BaseModel):
    image_id: str
    filename: str
    anchor_id: str
    page_number: int
    mode: Literal["below", "right", "appendix_page"] = "below"
    size_hint: Literal["small", "medium", "large"] = "medium"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class ProcessResult(BaseModel):
    job_id: str
    output_file_name: str
    output_file_path: str
    output_type: Literal["pdf", "docx"]
    placements: List[PlacementDecision]
    notes: List[str] = []
