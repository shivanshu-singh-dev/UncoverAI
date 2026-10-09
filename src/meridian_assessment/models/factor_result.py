"""Factor result models with full provenance tracking and human-governance overrides."""

from enum import StrEnum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class DeterminationMethod(StrEnum):
    DIRECT = "DIRECT"                          # O factor — structured field, direct mapping
    DETERMINISTIC_RULE = "DETERMINISTIC_RULE"  # Concept dictionary / rule match
    SEMANTIC_MATCH = "SEMANTIC_MATCH"          # Embedding-based match
    HUMAN_REVIEW = "HUMAN_REVIEW"              # Requires human review
    UNKNOWN = "UNKNOWN"                        # Could not determine


class ScoreStatus(StrEnum):
    COMPLETE = "COMPLETE"          # All factors resolved
    PARTIAL = "PARTIAL"            # Some factors resolved, some UNKNOWN
    REQUIRES_REVIEW = "REQUIRES_REVIEW"  # Ambiguous — needs human review
    UNKNOWN = "UNKNOWN"            # Cannot score


class CriticalityLevel(StrEnum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class SemanticMatchInfo(BaseModel):
    """Details of a semantic embedding match."""
    model_config = ConfigDict(frozen=True)

    embedding_model: str
    top_k: int
    candidate_concepts: list[str] = Field(default_factory=list)
    similarity_scores: list[float] = Field(default_factory=list)
    selected_candidate: Optional[str] = None
    selected_score_level: Optional[int] = None
    review_status: str = "OK"


class FactorResult(BaseModel):
    """Complete, auditable result for a single criticality factor (D/P/R/O/V)."""
    model_config = ConfigDict(frozen=True)

    factor: str = Field(..., description="Factor identifier: D, P, R, O, or V")
    factor_name: str = Field(..., description="Human-readable factor name")

    # Score (integer 0-3, or None if UNKNOWN)
    score: Optional[int] = Field(default=None, description="Factor score 0-3, or None if UNKNOWN")
    is_unknown: bool = Field(default=False, description="True if factor could not be determined")

    # Inputs
    source_fields: list[str] = Field(default_factory=list, description="Field names used as input")
    raw_input: Optional[str] = Field(default=None, description="Raw input text")
    normalized_input: Optional[str] = Field(default=None, description="Normalized input text")

    # Evidence
    matched_concepts: list[str] = Field(default_factory=list)
    matched_phrases: list[str] = Field(default_factory=list)
    semantic_matches: list[SemanticMatchInfo] = Field(default_factory=list)

    # Determination
    determination_method: DeterminationMethod = Field(default=DeterminationMethod.UNKNOWN)
    rationale: str = Field(default="")
    certainty: str = Field(default="")

    # Volume-specific
    volume_type: Optional[str] = Field(default=None, description="COUNT / POPULATION / FREQUENCY / UNKNOWN")
    normalized_volume: Optional[float] = Field(default=None)

    # Weighted contribution (filled after scoring)
    weight: Optional[float] = None
    weighted_contribution: Optional[float] = None

    def debug_dict(self) -> dict[str, Any]:
        """Return a debug representation for explainability."""
        d: dict[str, Any] = {
            "factor": self.factor,
            "score": self.score if not self.is_unknown else "UNKNOWN",
            "source_fields": self.source_fields,
            "raw_text": self.raw_input,
            "normalized_text": self.normalized_input,
            "deterministic_matches": self.matched_phrases,
            "matched_concepts": self.matched_concepts,
            "determination_method": self.determination_method.value,
            "rationale": self.rationale,
        }
        if self.semantic_matches:
            sm = self.semantic_matches[0]
            d.update({
                "embedding_model": sm.embedding_model,
                "top_k": sm.top_k,
                "candidate_concepts": sm.candidate_concepts,
                "similarity_scores": sm.similarity_scores,
                "selected_candidate": sm.selected_candidate,
                "review_status": sm.review_status,
            })
        if self.volume_type:
            d["volume_type"] = self.volume_type
        if self.normalized_volume is not None:
            d["normalized_volume"] = self.normalized_volume
        if self.weight is not None:
            d["weight"] = self.weight
            d["weighted_contribution"] = self.weighted_contribution
        return d


class OverrideRecord(BaseModel):
    """Records whether an O1-O4 override or O5 review was evaluated and triggered."""
    model_config = ConfigDict(frozen=True)

    override_id: str
    description: str
    condition_evaluated: str
    triggered: bool
    resulting_floor: Optional[str] = None
    evidence: Optional[str] = None
    rationale: str = ""


class HumanReviewRecord(BaseModel):
    """Records O5 user governance decision."""
    model_config = ConfigDict(frozen=True)

    user_decision: str = "KEEP_PROPOSED"  # KEEP_PROPOSED or CHANGE_CRITICALITY
    original_criticality: str
    final_criticality: str
    rationale: str = ""
    timestamp: Optional[str] = None


class CriticalityResult(BaseModel):
    """Complete criticality result with full factor provenance and audit trail."""
    model_config = ConfigDict(frozen=True)

    vendor_id: str
    vendor_name: str

    # Factor results
    D: FactorResult
    P: FactorResult
    R: FactorResult
    O: FactorResult
    V: FactorResult

    # Scoring
    base_score: Optional[float] = None
    score_status: ScoreStatus = ScoreStatus.UNKNOWN

    # Criticality Classification Progression
    provisional_criticality: Optional[str] = None  # From threshold table: Critical / High / Medium / Low
    overrides_evaluated: list[OverrideRecord] = Field(default_factory=list)
    automatic_override_applied: bool = False
    proposed_criticality: Optional[str] = None     # After O1-O4 safeguards

    # O5 Governance Layer
    human_review: Optional[HumanReviewRecord] = None
    final_criticality: Optional[str] = None        # Final resulting criticality (Critical / High / Medium / Low)

    # Downstream / Backwards-compat
    criticality_tier: Optional[str] = None         # Alias to final_criticality for backwards compat
    assessment_depth: Optional[str] = None         # Comprehensive / Targeted / Lightweight

    # Flags
    requires_review: bool = False
    review_notes: list[str] = Field(default_factory=list)
