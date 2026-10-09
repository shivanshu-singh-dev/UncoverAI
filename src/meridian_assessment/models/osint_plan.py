"""Pydantic data models for the Meridian OSINT Depth Engine and Investigation Planner."""

from enum import StrEnum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class CoverageState(StrEnum):
    """Investigation status for an individual source class."""
    NOT_STARTED = "NOT_STARTED"
    SEARCH_ATTEMPTED = "SEARCH_ATTEMPTED"
    RESULTS_FOUND = "RESULTS_FOUND"
    RELEVANT_SOURCE_IDENTIFIED = "RELEVANT_SOURCE_IDENTIFIED"
    REVIEWED_EVIDENCE_RECORDED = "REVIEWED_EVIDENCE_RECORDED"
    REVIEWED_NO_EVIDENCE = "REVIEWED_NO_EVIDENCE"
    UNAVAILABLE = "UNAVAILABLE"
    REQUIRES_MANUAL_REVIEW = "REQUIRES_MANUAL_REVIEW"


class SearchExecutionStatus(StrEnum):
    """Lifecycle status of a planned search query."""
    PLANNED = "PLANNED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class URLTriageStatus(StrEnum):
    """Downstream triage status of a candidate URL."""
    PENDING = "PENDING"
    RELEVANT = "RELEVANT"
    IRRELEVANT = "IRRELEVANT"
    INACCESSIBLE = "INACCESSIBLE"
    DUPLICATE = "DUPLICATE"


class GateStatus(StrEnum):
    """Investigation status for the G1–G4 investigative gates."""
    QUALIFYING_POSITIVE_EVIDENCE = "QUALIFYING_POSITIVE_EVIDENCE"
    QUALIFYING_NEGATIVE_EVIDENCE = "QUALIFYING_NEGATIVE_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"
    NOT_INVESTIGATED = "NOT_INVESTIGATED"


class AssessmentStopStatus(StrEnum):
    """High-level completion status of the OSINT assessment."""
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED_CONDITION_MET = "COMPLETED_CONDITION_MET"
    TIMEBOX_EXHAUSTED_INCOMPLETE = "TIMEBOX_EXHAUSTED_INCOMPLETE"


class CriticalityApprovalStatus(StrEnum):
    """Approval governance status of the criticality input used for planning."""
    APPROVED_BY_ANALYST = "APPROVED_BY_ANALYST"
    PROPOSED_UNREVIEWED = "PROPOSED_UNREVIEWED"
    PROVISIONAL_CALCULATED = "PROVISIONAL_CALCULATED"
    LEGACY_TIER = "LEGACY_TIER"


class SourceClassDefinition(BaseModel):
    """Static definition of an official S1–S12 source class."""
    model_config = ConfigDict(frozen=True)

    source_id: str
    name: str
    description: str
    default_evidence_grade: str
    reliability_rating: int
    is_corroborative: bool
    evidence_targets: list[str] = Field(default_factory=list)
    discovery_resources: list[str] = Field(default_factory=list)
    query_template_ids: list[str] = Field(default_factory=list)


class DiscoveryResource(BaseModel):
    """Catalog entry for an external search or technical discovery resource."""
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    category: str
    associated_sources: list[str] = Field(default_factory=list)
    purpose: str
    access_mode: str
    limitations: str


class PlannedQuery(BaseModel):
    """A vendor-specific query generated from approved templates."""
    model_config = ConfigDict(frozen=True)

    query_id: str
    vendor_id: str
    source_classes: list[str] = Field(default_factory=list)
    discovery_resources: list[str] = Field(default_factory=list)
    investigative_objectives: list[str] = Field(default_factory=list)
    template_used: str
    rendered_query: str
    expected_evidence_targets: list[str] = Field(default_factory=list)
    date_constraints: Optional[str] = None
    execution_status: SearchExecutionStatus = SearchExecutionStatus.PLANNED


class CandidateURL(BaseModel):
    """Interface model representing a candidate URL found during OSINT collection."""
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    url: str
    vendor_id: str
    source_class: str
    discovery_resource: str
    query_id: str
    triage_status: URLTriageStatus = URLTriageStatus.PENDING
    title: Optional[str] = None
    snippet: Optional[str] = None
    notes: Optional[str] = None


class SourceCoverageRecord(BaseModel):
    """Audit record tracking investigation progress for a single source class."""
    source_id: str
    source_name: str
    is_required: bool
    is_corroborative: bool = False
    status: CoverageState = CoverageState.NOT_STARTED
    candidate_urls_count: int = 0
    reviewed_urls_count: int = 0
    produced_new_evidence: bool = False
    notes: str = ""


class GateChecklistItem(BaseModel):
    """Investigative objective and qualifying criteria for G1–G4 gates."""
    gate_id: str
    title: str
    question: str
    status: GateStatus = GateStatus.NOT_INVESTIGATED
    investigative_objectives: list[str] = Field(default_factory=list)
    qualifying_criteria: str
    evidence_notes: str = ""


class TimeboxBudget(BaseModel):
    """Time budget tracking and measurement for the assessment."""
    planning_target: str
    min_hours: float
    max_hours: float
    is_provisional: bool = True
    actual_analyst_time_hours: float = 0.0
    actual_automated_execution_time_minutes: float = 0.0
    time_spent_searching_minutes: float = 0.0
    time_spent_reviewing_minutes: float = 0.0
    time_spent_resolving_ambiguous_minutes: float = 0.0
    time_spent_validating_minutes: float = 0.0
    is_exhausted: bool = False


class PolicyConflictWarning(BaseModel):
    """Explicit advisory flag surfacing tensions between team policy and rulebook."""
    source_id: str
    source_name: str
    issue: str
    rulebook_requirement: str
    team_policy_claim: str
    impact_notes: str


class OSINTInvestigationPlan(BaseModel):
    """Complete, auditable OSINT investigation plan for a vendor."""
    plan_id: str
    vendor_id: str
    vendor_name: str
    domain: str
    vendor_description: Optional[str] = None
    service_product_provided: Optional[str] = None
    business_process_supported: Optional[str] = None

    # Criticality inputs
    input_criticality: str
    approval_status: CriticalityApprovalStatus

    # Selected Depth Profile
    selected_depth_profile: str
    selection_reason: str
    required_source_classes: list[str] = Field(default_factory=list)
    corroborative_source_classes: list[str] = Field(default_factory=list)
    conditional_source_classes: list[dict[str, str]] = Field(default_factory=list)

    # Coverage & queries
    coverage_records: dict[str, SourceCoverageRecord] = Field(default_factory=dict)
    associated_discovery_resources: list[str] = Field(default_factory=list)
    planned_queries: list[PlannedQuery] = Field(default_factory=list)

    # Timebox & stopping condition
    timebox: TimeboxBudget
    stopping_rule_id: str
    stopping_rule_description: str
    stop_status: AssessmentStopStatus = AssessmentStopStatus.IN_PROGRESS
    stop_progress_notes: str = ""

    # Reviewer requirements & gates
    reviewer_requirements: dict[str, Any] = Field(default_factory=dict)
    gate_checklist: list[GateChecklistItem] = Field(default_factory=list)
    policy_conflicts: list[PolicyConflictWarning] = Field(default_factory=list)
    unresolved_planning_issues: list[str] = Field(default_factory=list)

    # Metadata
    generated_at: str
