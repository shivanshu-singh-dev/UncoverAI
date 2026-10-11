"""Pydantic data models for Levels 3B–9 of the Meridian Third-Party AI Risk Assessment POC.

Covers:
  - Level 3B: Candidate URL Triage (`URLTriageAction`, `URLTriageDecision`)
  - Level 4: Web/PDF Extraction & Point-of-Capture Evidence (`ExtractedDocument`, `EvidenceRecord`)
  - Level 5: Cleaning, RAG Chunking & Semantic Retrieval (`DocumentChunk`, `RetrievedChunkMatch`)
  - Level 6: Signal Classification & Local-LLM Contracts (`SignalClassificationResult`)
  - Level 7: Evidentiary Weighting & E1–E14 Rules (`ERuleApplication`, `WeightedEvidenceItem`)
  - Gates & Verdict: G1–G4 Gates & AI-Use Verdict (`EvaluatedGateResult`, `AIUseVerdictResult`)
  - Level 8: R1–R8 AI Security Risk Analysis (`AIRiskFinding`)
  - Level 9: Deterministic AI Security Risk Rating (`DeterministicRiskScoreResult`)
  - Human Validation & End-to-End Vendor Report (`HumanValidationRecord`, `VendorEndToEndAssessmentReport`)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Level 3B: URL Triage Models ──────────────────────────────────────────────


class URLTriageAction(StrEnum):
    """Downstream collection action selected by URL triage."""
    FETCH = "FETCH"
    DEFER = "DEFER"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class TriageDecisionMethod(StrEnum):
    """Method used to reach a URL triage decision."""
    DETERMINISTIC_HEURISTIC = "DETERMINISTIC_HEURISTIC"
    LOCAL_LLM = "LOCAL_LLM"
    ANALYST_OVERRIDE = "ANALYST_OVERRIDE"


class URLTriageDecision(BaseModel):
    """Structured triage assessment for a CandidateURL."""
    candidate_id: str
    vendor_id: str
    url: str
    action: URLTriageAction
    triage_status: str
    vendor_match: bool
    product_scope_relevance: str = "UNRESOLVED"  # MERIDIAN_SERVICE / VENDOR_PRODUCT / VENDOR_FIRM_WIDE / UNRELATED / UNRESOLVED
    candidate_source_class: str = "S2"
    target_gates: list[str] = Field(default_factory=list)
    evidence_questions_addressed: list[str] = Field(default_factory=list)
    rationale: str
    decision_method: TriageDecisionMethod = TriageDecisionMethod.DETERMINISTIC_HEURISTIC
    confidence: str = "MEDIUM"
    triaged_at: str
    model_used: Optional[str] = None
    model_execution_status: Optional[str] = None



# ─── Level 4: Web/PDF Extraction & Evidence Models ────────────────────────────


class ExtractionStatus(StrEnum):
    """Status of webpage or PDF extraction."""
    EXTRACTED = "EXTRACTED"
    EMPTY_CONTENT = "EMPTY_CONTENT"
    PASSWORD_PROTECTED = "PASSWORD_PROTECTED"
    SCANNED_IMAGE_ONLY = "SCANNED_IMAGE_ONLY"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    BLOCKED_UNSAFE_URL = "BLOCKED_UNSAFE_URL"
    FETCH_FAILED = "FETCH_FAILED"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceScope(StrEnum):
    """Organizational/service scope of an evidence item."""
    MERIDIAN_SERVICE = "MERIDIAN_SERVICE"
    VENDOR_PRODUCT = "VENDOR_PRODUCT"
    VENDOR_FIRM_WIDE = "VENDOR_FIRM_WIDE"
    THIRD_PARTY_UNRELATED = "THIRD_PARTY_UNRELATED"
    UNRESOLVED = "UNRESOLVED"


class AIType(StrEnum):
    """Technical classification of AI capability."""
    GENAI_LLM = "GENAI_LLM"
    CLASSICAL_ML = "CLASSICAL_ML"
    AGENTIC_AI = "AGENTIC_AI"
    NONE_AUTOMATION_ONLY = "NONE_AUTOMATION_ONLY"
    NONE = "NONE"
    UNRESOLVED = "UNRESOLVED"


class EvidenceStrength(StrEnum):
    """Evidentiary classification disposition under rules E1–E14."""
    CONFIRMED = "CONFIRMED"
    STRONG = "STRONG"
    INDICATIVE = "INDICATIVE"
    MARKETING = "MARKETING"
    DISCARDED_HOSTILE = "DISCARDED_HOSTILE"
    NONE = "NONE"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class ExtractedDocument(BaseModel):
    """Point-of-capture extracted webpage or PDF document."""
    document_id: str
    vendor_id: str
    url: str
    canonical_url: str
    source_class: str
    content_type: str = "text/html"
    title: Optional[str] = None
    raw_text: str = ""
    cleaned_text: str = ""
    page_texts: dict[int, str] = Field(default_factory=dict)
    content_sha256: str = ""
    byte_size: int = 0
    retrieved_at: str
    extraction_status: ExtractionStatus = ExtractionStatus.EXTRACTED
    extraction_limitations: list[str] = Field(default_factory=list)
    artifact_path: Optional[str] = None
    query_ids: list[str] = Field(default_factory=list)
    triage_action: Optional[str] = None
    is_synthetic_fixture: bool = False


class EvidenceRecord(BaseModel):
    """Complete, traceable evidence record linked to its point-of-capture source."""
    evidence_id: str
    document_id: Optional[str] = None
    vendor_id: str
    legal_entity: str
    domain: str
    source_class: str
    canonical_url: str
    original_url: str
    retrieved_at: str
    page_title: Optional[str] = None
    section_heading: Optional[str] = None
    verbatim_excerpt: str
    page_number: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    content_sha256: str
    capture_artifact_ref: Optional[str] = None
    scope: EvidenceScope = EvidenceScope.UNRESOLVED
    ai_type: AIType = AIType.UNRESOLVED
    named_model: Optional[str] = None
    named_provider: Optional[str] = None
    hosting_infrastructure: Optional[str] = None
    source_reliability_grade: str = "C"
    source_reliability_rating: int = 2
    e_rule_ids: list[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength = EvidenceStrength.REQUIRES_REVIEW
    prior_strength_before_downgrade: Optional[EvidenceStrength] = None
    is_historical_or_removed: bool = False
    published_or_observed_date: Optional[str] = None
    is_opt_in_only: bool = False
    activation_caveats: Optional[str] = None
    no_training_commitment: bool = False
    zdr_commitment: bool = False
    hostile_instruction_detected: bool = False
    security_events: list[str] = Field(default_factory=list)
    risk_tags: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    query_ids: list[str] = Field(default_factory=list)
    triage_decision: Optional[str] = None
    extraction_status: ExtractionStatus = ExtractionStatus.EXTRACTED
    extraction_limitations: list[str] = Field(default_factory=list)
    is_reviewed_and_accepted: bool = False
    reviewer_notes: Optional[str] = None
    is_synthetic_test_fixture: bool = False


# ─── Level 5: Semantic Filtering & RAG Chunking Models ────────────────────────


class EmbeddingStatus(StrEnum):
    """Status of vector embedding generation for a document chunk."""
    EMBEDDED = "EMBEDDED"
    UNAVAILABLE_NOT_CONFIGURED = "UNAVAILABLE_NOT_CONFIGURED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class DocumentChunk(BaseModel):
    """A traceable text chunk derived from an ExtractedDocument / EvidenceRecord."""
    chunk_id: str
    document_id: str
    evidence_id: Optional[str] = None
    vendor_id: str
    url: str
    source_class: str
    retrieved_at: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    token_estimate: int
    content_sha256: str
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    embedding_status: EmbeddingStatus = EmbeddingStatus.UNAVAILABLE_NOT_CONFIGURED
    embedding_model_version: Optional[str] = None


class RetrievedChunkMatch(BaseModel):
    """A chunk retrieved against a risk-query concept."""
    chunk: DocumentChunk
    concept_id: str
    concept_query: str
    similarity_score: Optional[float] = None
    retrieval_method: str = "UNAVAILABLE"


class SemanticRetrievalSummary(BaseModel):
    """Summary of chunking and semantic retrieval execution."""
    status: str  # COMPLETED / MODEL_UNAVAILABLE_NOT_CONFIGURED / NO_DOCUMENTS
    model_name: str
    model_available: bool = False
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    chunks_created: int = 0
    chunks_embedded: int = 0
    matches: list[RetrievedChunkMatch] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# ─── Level 6: Signal Classification Models ────────────────────────────────────


class SignalCategory(StrEnum):
    """Distinction between operational AI, marketing commentary, non-AI automation, and hostile text."""
    OPERATIONAL_AI = "OPERATIONAL_AI"
    MARKETING_COMMENTARY = "MARKETING_COMMENTARY"
    NON_AI_AUTOMATION = "NON_AI_AUTOMATION"
    HOSTILE_UNTRUSTED = "HOSTILE_UNTRUSTED"
    NO_AI_SIGNAL = "NO_AI_SIGNAL"
    UNRESOLVED = "UNRESOLVED"


class ClassificationMethod(StrEnum):
    """Execution mode used to produce a SignalClassificationResult."""
    DETERMINISTIC_RULES = "DETERMINISTIC_RULES"
    LOCAL_LLM_VALIDATED = "LOCAL_LLM_VALIDATED"
    LLM_UNAVAILABLE_DETERMINISTIC_FALLBACK = "LLM_UNAVAILABLE_DETERMINISTIC_FALLBACK"


class SignalClassificationResult(BaseModel):
    """Structured evaluation of an extracted evidence item."""
    classification_id: str
    evidence_id: str
    vendor_id: str
    signal_category: SignalCategory
    ai_type: AIType
    scope: EvidenceScope
    named_provider: Optional[str] = None
    named_model: Optional[str] = None
    hosting_infrastructure: Optional[str] = None
    capability_summary: str
    known_inputs: list[str] = Field(default_factory=list)
    known_outputs: list[str] = Field(default_factory=list)
    customer_data_involved: Optional[bool] = None
    is_opt_in_only: bool = False
    activation_opt_in_caveats: Optional[str] = None
    no_training_commitment: bool = False
    zdr_commitment: bool = False
    supporting_verbatim_text: str
    rationale: str
    confidence: str = "MEDIUM"
    review_status: str = "REQUIRES_REVIEW"
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_e_rules: list[str] = Field(default_factory=list)
    prompt_injection_detected: bool = False
    security_events: list[str] = Field(default_factory=list)
    classification_method: ClassificationMethod = ClassificationMethod.DETERMINISTIC_RULES
    model_used: Optional[str] = None
    model_execution_status: Optional[str] = None



# ─── Level 7: Evidentiary Weighting & E1–E14 Models ───────────────────────────


class ERuleApplication(BaseModel):
    """Audit record of a single E1–E14 rule evaluation on an evidence item."""
    rule_id: str
    rule_name: str
    triggered: bool
    resulting_strength: Optional[EvidenceStrength] = None
    rationale: str


class EvidentiaryWeightBand(StrEnum):
    """Broad weight bands from the Meridian case-study brief."""
    HIGH = "HIGH"        # 0.8 - 1.0
    MEDIUM = "MEDIUM"    # 0.5 - 0.7
    LOW = "LOW"          # 0.1 - 0.4
    NONE = "NONE"        # 0.0 (discarded / hostile / non-AI)


class WeightedEvidenceItem(BaseModel):
    """Result of applying E1–E14 rules and evidentiary weighting to an EvidenceRecord."""
    evidence_id: str
    vendor_id: str
    source_class: str
    source_reliability_grade: str
    source_reliability_rating: int
    evidence_strength: EvidenceStrength
    prior_strength_before_downgrade: Optional[EvidenceStrength] = None
    applied_e_rules: list[ERuleApplication] = Field(default_factory=list)
    fired_e_rule_ids: list[str] = Field(default_factory=list)
    corroborated_by_evidence_ids: list[str] = Field(default_factory=list)
    corroborating_source_classes: list[str] = Field(default_factory=list)
    weight_band: EvidentiaryWeightBand = EvidentiaryWeightBand.LOW
    weight_band_range: tuple[float, float] = (0.1, 0.4)
    configured_scalar_weight: Optional[float] = None
    weight_status: str = "UNCONFIGURED_BAND_ONLY"
    disposition_explanation: str = ""


# ─── Gates G1–G4 & AI-Use Verdict Models ──────────────────────────────────────


class EvaluatedGateResult(BaseModel):
    """Detailed G1–G4 gate evaluation with evidence links and analyst override support."""
    gate_id: str
    title: str
    question: str
    status: str  # QUALIFYING_POSITIVE_EVIDENCE / QUALIFYING_NEGATIVE_EVIDENCE / UNRESOLVED / NOT_INVESTIGATED
    rationale: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    evaluated_at: str
    review_status: str = "AUTOMATED_PROPOSED"
    open_questions: list[str] = Field(default_factory=list)
    analyst_override_rationale: Optional[str] = None


class AIUseVerdictValue(StrEnum):
    """Final or interim AI-use verdict."""
    YES = "YES"
    INCONCLUSIVE = "INCONCLUSIVE"
    NO = "NO"
    INCOMPLETE_ASSESSMENT = "INCOMPLETE_ASSESSMENT"


class AIUseVerdictResult(BaseModel):
    """Rulebook-compliant AI-use verdict with completeness safeguards."""
    verdict: AIUseVerdictValue
    verdict_statement: str
    basis_summary: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    corroborating_source_classes: list[str] = Field(default_factory=list)
    depth_completed: bool = False
    incomplete_reasons: list[str] = Field(default_factory=list)
    requires_human_review: bool = True


# ─── Level 8: R1–R8 AI Security Risk Analysis Models ──────────────────────────


class ExposureDetermination(StrEnum):
    """Explicit distinction between confirmed exposure, indicated risk, missing disclosure, Mitigated, or unknown."""
    CONFIRMED_EXPOSURE = "CONFIRMED_EXPOSURE"
    INDICATED_BY_EVIDENCE = "INDICATED_BY_EVIDENCE"
    CONTROL_NOT_PUBLICLY_EVIDENCED = "CONTROL_NOT_PUBLICLY_EVIDENCED"
    MITIGATED_BY_EVIDENCE = "MITIGATED_BY_EVIDENCE"
    UNKNOWN_UNRESOLVED = "UNKNOWN_UNRESOLVED"


class FindingSeverity(StrEnum):
    """Ordered severity levels for R1–R8 risk findings."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"
    UNRESOLVED = "UNRESOLVED"


class AIRiskFinding(BaseModel):
    """Structured R1–R8 AI security risk finding backed by traceable evidence."""
    finding_id: str
    vendor_id: str
    risk_rule_id: str  # R1 .. R8
    risk_title: str
    finding_description: str
    exposure_determination: ExposureDetermination
    severity: FindingSeverity = FindingSeverity.UNRESOLVED
    evidence_ids: list[str] = Field(default_factory=list)
    quotations: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    retrieval_dates: list[str] = Field(default_factory=list)
    fired_e_rules: list[str] = Field(default_factory=list)
    fired_gate_rules: list[str] = Field(default_factory=list)
    exposed_assets_data_processes: list[str] = Field(default_factory=list)
    known_controls_and_mitigations: list[str] = Field(default_factory=list)
    confidence: str = "MEDIUM"
    limitations: list[str] = Field(default_factory=list)
    recommended_remediation: str
    requires_human_review: bool = True
    dedup_exposure_key: str
    analyst_explanation: Optional[str] = None
    summary_method: str = "DETERMINISTIC_TEMPLATE"
    model_used: Optional[str] = None
    model_execution_status: Optional[str] = None



# ─── Level 9: Deterministic AI Security Risk Rating Models ────────────────────


class ScoringPolicyStatus(StrEnum):
    """Configuration readiness status of the deterministic AI risk scoring policy."""
    APPROVED_ACTIVE = "APPROVED_ACTIVE"
    PROVISIONAL_HUMAN_SELECTED = "PROVISIONAL_HUMAN_SELECTED"
    SCORING_POLICY_UNCONFIGURED = "SCORING_POLICY_UNCONFIGURED"
    TEST_FIXTURE_ONLY = "TEST_FIXTURE_ONLY"


class FindingScoreBreakdown(BaseModel):
    """Per-finding numerical contribution breakdown when a scoring policy is active."""
    finding_id: str
    risk_rule_id: str
    dedup_exposure_key: str
    severity_label: str
    severity_numeric: Optional[float] = None
    evidentiary_weight: Optional[float] = None
    weighted_contribution: Optional[float] = None
    source_evidence_ids: list[str] = Field(default_factory=list)
    is_deduplicated_primary: bool = True
    notes: str = ""


class DeterministicRiskScoreResult(BaseModel):
    """Result of Level 9 deterministic risk formula execution (`Risk Score = M_c * sum(S_i * W_i)`)."""
    vendor_id: str
    policy_status: ScoringPolicyStatus
    policy_version: str
    calculation_method: str
    status_message: str
    criticality_tier: str
    criticality_multiplier: Optional[float] = None
    finding_breakdowns: list[FindingScoreBreakdown] = Field(default_factory=list)
    distinct_findings_scored: int = 0
    duplicates_suppressed: int = 0
    sum_weighted_exposures: Optional[float] = None
    raw_risk_score: Optional[float] = None
    final_risk_score: Optional[float] = None
    overall_risk_rating: Optional[str] = None
    escalation_rules_evaluated: list[dict[str, Any]] = Field(default_factory=list)
    escalation_applied: bool = False
    unconfigured_parameters: list[str] = Field(default_factory=list)
    assessment_completeness_status: str = "INCOMPLETE"
    calculated_at: str


# ─── Human Validation & End-to-End Vendor Assessment Report ──────────────────


class HumanValidationRecord(BaseModel):
    """Audit record of an analyst validation or override decision across any pipeline stage."""
    validation_id: str
    vendor_id: str
    target_type: str  # CRITICALITY_O5 / GATE / VERDICT / RISK_FINDING / SCORING_POLICY
    target_id: str
    proposed_automated_finding: str
    evidence_references: list[str] = Field(default_factory=list)
    human_decision: str
    rationale: str
    reviewer_id: str = "Primary Analyst"
    reviewed_at: str
    original_status: str
    updated_status: str
    outstanding_questions: list[str] = Field(default_factory=list)


class StageExecutionRecord(BaseModel):
    """Execution status of one of the 9 project levels during an end-to-end vendor run."""
    level_number: int
    level_name: str
    status: str  # COMPLETED / PARTIAL / NOT_CONFIGURED / UNAVAILABLE / REQUIRES_REVIEW / SKIPPED_NO_INPUT
    summary: str
    limitations: list[str] = Field(default_factory=list)
    selected_model: Optional[str] = None
    model_task: Optional[str] = None
    model_execution_status: Optional[str] = None


class VendorEndToEndAssessmentReport(BaseModel):
    """Complete, auditable per-vendor assessment report spanning all 9 project levels."""
    report_id: str
    generated_at: str
    is_synthetic_evidence_run: bool = False

    # 1. Vendor Identity & Scope
    vendor_id: str
    legal_entity: str
    domain: str
    category: Optional[str] = None
    vendor_description: Optional[str] = None
    service_product_provided: Optional[str] = None
    business_process_supported: Optional[str] = None

    # 2. Level 1: Criticality Tiering & O1–O5 Governance
    criticality_summary: dict[str, Any] = Field(default_factory=dict)

    # 3. Level 2 & 3: OSINT Depth, Sources, Queries, URL Triage & Completeness
    osint_plan_summary: dict[str, Any] = Field(default_factory=dict)
    url_triage_decisions: list[URLTriageDecision] = Field(default_factory=list)

    # 4. Level 4: Extracted Documents & Evidence Records
    extracted_documents_count: int = 0
    evidence_records: list[EvidenceRecord] = Field(default_factory=list)

    # 5. Level 5: Semantic Filtering & RAG Chunking
    semantic_retrieval: SemanticRetrievalSummary

    # 6. Level 6 & 7: Signal Classifications, E1–E14 & Evidentiary Weights
    signal_classifications: list[SignalClassificationResult] = Field(default_factory=list)
    weighted_evidence: list[WeightedEvidenceItem] = Field(default_factory=list)

    # 7. Gates G1–G4, AI-Use Verdict, AI Type/Scope & Provider/Model/Hosting
    gates: list[EvaluatedGateResult] = Field(default_factory=list)
    ai_use_verdict: AIUseVerdictResult
    determined_ai_types: list[str] = Field(default_factory=list)
    determined_scope: str = "UNRESOLVED"
    named_models: list[str] = Field(default_factory=list)
    named_providers: list[str] = Field(default_factory=list)
    hosting_infrastructures: list[str] = Field(default_factory=list)
    supply_chain_identity_status: str = "UNKNOWN"

    # 8. Level 8: R1–R8 Findings, Known Controls & Recommendations
    risk_findings: list[AIRiskFinding] = Field(default_factory=list)
    known_controls_summary: list[str] = Field(default_factory=list)
    remediation_recommendations: list[dict[str, str]] = Field(default_factory=list)

    # 9. Level 9: Deterministic Risk Score & Policy Status
    deterministic_risk_score: DeterministicRiskScoreResult

    # 10. Human Validation & Remaining Questions
    human_Review_required: bool = True
    human_review_triggers: list[str] = Field(default_factory=list)
    human_validations: list[HumanValidationRecord] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)

    # 11. Nine-Level Stage Execution Ledger, Model Routing Telemetry & Limitations
    stage_ledger: list[StageExecutionRecord] = Field(default_factory=list)
    model_routing_telemetry: dict[str, Any] = Field(default_factory=dict)
    unavailable_integrations: list[str] = Field(default_factory=list)
    assessment_limitations: list[str] = Field(default_factory=list)

