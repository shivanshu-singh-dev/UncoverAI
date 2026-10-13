"""Unit tests verifying the 21-Field Canonical Meridian Vendor Inventory Schema and Exporter.

Tests:
1. Exact preservation of all 10 supplied vendor input fields without mutation.
2. Full derivation and typing of all 11 student-derived assessment fields.
3. Strict mapping of `AI Usage Detected` to 'Yes', 'No', or 'Inconclusive'.
4. RFC 4180 CSV export validation (21 columns, proper escaping of multiline text and quotes).
5. Multi-vendor batch CSV export.
6. Analytical rigor matching the Larkspur worked example (rationale, depth, providers, risk, actions).
"""

import csv
import io
import pytest

from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.models.ai_risk import (
    AIUseVerdictResult,
    AIUseVerdictValue,
    DeterministicRiskScoreResult,
    EvidenceRecord,
    EvidenceScope,
    EvidenceStrength,
    EvidentiaryWeightBand,
    FindingSeverity,
    ScoringPolicyStatus,
    SemanticRetrievalSummary,
    VendorEndToEndAssessmentReport,
    WeightedEvidenceItem,
    AIRiskFinding,
    ExposureDetermination,
)
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.services.end_to_end_pipeline import MeridianEndToEndPipeline
from meridian_assessment.services.readiness_and_exports import (
    CANONICAL_VENDOR_INVENTORY_COLUMNS,
    AssessmentReportExporter,
    build_canonical_vendor_inventory_record,
)


def _build_test_report_and_vendor(verdict: AIUseVerdictValue = AIUseVerdictValue.YES) -> tuple[VendorEndToEndAssessmentReport, MeridianVendor]:
    vendor = MeridianVendor(
        vendor_id="V-001",
        vendor_name="AutomWorx",
        domain="www.automworx.com",
        category="Professional Services",
        vendor_description="An independent consulting firm specializing in workload automation solutions.",
        service_product_provided="Implementation, migration, managed support and modernization of the enterprise workload automation estate.",
        business_process_supported="Batch scheduling across core systems",
        data_classification_accessed="No direct customer data. Privileged access to production job schedules, system configuration and service credentials.",
        operational_dependency="High",
        data_volume_annual="Not applicable",
    )

    ev_rec = EvidenceRecord(
        evidence_id="EV_V001_001",
        vendor_id="V-001",
        legal_entity="AutomWorx",
        domain="www.automworx.com",
        source_class="S1",
        source_reliability_grade="A",
        canonical_url="https://docs.automworx.com/admin/scheduler",
        original_url="https://docs.automworx.com/admin/scheduler",
        retrieved_at="2026-10-13T10:00:00Z",
        published_or_observed_date="2026-01-15",
        content_sha256="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        verbatim_excerpt="AutomWorx integrates deterministic rule-based scheduling with automated cron job execution.",
        scope=EvidenceScope.MERIDIAN_SERVICE,
        no_training_commitment=True,
    )

    weighted = WeightedEvidenceItem(
        evidence_id="EV_V001_001",
        vendor_id="V-001",
        source_class="S1",
        source_reliability_grade="A",
        source_reliability_rating=1,
        evidence_strength=EvidenceStrength.STRONG,
        weight_band=EvidentiaryWeightBand.MEDIUM,
        weight_status="UNCONFIGURED_BAND_ONLY",
        fired_e_rule_ids=["E6"],
        disposition_explanation="Deterministic batch scheduling without AI model training.",
    )

    finding = AIRiskFinding(
        finding_id="RF_V001_R1",
        vendor_id="V-001",
        risk_rule_id="R1",
        risk_title="Customer Data Used for Model Training or Fine-Tuning",
        finding_description="Contract-grade disclosure confirms customer data is not used for model training.",
        exposure_determination=ExposureDetermination.MITIGATED_BY_EVIDENCE,
        severity=FindingSeverity.LOW,
        evidence_ids=["EV_V001_001"],
        quotations=["AutomWorx integrates deterministic rule-based scheduling with automated cron job execution."],
        source_urls=["https://docs.automworx.com/admin/scheduler"],
        retrieval_dates=["2026-10-13T10:00:00Z"],
        recommended_remediation="Execute formal AI Data Processing Addendum prohibiting customer data training.",
        dedup_exposure_key="V-001:R1:TRAINING",
    )

    report = VendorEndToEndAssessmentReport(
        report_id="RPT_V001_20261013120000",
        generated_at="2026-10-13T12:00:00Z",
        is_synthetic_evidence_run=False,
        vendor_id=vendor.vendor_id,
        legal_entity=vendor.vendor_name,
        domain=vendor.domain,
        category=vendor.category,
        vendor_description=vendor.vendor_description,
        service_product_provided=vendor.service_product_provided,
        business_process_supported=vendor.business_process_supported,
        data_classification_accessed=vendor.data_classification_accessed,
        operational_dependency=vendor.operational_dependency,
        data_volume_annual=vendor.data_volume_annual,
        criticality_summary={
            "final_criticality": "High",
            "criticality_tier": "High",
            "base_score": 1.3,
            "provisional_criticality": "Medium",
            "automatic_override_applied": True,
            "overrides_evaluated": [
                {
                    "override_id": "O1",
                    "description": "Privileged / administrative access to production systems",
                    "triggered": True,
                    "resulting_floor": "High",
                    "rationale": "Vendor holds confirmed privileged access to production job schedules.",
                }
            ],
            "factors": {
                "D": {"score": 3, "rationale": "Privileged production credentials accessed."},
                "O": {"score": 2, "rationale": "High operational dependency mapped."},
                "P": {"score": 0, "rationale": "No payment flow involvement."},
                "R": {"score": 0, "rationale": "No regulatory obligations."},
                "V": {"score": 0, "rationale": "Not applicable."},
            },
        },
        osint_plan_summary={
            "selected_depth_profile": "Comprehensive",
            "selection_reason": "High criticality tier mandates comprehensive investigation",
        },
        evidence_records=[ev_rec],
        weighted_evidence=[weighted],
        semantic_retrieval=SemanticRetrievalSummary(
            model_name="all-MiniLM-L6-v2",
            status="DETERMINISTIC_PASSTHROUGH",
            chunks_created=5,
            chunks_embedded=0,
            retrieval_duration_ms=10.0,
        ),
        ai_use_verdict=AIUseVerdictResult(
            verdict=verdict,
            verdict_statement="AI-Use evaluation result",
            basis_summary="Automated batch scheduling without neural model usage",
            depth_completed=True,
            incomplete_reasons=[],
        ),
        risk_findings=[finding],
        deterministic_risk_score=DeterministicRiskScoreResult(
            vendor_id="V-001",
            policy_status=ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED,
            policy_version="1.0.0",
            calculation_method="M_c * sum(S_i * W_i)",
            status_message="Scoring policy unconfigured; qualitative risk classification applied.",
            criticality_tier="High",
            calculated_at="2026-10-13T12:00:00Z",
        ),
    )
    return report, vendor


def test_canonical_columns_count_and_names():
    """Verify that CANONICAL_VENDOR_INVENTORY_COLUMNS has exactly 21 fields in the specified order."""
    assert len(CANONICAL_VENDOR_INVENTORY_COLUMNS) == 21
    expected_10_inputs = [
        "Vendor ID",
        "Vendor Name",
        "Vendor Description",
        "Service / Product Provided to Meridian",
        "Category",
        "Public Website",
        "Business Process Supported",
        "Operational Dependency",
        "Data Classification Accessed",
        "Data Volume (annual)",
    ]
    expected_11_derived = [
        "Criticality Tier",
        "Criticality Rationale",
        "Assessment Depth Applied",
        "AI Usage Detected",
        "Evidence Description along with OSINT Source(s) / Evidence Link",
        "How Vendor May Be Using AI",
        "Named AI Sub-processors / Model Providers",
        "AI Security Risk Classification",
        "Risk Rationale",
        "Recommended Action",
        "Assessed By / Date",
    ]
    assert CANONICAL_VENDOR_INVENTORY_COLUMNS[:10] == expected_10_inputs
    assert CANONICAL_VENDOR_INVENTORY_COLUMNS[10:] == expected_11_derived


def test_build_canonical_vendor_inventory_record_preserves_inputs():
    """Verify that all 10 supplied vendor input fields are preserved verbatim."""
    report, vendor = _build_test_report_and_vendor()
    record = build_canonical_vendor_inventory_record(report, vendor)

    assert record["Vendor ID"] == "V-001"
    assert record["Vendor Name"] == "AutomWorx"
    assert record["Vendor Description"] == vendor.vendor_description
    assert record["Service / Product Provided to Meridian"] == vendor.service_product_provided
    assert record["Category"] == "Professional Services"
    assert record["Public Website"] == "www.automworx.com"
    assert record["Business Process Supported"] == "Batch scheduling across core systems"
    assert record["Operational Dependency"] == "High"
    assert record["Data Classification Accessed"] == vendor.data_classification_accessed
    assert record["Data Volume (annual)"] == "Not applicable"


def test_ai_usage_detected_strict_three_state_mapping():
    """Verify that AI Usage Detected maps strictly to Yes, No, or Inconclusive."""
    # 1. YES
    rep_yes, v = _build_test_report_and_vendor(verdict=AIUseVerdictValue.YES)
    rec_yes = build_canonical_vendor_inventory_record(rep_yes, v)
    assert rec_yes["AI Usage Detected"] == "Yes"

    # 2. NO (with depth completed)
    rep_no, v = _build_test_report_and_vendor(verdict=AIUseVerdictValue.NO)
    rec_no = build_canonical_vendor_inventory_record(rep_no, v)
    assert rec_no["AI Usage Detected"] == "No"

    # 3. INCONCLUSIVE
    rep_inc, v = _build_test_report_and_vendor(verdict=AIUseVerdictValue.INCONCLUSIVE)
    rec_inc = build_canonical_vendor_inventory_record(rep_inc, v)
    assert rec_inc["AI Usage Detected"] == "Inconclusive"

    # 4. INCOMPLETE_ASSESSMENT
    rep_incomp, v = _build_test_report_and_vendor(verdict=AIUseVerdictValue.INCOMPLETE_ASSESSMENT)
    rec_incomp = build_canonical_vendor_inventory_record(rep_incomp, v)
    assert rec_incomp["AI Usage Detected"] == "Inconclusive"


def test_criticality_rationale_includes_factors_and_overrides():
    """Verify that Criticality Rationale synthesizes D, O, P, R, V factors and O1 safeguard."""
    report, vendor = _build_test_report_and_vendor()
    record = build_canonical_vendor_inventory_record(report, vendor)

    rat = record["Criticality Rationale"]
    assert "High Criticality Tier" in rat
    assert "Base scoring calculation yielded 1.3" in rat
    assert "Safeguard O1 triggered" in rat
    assert "Data Sensitivity D=3" in rat
    assert "Operational Dependency O=2" in rat


def test_evidence_description_includes_traceable_provenance():
    """Verify that Evidence Description contains exact excerpt, URL, dates, SHA-256, and strength."""
    report, vendor = _build_test_report_and_vendor()
    record = build_canonical_vendor_inventory_record(report, vendor)

    ev_desc = record["Evidence Description along with OSINT Source(s) / Evidence Link"]
    assert "EV_V001_001" in ev_desc
    assert "https://docs.automworx.com/admin/scheduler" in ev_desc
    assert "SHA-256: abc1234567890abcdef" in ev_desc
    assert "AutomWorx integrates deterministic rule-based scheduling" in ev_desc
    assert "Strength=STRONG" in ev_desc


def test_recommended_action_contains_ownership_deadlines_and_controls():
    """Verify that Recommended Action specifies TPRM ownership, deadlines, and governance controls."""
    report, vendor = _build_test_report_and_vendor()
    record = build_canonical_vendor_inventory_record(report, vendor)

    act = record["Recommended Action"]
    assert "Meridian Third-Party Risk Management" in act
    assert "Deadline:" in act
    assert "AI Data Processing Addendum (DPA)" in act
    assert "Zero Data Retention (ZDR)" in act


def test_to_inventory_csv_valid_rfc4180():
    """Verify that to_inventory_csv produces valid, parseable RFC 4180 CSV with all 21 columns."""
    report, vendor = _build_test_report_and_vendor()
    csv_str = AssessmentReportExporter.to_inventory_csv(report, vendor)

    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert list(row.keys()) == CANONICAL_VENDOR_INVENTORY_COLUMNS
    assert row["Vendor ID"] == "V-001"
    assert row["Vendor Name"] == "AutomWorx"
    assert row["AI Usage Detected"] == "Yes"
    assert "Meridian Third-Party AI Risk Governance Team" in row["Assessed By / Date"]


def test_to_batch_inventory_csv():
    """Verify that to_batch_inventory_csv exports multiple vendors cleanly."""
    rep1, v1 = _build_test_report_and_vendor(verdict=AIUseVerdictValue.YES)
    rep2, v2 = _build_test_report_and_vendor(verdict=AIUseVerdictValue.NO)
    rep2.vendor_id = "V-002"
    rep2.legal_entity = "Fiserv, Inc."
    v2.vendor_id = "V-002"
    v2.vendor_name = "Fiserv, Inc."

    batch_csv = AssessmentReportExporter.to_batch_inventory_csv([rep1, rep2], [v1, v2])
    reader = csv.DictReader(io.StringIO(batch_csv))
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["Vendor ID"] == "V-001"
    assert rows[0]["AI Usage Detected"] == "Yes"
    assert rows[1]["Vendor ID"] == "V-002"
    assert rows[1]["AI Usage Detected"] == "No"


def test_pipeline_instantiates_all_ten_input_fields_offline():
    """Verify that EndToEndAssessmentPipeline populates all 10 input fields on report."""
    loader = MeridianCSVVendorLoader()
    vendors = loader.load("data/input/meridian_vendors.csv")
    v001 = vendors[0]

    pipeline = MeridianEndToEndPipeline()
    report = pipeline.run_vendor_assessment(
        v001,
        execute_searches=False,
        execute_extraction=False,
        reuse_cached_artifacts=False,
        use_verified_local_artifacts_only=True,
    )

    assert report.vendor_id == "V-001"
    assert report.legal_entity == "AutomWorx"
    assert report.domain == "www.automworx.com"
    assert report.category == "Professional Services"
    assert report.vendor_description == v001.vendor_description
    assert report.service_product_provided == v001.service_product_provided
    assert report.business_process_supported == v001.business_process_supported
    assert report.data_classification_accessed == v001.data_classification_accessed
    assert report.operational_dependency == v001.operational_dependency
    assert report.data_volume_annual == v001.data_volume_annual

    # Check 21-column inventory conversion
    rec = AssessmentReportExporter.to_inventory_dict(report, v001)
    assert len(rec) == 21
    for col in CANONICAL_VENDOR_INVENTORY_COLUMNS:
        assert col in rec
        assert rec[col] is not None


def test_url_triage_blocks_court_dockets_and_bankruptcy_aggregators():
    """Verify that bankruptcy claims dockets and directory scrapers are deterministically rejected."""
    from meridian_assessment.models.osint_plan import CandidateURL, OSINTInvestigationPlan
    from meridian_assessment.services.osint.url_triage import OSINTURLTriageService, _is_blocked_unreliable_host

    assert _is_blocked_unreliable_host("cases.stretto.com")
    assert _is_blocked_unreliable_host("epiq11.com")
    assert _is_blocked_unreliable_host("rankiteo.com")
    assert not _is_blocked_unreliable_host("automworx.com")
    assert not _is_blocked_unreliable_host("github.com")

    triage_service = OSINTURLTriageService()
    cand = CandidateURL(
        candidate_id="CURL_TEST_01",
        url="https://cases.stretto.com/llflooring/dockets/1301908232480000000425.pdf",
        vendor_id="V-001",
        title="In re LL Flooring Holdings - Creditor Service Matrix including AutomWorx Inc.",
        snippet="AutomWorx Inc. listed on schedule of unsecured creditors.",
    )
    dec = triage_service.triage_candidate(
        cand,
        vendor_name="AutomWorx",
        domain="www.automworx.com",
    )
    assert dec.action.value == "REJECT"
    assert dec.product_scope_relevance == "UNRELATED"
    assert "untrusted court docket or directory aggregator" in dec.rationale


def test_entity_detection_avoids_substring_false_positives():
    """Verify that _detect_entity does not match substrings like 'laws' -> 'AWS' or 'titanium' -> 'Titan'."""
    from meridian_assessment.services.classification.signal_classifier import (
        SignalClassificationService,
        _KNOWN_AI_MODELS,
        _KNOWN_AI_PROVIDERS,
        _KNOWN_HOSTING_INFRA,
    )

    # 1. No false positive from English substrings
    non_ai_legal_text = "the court reviewed relevant state laws and lawsuits regarding titanium fixtures"
    p = SignalClassificationService._detect_entity(non_ai_legal_text, _KNOWN_AI_PROVIDERS)
    m = SignalClassificationService._detect_entity(non_ai_legal_text, _KNOWN_AI_MODELS)
    h = SignalClassificationService._detect_entity(non_ai_legal_text, _KNOWN_HOSTING_INFRA)
    assert p is None
    assert m is None
    assert h is None

    # 2. Genuine AI entities with context match correctly
    ai_text = "integrates Amazon Titan embedding foundation model deployed in AWS Bedrock"
    p2 = SignalClassificationService._detect_entity(ai_text, _KNOWN_AI_PROVIDERS)
    m2 = SignalClassificationService._detect_entity(ai_text, _KNOWN_AI_MODELS)
    h2 = SignalClassificationService._detect_entity(ai_text, _KNOWN_HOSTING_INFRA)
    assert p2 == "AWS Bedrock"
    assert m2 == "Amazon Titan"
    assert h2 == "Amazon Web Services (AWS)"


def test_automworx_non_ai_signals_do_not_populate_confirmed_caps_or_subprocessors():
    """Verify that non-AI automation signals are not presented as confirmed AI production capabilities."""
    from meridian_assessment.models.ai_risk import AIType, SignalCategory, SignalClassificationResult

    report, vendor = _build_test_report_and_vendor(verdict=AIUseVerdictValue.NO)
    report.signal_classifications = [
        SignalClassificationResult(
            classification_id="CLS_001",
            evidence_id="EV_V001_001",
            vendor_id="V-001",
            signal_category=SignalCategory.NON_AI_AUTOMATION,
            ai_type=AIType.NONE_AUTOMATION_ONLY,
            scope=EvidenceScope.VENDOR_PRODUCT,
            capability_summary="No operational AI capability or marketing AI claim detected in excerpt.",
            supporting_verbatim_text="Deterministic batch scheduling without ML/LLM.",
            confidence="HIGH",
            rationale="Batch scheduling only.",
        )
    ]
    report.named_providers = []
    report.named_models = []

    record = build_canonical_vendor_inventory_record(report, vendor)
    assert record["AI Usage Detected"] == "No"

    how_ai = record["How Vendor May Be Using AI"]
    assert "CONFIRMED AI USE CASES:\nNone evidenced in public technical documentation" in how_ai
    assert "Confirmed Production Capability: No operational AI" not in how_ai

    subproc = record["Named AI Sub-processors / Model Providers"]
    assert "Confirmed AI Sub-processors: None evidenced in public documentation." in subproc
    assert "Confirmed Foundation Models: None evidenced in public documentation." in subproc


def test_large_evidence_register_bounded_and_prioritized():
    """Verify that a large evidence register is prioritized and bounded to prevent megabyte-scale bloat."""
    report, vendor = _build_test_report_and_vendor()
    # Create 30 mock evidence records
    extra_records = []
    for i in range(1, 31):
        extra_records.append(
            EvidenceRecord(
                evidence_id=f"EV_V001_{i:03d}",
                vendor_id="V-001",
                legal_entity="AutomWorx",
                domain="www.automworx.com",
                source_class="S2",
                source_reliability_grade="B",
                canonical_url=f"https://docs.automworx.com/page_{i}",
                original_url=f"https://docs.automworx.com/page_{i}",
                retrieved_at="2026-10-13T10:00:00Z",
                content_sha256="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                verbatim_excerpt=f"Documentation passage number {i} detailing workload automation configuration.",
                scope=EvidenceScope.VENDOR_PRODUCT,
            )
        )
    report.evidence_records = extra_records

    record = build_canonical_vendor_inventory_record(report, vendor)
    ev_desc = record["Evidence Description along with OSINT Source(s) / Evidence Link"]

    # Must contain top records and an overflow summary note
    assert "EV_V001_001" in ev_desc
    assert "EV_V001_015" in ev_desc
    assert "[Evidentiary Register Note: 15 additional point-of-capture evidence items verified and retained in complete evidentiary ledger.]" in ev_desc
    # Must NOT contain all 30 detailed blocks
    assert "EV_V001_025" not in ev_desc

