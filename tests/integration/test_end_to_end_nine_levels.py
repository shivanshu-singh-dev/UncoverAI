"""Integration tests for the complete 9-level Meridian End-to-End AI Risk Pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.models.ai_risk import (
    AIUseVerdictValue,
    ScoringPolicyStatus,
)
from meridian_assessment.services.end_to_end_pipeline import (
    HumanValidationService,
    MeridianEndToEndPipeline,
)
from meridian_assessment.services.extraction.evidence_repository import EvidenceRepository
from meridian_assessment.services.offline_demo_fixtures import (
    build_synthetic_demo_adapters_for_vendor,
)
from meridian_assessment.services.rag.chunker_retriever import (
    DeterministicFakeEmbeddingProvider,
)
from meridian_assessment.services.risk.deterministic_scorer import (
    DeterministicRiskScoringEngine,
)


def test_end_to_end_all_six_vendors_offline_unconfigured_mode(tmp_path: Path) -> None:
    """Verify all 6 Meridian vendors execute all 9 levels cleanly in default unconfigured offline mode."""
    vendors = MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))
    assert len(vendors) == 6

    repo = EvidenceRepository(
        db_path=tmp_path / "unconf_evidence.sqlite3",
        artifact_dir=tmp_path / "unconf_artifacts",
    )
    pipeline = MeridianEndToEndPipeline(
        evidence_repository=repo,
    )

    expected_criticality = {
        "V-001": "High",
        "V-002": "High",
        "V-003": "High",
        "V-004": "Low",
        "V-005": "High",
        "V-006": "Critical",
    }

    for v in vendors:
        report = pipeline.run_vendor_assessment(
            v,
            execute_searches=False,
            execute_extraction=False,
            persist_to_repository=True,
        )
        assert report.vendor_id == v.vendor_id
        assert report.criticality_summary["final_criticality"] == expected_criticality[v.vendor_id]
        # Must record all 9 levels in the stage ledger
        assert [s.level_number for s in report.stage_ledger] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # In unconfigured search mode, AI-use verdict must be INCOMPLETE_ASSESSMENT (not false-negative NO)
        assert report.ai_use_verdict.verdict == AIUseVerdictValue.INCOMPLETE_ASSESSMENT
        # Level 9 must report SCORING_POLICY_UNCONFIGURED
        assert (
            report.deterministic_risk_score.policy_status
            == ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED
        )
        assert report.deterministic_risk_score.final_risk_score is None
        assert len(report.unavailable_integrations) >= 1


def test_end_to_end_all_six_vendors_with_synthetic_test_fixtures(tmp_path: Path) -> None:
    """Verify all 6 Meridian vendors execute all 9 levels with labelled offline synthetic fixtures."""
    vendors = MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))
    repo = EvidenceRepository(
        db_path=tmp_path / "synth_evidence.sqlite3",
        artifact_dir=tmp_path / "synth_artifacts",
    )
    test_scorer = DeterministicRiskScoringEngine(
        Path("tests/fixtures/test_only_ai_risk_scoring_policy.yaml")
    )

    reports_by_id = {}
    for v in vendors:
        stub_search, stub_fetch = build_synthetic_demo_adapters_for_vendor(v)
        pipeline = MeridianEndToEndPipeline(
            search_backend=stub_search,
            fetch_adapter=stub_fetch,
            evidence_repository=repo,
            embedding_provider=DeterministicFakeEmbeddingProvider(),
            scoring_engine=test_scorer,
        )
        # Mark all planned required source classes as reviewed when running full synthetic walkthrough
        all_source_ids = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]
        rep = pipeline.run_vendor_assessment(
            v,
            execute_searches=True,
            execute_extraction=True,
            mark_extracted_sources_reviewed=True,
            reviewed_no_evidence_sources=all_source_ids,
            persist_to_repository=True,
        )
        reports_by_id[v.vendor_id] = rep
        assert [s.level_number for s in rep.stage_ledger] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert len(rep.url_triage_decisions) >= 2
        assert rep.extracted_documents_count >= 2
        assert len(rep.evidence_records) >= 1

    # V-001 (AutomWorx synthetic fixture): deterministic cron/batch scheduling only -> NO AI verdict
    v1_rep = reports_by_id["V-001"]
    assert v1_rep.ai_use_verdict.verdict == AIUseVerdictValue.NO

    # V-002 (Fiserv synthetic fixture): S1 subprocessor + S2 API docs in service scope -> YES AI verdict + R1-R8 findings + test score
    v2_rep = reports_by_id["V-002"]
    assert v2_rep.ai_use_verdict.verdict == AIUseVerdictValue.YES
    assert len(v2_rep.risk_findings) >= 3
    assert (
        v2_rep.deterministic_risk_score.policy_status
        == ScoringPolicyStatus.TEST_FIXTURE_ONLY
    )
    assert v2_rep.deterministic_risk_score.final_risk_score is not None
    assert v2_rep.deterministic_risk_score.final_risk_score > 0.0

    # Test HumanValidationService audit recording and rationale enforcement
    with pytest.raises(ValueError):
        HumanValidationService.create_validation_record(
            vendor_id="V-002",
            target_type="VERDICT",
            target_id="AI_USE_VERDICT",
            proposed_automated_finding="YES",
            human_decision="OVERRIDE_AI_VERDICT",
            rationale="   ",
            original_status="YES",
            updated_status="INCONCLUSIVE",
        )

    hv_rec = HumanValidationService.create_validation_record(
        vendor_id="V-002",
        target_type="VERDICT",
        target_id="AI_USE_VERDICT",
        proposed_automated_finding="YES",
        human_decision="CONFIRM_ASSESSMENT",
        rationale="Reviewed synthetic S1 subprocessor and S2 API documentation fixtures.",
        original_status="YES",
        updated_status="YES",
        reviewer_id="Senior TPRM Reviewer",
    )
    repo.save_human_validation(hv_rec)
    assert hv_rec.validation_id.startswith("HVAL_V-002")
