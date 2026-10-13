"""Unit and UI integration tests for the Frontend, Analyst Workflow, Exports, and Pre-Search Readiness Pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.models.ai_risk import (
    AIUseVerdictValue,
    ScoringPolicyStatus,
)
from meridian_assessment.services.end_to_end_pipeline import MeridianEndToEndPipeline
from meridian_assessment.services.extraction.evidence_repository import EvidenceRepository
from meridian_assessment.services.offline_demo_fixtures import (
    build_synthetic_demo_adapters_for_vendor,
)
from meridian_assessment.services.readiness_and_exports import (
    AssessmentReportExporter,
    CANONICAL_NINE_LEVEL_NAMES,
    SystemReadinessInspector,
    build_nine_level_stage_tracker,
    validate_vendor_assessment_input,
)


# ─── 1. System Readiness Panel: Ollama Online + Both Models Installed ─────────


def test_system_readiness_ollama_online_and_both_models_installed() -> None:
    """Verify readiness report when Ollama is reachable and both llama3.2:latest and mistral:latest are installed."""

    def fake_tags_get(url: str, timeout: float) -> dict[str, Any]:
        assert url.endswith("/api/tags")
        return {
            "models": [
                {"name": "llama3.2:latest", "size": 2_000_000_000},
                {"name": "mistral:latest", "size": 4_400_000_000},
            ]
        }

    env = {
        "MERIDIAN_ENABLE_LOCAL_LLM": "true",
        "MERIDIAN_OLLAMA_ENDPOINT": "http://localhost:11434",
        "MERIDIAN_OLLAMA_TRIAGE_MODEL": "llama3.2:latest",
        "MERIDIAN_OLLAMA_ANALYSIS_MODEL": "mistral:latest",
        "MERIDIAN_OLLAMA_SUMMARY_MODEL": "mistral:latest",
        "MERIDIAN_OLLAMA_UNLOAD_ON_SWITCH": "true",
        "MERIDIAN_SEARCH_PROVIDER": "serpapi",
        "MERIDIAN_SERPAPI_KEY": "",
        "MERIDIAN_ENABLE_RAG_EMBEDDINGS": "false",
        "MERIDIAN_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    }
    report = SystemReadinessInspector.check_system_readiness(
        env=env,
        load_dotenv=False,
        http_get_fn=fake_tags_get,
    )

    assert report.ollama_reachable is True
    assert report.llama3_2_installed is True
    assert report.mistral_installed is True
    assert report.llm_routing_enabled_in_env is True
    assert report.triage_model == "llama3.2:latest"
    assert report.analysis_model == "mistral:latest"
    assert report.summary_model == "mistral:latest"
    assert report.unload_on_model_switch is True


# ─── 2. System Readiness Panel: Ollama Offline or Model Missing ───────────────


def test_system_readiness_ollama_offline_or_missing_model() -> None:
    """Verify readiness report when Ollama is unreachable or when mistral:latest is missing."""

    def offline_get(url: str, timeout: float) -> dict[str, Any]:
        raise ConnectionRefusedError("Connection refused")

    rep_offline = SystemReadinessInspector.check_system_readiness(
        env={"MERIDIAN_ENABLE_LOCAL_LLM": "true"},
        load_dotenv=False,
        http_get_fn=offline_get,
    )
    assert rep_offline.ollama_reachable is False
    assert rep_offline.llama3_2_installed is False
    assert rep_offline.mistral_installed is False
    assert "unreachable" in rep_offline.ollama_status_message.lower()

    def missing_mistral_get(url: str, timeout: float) -> dict[str, Any]:
        return {"models": [{"name": "llama3.2:latest"}]}

    rep_partial = SystemReadinessInspector.check_system_readiness(
        env={"MERIDIAN_ENABLE_LOCAL_LLM": "true"},
        load_dotenv=False,
        http_get_fn=missing_mistral_get,
    )
    assert rep_partial.ollama_reachable is True
    assert rep_partial.llama3_2_installed is True
    assert rep_partial.mistral_installed is False


# ─── 3. System Readiness Panel: Blank MERIDIAN_SERPAPI_KEY & Secret Safety ───


def test_system_readiness_blank_serpapi_key_shows_actionable_message_and_never_leaks_secret() -> None:
    """Verify blank MERIDIAN_SERPAPI_KEY produces actionable message and configured key never leaks secret value."""
    rep_blank = SystemReadinessInspector.check_system_readiness(
        env={
            "MERIDIAN_SEARCH_PROVIDER": "serpapi",
            "MERIDIAN_SERPAPI_KEY": "",
        },
        load_dotenv=False,
        http_get_fn=lambda u, t: {"models": []},
    )
    assert rep_blank.search_provider == "serpapi"
    assert rep_blank.search_configured is False
    assert (
        rep_blank.search_status_message
        == "Search is not configured: add MERIDIAN_SERPAPI_KEY to the local .env file."
    )

    secret_val = "SECRET_SERPAPI_TOKEN_DO_NOT_DISPLAY_999"
    rep_configured = SystemReadinessInspector.check_system_readiness(
        env={
            "MERIDIAN_SEARCH_PROVIDER": "serpapi",
            "MERIDIAN_SERPAPI_KEY": secret_val,
        },
        load_dotenv=False,
        http_get_fn=lambda u, t: {"models": []},
    )
    assert rep_configured.search_configured is True
    assert secret_val not in rep_configured.model_dump_json()


# ─── 4. Synthetic-Fixture / Demo Mode Renders All Nine Canonical Levels ──────


def test_synthetic_fixture_mode_renders_all_nine_canonical_levels_and_level_3b_substep(
    tmp_path: Path,
) -> None:
    """Verify running a vendor in Synthetic-Fixture Mode populates all 9 canonical project levels and Level 3B substep."""
    vendors = {v.vendor_id: v for v in MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))}
    fiserv = vendors["V-002"]

    stub_search, stub_fetch = build_synthetic_demo_adapters_for_vendor(fiserv)
    repo = EvidenceRepository(
        db_path=tmp_path / "synth_ui.sqlite3",
        artifact_dir=tmp_path / "synth_ui_artifacts",
    )
    progress_events: list[tuple[int, str, str]] = []

    pipeline = MeridianEndToEndPipeline(
        search_backend=stub_search,
        fetch_adapter=stub_fetch,
        evidence_repository=repo,
        enable_local_llm_from_env=False,
    )
    all_sources = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]
    report = pipeline.run_vendor_assessment(
        fiserv,
        execute_searches=True,
        execute_extraction=True,
        mark_extracted_sources_reviewed=True,
        reviewed_no_evidence_sources=all_sources,
        persist_to_repository=True,
        progress_callback=lambda lvl, title, detail: progress_events.append((lvl, title, detail)),
    )

    assert report.is_synthetic_evidence_run is True
    assert [lvl for lvl, _, _ in progress_events] == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    tracker_rows = build_nine_level_stage_tracker(report)
    assert len(tracker_rows) == 9
    for idx, row in enumerate(tracker_rows, start=1):
        assert row.level_number == idx
        assert row.level_name == CANONICAL_NINE_LEVEL_NAMES[idx]
        assert row.relevant_counts != ""
        assert row.execution_method != ""

    # Level 3 must present Level 3B as a substep of Level 3
    lvl3_row = tracker_rows[2]
    assert any(sub["substep_id"] == "Level 3B" for sub in lvl3_row.substeps)


# ─── 5. Real OSINT Mode with Missing Search Credentials -> INCOMPLETE_ASSESSMENT


def test_real_osint_mode_with_missing_search_credentials_shows_incomplete_assessment(
    tmp_path: Path,
) -> None:
    """Verify Real OSINT Mode with blank MERIDIAN_SERPER_API_KEY marks queries UNAVAILABLE, blocks Level 3, and yields INCOMPLETE_ASSESSMENT."""
    from meridian_assessment.services.osint import SerperSearchBackend

    vendors = {v.vendor_id: v for v in MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))}
    fiserv = vendors["V-002"]

    repo = EvidenceRepository(
        db_path=tmp_path / "real_unconf.sqlite3",
        artifact_dir=tmp_path / "real_unconf_artifacts",
    )
    pipeline = MeridianEndToEndPipeline(
        search_backend=SerperSearchBackend(api_key=""),
        evidence_repository=repo,
        enable_local_llm_from_env=False,
    )
    report = pipeline.run_vendor_assessment(
        fiserv,
        execute_searches=True,
        execute_extraction=True,
        persist_to_repository=True,
    )

    assert report.is_synthetic_evidence_run is False
    assert report.ai_use_verdict.verdict == AIUseVerdictValue.INCOMPLETE_ASSESSMENT
    assert report.ai_use_verdict.verdict != AIUseVerdictValue.NO
    assert report.osint_plan_summary["planned_queries_count"] > 0
    assert (
        report.osint_plan_summary["unavailable_queries_count"]
        == report.osint_plan_summary["planned_queries_count"]
    )

    tracker_rows = build_nine_level_stage_tracker(report)
    lvl3 = tracker_rows[2]
    assert lvl3.status == "NOT_CONFIGURED"
    assert lvl3.badge_label == "BLOCKED BY CONFIG"
    assert lvl3.blocker is not None


# ─── 6. Preserving UNKNOWN When Vendor Inputs Are Unknown ────────────────────


def test_validate_vendor_assessment_input_preserves_unknown_without_coercing_to_zero() -> None:
    """Verify validate_vendor_assessment_input preserves UNKNOWN and MeridianCriticalityEngine flags partial/review status."""
    vendor, errors, warnings = validate_vendor_assessment_input(
        vendor_id="V-099",
        vendor_name="Unknown Inputs Vendor",
        domain="https://unknownvendor.example.com/path",
        service_product_provided="",
        business_process_supported="UNKNOWN",
        data_classification_accessed="UNKNOWN",
        operational_dependency="UNKNOWN",
        data_volume_annual="",
        is_sole_source=False,
    )
    assert not errors
    assert vendor is not None
    assert vendor.domain == "unknownvendor.example.com"
    assert vendor.data_classification_accessed == "UNKNOWN"
    assert vendor.operational_dependency == "UNKNOWN"
    assert vendor.data_volume_annual == "UNKNOWN"
    assert len(warnings) >= 3

    from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine

    crit = MeridianCriticalityEngine().evaluate(vendor)
    assert crit.D.is_unknown is True
    assert crit.D.score is None
    assert crit.O.is_unknown is True
    assert crit.O.score is None
    assert crit.V.is_unknown is True
    assert crit.V.score is None
    assert crit.requires_review is True


# ─── 7 & 8. Displaying E1–E14 Evidence Records & R1–R8 Findings ──────────────


def test_evidence_records_and_risk_findings_preserve_provenance_and_separate_llm_explanation(
    tmp_path: Path,
) -> None:
    """Verify E1–E14 evidence provenance and R1–R8 deterministic fields vs LLM explanation separation."""
    vendors = {v.vendor_id: v for v in MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))}
    fiserv = vendors["V-002"]

    stub_search, stub_fetch = build_synthetic_demo_adapters_for_vendor(fiserv)
    pipeline = MeridianEndToEndPipeline(
        search_backend=stub_search,
        fetch_adapter=stub_fetch,
        evidence_repository=EvidenceRepository(
            db_path=tmp_path / "ev_rf.sqlite3",
            artifact_dir=tmp_path / "ev_rf_artifacts",
        ),
        enable_local_llm_from_env=False,
    )
    report = pipeline.run_vendor_assessment(
        fiserv,
        execute_searches=True,
        execute_extraction=True,
        mark_extracted_sources_reviewed=True,
        reviewed_no_evidence_sources=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"],
        persist_to_repository=True,
    )

    assert len(report.evidence_records) >= 2
    for ev in report.evidence_records:
        assert ev.canonical_url.startswith("https://")
        assert ev.source_class.startswith("S")
        assert ev.retrieved_at != ""
        assert len(ev.content_sha256) == 64
        assert ev.verbatim_excerpt != ""
        assert len(ev.e_rule_ids) >= 1

    assert len(report.risk_findings) >= 3
    for rf in report.risk_findings:
        assert rf.risk_rule_id.startswith("R")
        assert rf.severity.value in ("Low", "Medium", "High", "Critical")
        assert rf.finding_description != ""
        assert len(rf.evidence_ids) >= 1
        assert rf.recommended_remediation != ""
        assert rf.summary_method in ("DETERMINISTIC_TEMPLATE", "FALLBACK_DETERMINISTIC", "LOCAL_LLM_VALIDATED")


# ─── 9. Level 9 Displayed as SCORING_POLICY_UNCONFIGURED ─────────────────────


def test_level_9_displayed_as_scoring_policy_unconfigured_by_default(tmp_path: Path) -> None:
    """Verify Level 9 reports SCORING_POLICY_UNCONFIGURED and final_risk_score=None when production policy is absent."""
    readiness = SystemReadinessInspector.check_system_readiness(
        env={"MERIDIAN_ENABLE_LOCAL_LLM": "false"},
        load_dotenv=False,
        http_get_fn=lambda u, t: {"models": []},
    )
    assert readiness.scoring_policy_status == ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED.value
    assert readiness.scoring_policy_ready is False
    assert "SCORING_POLICY_UNCONFIGURED" in readiness.scoring_status_message


# ─── 10. Clean JSON, CSV, Markdown, and HTML Exports Without Secrets ─────────


def test_exports_json_csv_markdown_html_contain_required_sections_and_no_secrets(
    tmp_path: Path,
) -> None:
    """Verify AssessmentReportExporter produces complete JSON, CSV, Markdown, and HTML outputs with zero secrets."""
    vendors = {v.vendor_id: v for v in MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))}
    fiserv = vendors["V-002"]

    stub_search, stub_fetch = build_synthetic_demo_adapters_for_vendor(fiserv)
    pipeline = MeridianEndToEndPipeline(
        search_backend=stub_search,
        fetch_adapter=stub_fetch,
        evidence_repository=EvidenceRepository(
            db_path=tmp_path / "exp.sqlite3",
            artifact_dir=tmp_path / "exp_artifacts",
        ),
        enable_local_llm_from_env=False,
    )
    report = pipeline.run_vendor_assessment(
        fiserv,
        execute_searches=True,
        execute_extraction=True,
        mark_extracted_sources_reviewed=True,
        reviewed_no_evidence_sources=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"],
        persist_to_repository=True,
    )

    json_out = AssessmentReportExporter.to_json(report)
    csv_out = AssessmentReportExporter.to_evidence_csv(report)
    md_out = AssessmentReportExporter.to_markdown(report)
    html_out = AssessmentReportExporter.to_html(report)

    assert report.report_id in json_out
    assert "canonical_url" in csv_out
    assert "https://trust.www.fiserv.com/subprocessors-and-ai-policy" in csv_out
    assert report.report_id in md_out
    assert "SYNTHETIC FIXTURE / DEMO WALKTHROUGH" in md_out
    assert "SCORING_POLICY_UNCONFIGURED" in md_out
    assert "Level 3B" in md_out
    assert "<html" in html_out
    assert report.report_id in html_out
    assert "https://trust.www.fiserv.com/subprocessors-and-ai-policy" in html_out


from meridian_assessment.services.classification import (
    OllamaClassificationAdapter,
    UnconfiguredOllamaAdapter,
)
from meridian_assessment.services.readiness_and_exports import (
    AssessmentReportExporter,
    CANONICAL_NINE_LEVEL_NAMES,
    SystemReadinessInspector,
    build_nine_level_stage_tracker,
    format_analyst_stage_tracker,
    format_analyst_unavailable_integrations,
    format_plain_status,
    validate_vendor_assessment_input,
)


def test_no_contradictory_llm_status_between_readiness_and_pipeline() -> None:
    """Verify SystemReadinessInspector and MeridianEndToEndPipeline never produce contradictory Ollama/Mistral status claims."""

    def online_both_models(url: str, timeout: float) -> dict[str, Any]:
        return {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:latest"},
            ]
        }

    base_env = {
        "MERIDIAN_OLLAMA_ENABLED": "true",
        "MERIDIAN_ENABLE_LOCAL_LLM": "true",
        "MERIDIAN_OLLAMA_ENDPOINT": "http://localhost:11434",
        "MERIDIAN_OLLAMA_TRIAGE_MODEL": "llama3.2:latest",
        "MERIDIAN_OLLAMA_ANALYSIS_MODEL": "mistral:latest",
        "MERIDIAN_OLLAMA_SUMMARY_MODEL": "mistral:latest",
    }

    # Case A: Server online + both models installed + enabled -> Ready in both readiness & pipeline
    rep_ready = SystemReadinessInspector.check_system_readiness(
        env=base_env,
        load_dotenv=False,
        http_get_fn=online_both_models,
    )
    assert rep_ready.local_analysis_ready is True
    assert rep_ready.local_triage_ready is True
    assert rep_ready.local_classifier_ready is True
    assert rep_ready.local_summary_ready is True
    assert rep_ready.local_analysis_state == "READY"
    assert rep_ready.local_analysis_label == "Local AI analysis ready"

    pipe_ready = MeridianEndToEndPipeline(
        enable_local_llm_from_env=True,
        readiness=rep_ready,
    )
    assert pipe_ready.model_router.config.enabled is True
    assert isinstance(pipe_ready.classification_service.llm_adapter, OllamaClassificationAdapter)
    cls_ready, cls_msg = pipe_ready.classification_service.llm_adapter.is_configured()
    assert cls_ready is True
    assert "not configured" not in cls_msg.lower()

    # Case B: Analyst explicitly selects deterministic / rule-based run -> Consistent disabled reason
    rep_user_off = SystemReadinessInspector.check_system_readiness(
        env=base_env,
        load_dotenv=False,
        http_get_fn=online_both_models,
        local_analysis_override=False,
    )
    assert rep_user_off.local_analysis_ready is False
    assert rep_user_off.local_analysis_state == "DISABLED_BY_USER"
    assert rep_user_off.local_analysis_label == "Rule-based analysis only"

    pipe_user_off = MeridianEndToEndPipeline(
        enable_local_llm_from_env=False,
        readiness=rep_user_off,
    )
    assert pipe_user_off.model_router.config.enabled is False
    assert isinstance(pipe_user_off.classification_service.llm_adapter, UnconfiguredOllamaAdapter)
    _, off_msg = pipe_user_off.classification_service.llm_adapter.is_configured()
    assert "disabled for deterministic run" in off_msg.lower()
    assert "set meridian_ollama_enabled=true" not in off_msg.lower()

    # Case C: Enabled in .env, but Ollama server unreachable -> Both report unavailable
    def offline_get(url: str, timeout: float) -> dict[str, Any]:
        raise ConnectionRefusedError("Ollama offline")

    rep_offline = SystemReadinessInspector.check_system_readiness(
        env=base_env,
        load_dotenv=False,
        http_get_fn=offline_get,
    )
    assert rep_offline.local_analysis_ready is False
    assert rep_offline.local_analysis_state == "SERVER_UNREACHABLE"
    assert rep_offline.local_analysis_label == "Local analysis unavailable"

    pipe_offline = MeridianEndToEndPipeline(
        enable_local_llm_from_env=True,
        readiness=rep_offline,
    )
    assert pipe_offline.model_router.config.enabled is False
    _, offline_msg = pipe_offline.classification_service.llm_adapter.is_configured()
    assert "unreachable" in offline_msg.lower()

    # Case D: Ollama reachable & llama3.2 installed, but mistral missing -> Partial/Missing model
    rep_no_mistral = SystemReadinessInspector.check_system_readiness(
        env=base_env,
        load_dotenv=False,
        http_get_fn=lambda u, t: {"models": [{"name": "llama3.2:latest"}]},
    )
    assert rep_no_mistral.local_triage_ready is True
    assert rep_no_mistral.local_classifier_ready is False
    assert rep_no_mistral.local_analysis_ready is False
    assert rep_no_mistral.local_analysis_state == "MODEL_MISSING"

    pipe_no_mistral = MeridianEndToEndPipeline(
        enable_local_llm_from_env=True,
        readiness=rep_no_mistral,
    )
    assert pipe_no_mistral.triage_service.llm_provider is not None
    assert isinstance(pipe_no_mistral.classification_service.llm_adapter, UnconfiguredOllamaAdapter)
    _, no_mistral_msg = pipe_no_mistral.classification_service.llm_adapter.is_configured()
    assert "not installed" in no_mistral_msg.lower()


def test_plain_language_analyst_formatting_and_level_7_evidentiary_weighting(
    tmp_path: Path,
) -> None:
    """Verify analyst stage tracker and notices use plain product language and preserve Level 7 as Evidentiary Weighting."""
    assert CANONICAL_NINE_LEVEL_NAMES[7] == "Evidentiary Weighting"
    assert format_plain_status("SCORING_POLICY_UNCONFIGURED") == "Calibrated numerical risk score unavailable"
    assert format_plain_status("INCOMPLETE_ASSESSMENT") == "Incomplete Assessment"
    assert format_plain_status("QUALIFYING_POSITIVE_EVIDENCE") == "Confirmed Positive Evidence"

    vendors = {v.vendor_id: v for v in MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))}
    fiserv = vendors["V-002"]
    stub_search, stub_fetch = build_synthetic_demo_adapters_for_vendor(fiserv)

    pipeline = MeridianEndToEndPipeline(
        search_backend=stub_search,
        fetch_adapter=stub_fetch,
        evidence_repository=EvidenceRepository(
            db_path=tmp_path / "plain.sqlite3",
            artifact_dir=tmp_path / "plain_artifacts",
        ),
        enable_local_llm_from_env=False,
    )
    report = pipeline.run_vendor_assessment(
        fiserv,
        execute_searches=True,
        execute_extraction=True,
        mark_extracted_sources_reviewed=True,
        reviewed_no_evidence_sources=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"],
        persist_to_repository=True,
    )

    analyst_stages = format_analyst_stage_tracker(report)
    assert len(analyst_stages) == 9
    assert analyst_stages[6]["Stage"] == "7"
    assert analyst_stages[6]["Assessment Stage"] == "Evidentiary Weighting"

    forbidden_tokens = [
        "MERIDIAN_SERPAPI_KEY",
        "MERIDIAN_OLLAMA_ENABLED",
        "MERIDIAN_ENABLE_RAG_EMBEDDINGS",
        "llama3.2:latest",
        "mistral:latest",
        "sentence-transformers",
        "SCORING_POLICY_UNCONFIGURED",
        "StubSearchBackend",
        "StubFetchAdapter",
    ]
    serialized_stages = str(analyst_stages)
    for tok in forbidden_tokens:
        assert tok not in serialized_stages

    notices = format_analyst_unavailable_integrations(report)
    assert "Calibrated numerical risk score unavailable" in notices
    for tok in forbidden_tokens:
        assert tok not in str(notices)


def test_streamlit_app_renders_cleanly_without_exceptions() -> None:
    """Smoke-test the full Streamlit application (`app.py`) across Portfolio, Assessments, Evidence, Reports, and Settings."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parents[2] / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert not at.exception

    # Navigate across all 5 top-level sections via top navigation buttons and verify zero exceptions
    for nav_section in ["Assessments", "Evidence", "Reports", "Settings", "Portfolio"]:
        at.button(key=f"uc_top_nav_{nav_section}").click()
        at.run(timeout=30)
        assert not at.exception
        assert at.session_state["nav_section"] == nav_section

    # Trigger sample walkthrough on Assessments and verify Evidence and Reports render populated views cleanly
    at.button(key="uc_top_nav_Assessments").click()
    at.run(timeout=30)
    assert not at.exception
    at.selectbox(key="assess_execution_mode_select").set_value(
        "Sample Assessment (Pre-Collected Demonstration Fixtures)"
    )
    for ck_key in ["assess_ai_l3b", "assess_ai_l5", "assess_ai_l6", "assess_ai_l8"]:
        at.checkbox(key=ck_key).set_value(False)
    at.run(timeout=30)
    assert not at.exception
    at.button(key="btn_execute_assessment_primary").click()
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["e2e_report"] is not None

    for nav_section in ["Evidence", "Reports", "Portfolio"]:
        at.button(key=f"uc_top_nav_{nav_section}").click()
        at.run(timeout=30)
        assert not at.exception
