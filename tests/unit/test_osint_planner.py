"""Unit tests for the Meridian OSINT Depth Engine, Canonical S1–S12 Taxonomy, and Active Effort Timebox Tracking."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.models.factor_result import CriticalityLevel, HumanReviewRecord
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.models.osint_plan import (
    AssessmentStopStatus,
    CoverageState,
    CriticalityApprovalStatus,
    GateChecklistItem,
    GateStatus,
    SearchExecutionStatus,
    SourceCoverageRecord,
    TimeboxBudget,
    TimerState,
)
from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine
from meridian_assessment.services.osint.coverage_tracker import OSINTCoverageTracker
from meridian_assessment.services.osint.investigation_planner import OSINTInvestigationPlanner
from meridian_assessment.services.osint.query_planner import OSINTQueryPlanner
from meridian_assessment.services.osint.source_registry import OSINTSourceRegistry
from meridian_assessment.services.osint.stop_conditions import StopConditionEvaluator


@pytest.fixture
def sample_vendor() -> MeridianVendor:
    return MeridianVendor(
        vendor_id="V-001",
        vendor_name="AutomWorx",
        domain="automworx.com",
        service_product_provided="Automated batch job scheduling and execution",
        business_process_supported="IT operations / batch processing workflow",
        data_classification_accessed="Production service credentials and system access tokens",
        operational_dependency="High",
        data_volume_annual="N/A",
    )


@pytest.fixture
def planner() -> OSINTInvestigationPlanner:
    return OSINTInvestigationPlanner()


# ─── 1. Unified Depth Profiles & Simplified POC Timeboxes ────────────────────

class TestDepthSelectionAndTimeboxes:
    def test_critical_selects_critical_profile_and_30m_timebox(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        plan = planner.plan_investigation(sample_vendor, "Critical")
        assert plan.selected_depth_profile == "Critical"
        assert len(plan.required_source_classes) == 12  # S1-S12
        assert set(plan.required_source_classes) == {f"S{i}" for i in range(1, 13)}
        assert plan.conditional_source_classes == []
        assert set(plan.corroborative_source_classes) == {"S9", "S10", "S11"}
        assert plan.reviewer_requirements.get("second_reviewer") is True
        # POC Timebox: 30 minutes
        assert plan.timebox.planning_target == "30 minutes"
        assert plan.timebox.planned_minutes == 30
        assert plan.timebox.elapsed_wall_clock_minutes == 0.0
        assert plan.timebox.actual_analyst_effort_minutes == 0.0
        assert plan.timebox.timer_state == TimerState.NOT_STARTED

    def test_high_selects_high_profile_and_20m_timebox(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        plan = planner.plan_investigation(sample_vendor, "High")
        assert plan.selected_depth_profile == "High"
        assert set(plan.required_source_classes) == {"S1", "S2", "S3", "S5", "S6", "S7", "S8"}
        cond_ids = {c["source_id"] for c in plan.conditional_source_classes}
        assert cond_ids == {"S4", "S9", "S11"}
        assert set(plan.corroborative_source_classes) == {"S10", "S12"}
        assert plan.reviewer_requirements.get("analyst_validation") is True
        assert plan.reviewer_requirements.get("second_reviewer") is False
        # POC Timebox: 20 minutes
        assert plan.timebox.planning_target == "20 minutes"
        assert plan.timebox.planned_minutes == 20

    def test_medium_selects_medium_profile_and_10m_timebox(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        plan = planner.plan_investigation(sample_vendor, "Medium")
        assert plan.selected_depth_profile == "Medium"
        assert set(plan.required_source_classes) == {"S1", "S2", "S3", "S6"}
        cond_ids = {c["source_id"] for c in plan.conditional_source_classes}
        assert cond_ids == {"S4", "S9"}
        assert set(plan.corroborative_source_classes) == {"S12"}
        # POC Timebox: 10 minutes
        assert plan.timebox.planning_target == "10 minutes"
        assert plan.timebox.planned_minutes == 10

    def test_low_selects_low_profile_and_5m_timebox(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        plan = planner.plan_investigation(sample_vendor, "Low")
        assert plan.selected_depth_profile == "Low"
        assert set(plan.required_source_classes) == {"S1", "S3", "S12"}
        cond_ids = {c["source_id"] for c in plan.conditional_source_classes}
        assert cond_ids == {"S2"}
        assert plan.corroborative_source_classes == []
        # POC Timebox: 5 minutes
        assert plan.timebox.planning_target == "5 minutes"
        assert plan.timebox.planned_minutes == 5

    def test_o5_human_review_changes_criticality_preserving_evidence_and_timer_history(
        self, planner: OSINTInvestigationPlanner
    ):
        loader = MeridianCSVVendorLoader()
        vendors = loader.load(Path("data/input/meridian_vendors.csv"))
        engine = MeridianCriticalityEngine()

        # Terrapin (V-004) starts as Low (5-minute timebox)
        terrapin = next(v for v in vendors if v.vendor_id == "V-004")
        crit_res = engine.evaluate(terrapin)
        initial_plan = planner.plan_investigation(terrapin, criticality_input=crit_res)
        assert initial_plan.selected_depth_profile == "Low"
        assert initial_plan.timebox.planned_minutes == 5
        assert initial_plan.approval_status == CriticalityApprovalStatus.PROPOSED_UNREVIEWED

        # Record S1 evidence, G1 gate status, and 6 minutes of active analyst effort (exhausting the 5m Low timebox)
        t0 = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)
        OSINTCoverageTracker.start_timer(initial_plan, now=t0)
        OSINTCoverageTracker.pause_timer(initial_plan, now=t0 + timedelta(minutes=6))
        tracker = OSINTCoverageTracker(initial_plan.coverage_records)
        tracker.record_evidence_found(
            "S1",
            evidence_references=["https://terrapin.com/trust/dpa"],
            notes="DPA confirms AI portfolio analytics subprocessor.",
        )
        tracker.sync_plan_stop_status(initial_plan)
        OSINTCoverageTracker.update_plan_gate(
            initial_plan, "G1", GateStatus.QUALIFYING_POSITIVE_EVIDENCE, evidence_notes="Confirmed in S1 DPA"
        )
        assert initial_plan.timebox.actual_analyst_effort_minutes == pytest.approx(6.0)
        assert initial_plan.stop_status == AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE

        # Analyst applies O5 human review override -> High (20-minute timebox), passing existing_plan
        o5_override = HumanReviewRecord(
            user_decision="CHANGE_CRITICALITY",
            original_criticality="Low",
            final_criticality="High",
            rationale="O5 escalation due to wealth portfolio exposure.",
        )
        updated_plan = planner.plan_investigation(
            terrapin,
            criticality_input=crit_res,
            human_review=o5_override,
            existing_plan=initial_plan,
        )
        assert updated_plan.approval_status == CriticalityApprovalStatus.APPROVED_BY_ANALYST
        assert updated_plan.input_criticality == "High"
        assert updated_plan.selected_depth_profile == "High"
        assert updated_plan.timebox.planning_target == "20 minutes"
        assert updated_plan.timebox.planned_minutes == 20
        assert set(updated_plan.required_source_classes) == {"S1", "S2", "S3", "S5", "S6", "S7", "S8"}

        # Verify S1 evidence, G1 status, and 6.0m timer history were preserved without being discarded!
        assert updated_plan.coverage_records["S1"].status == CoverageState.REVIEWED_EVIDENCE_RECORDED
        assert updated_plan.coverage_records["S1"].evidence_references == ["https://terrapin.com/trust/dpa"]
        assert next(g for g in updated_plan.gate_checklist if g.gate_id == "G1").status == GateStatus.QUALIFYING_POSITIVE_EVIDENCE
        assert updated_plan.timebox.actual_analyst_effort_minutes == pytest.approx(6.0)
        assert updated_plan.timebox.timer_state == TimerState.PAUSED
        # Under the new 20m High budget, 6.0m is no longer exhausted!
        assert updated_plan.timebox.is_exhausted is False
        assert updated_plan.stop_status == AssessmentStopStatus.IN_PROGRESS

    def test_representative_dataset_plans_across_all_depths(
        self, planner: OSINTInvestigationPlanner
    ):
        loader = MeridianCSVVendorLoader()
        vendors = {v.vendor_id: v for v in loader.load(Path("data/input/meridian_vendors.csv"))}
        engine = MeridianCriticalityEngine()

        # Critical: V-006 The Clearing House (2.70 -> Critical)
        tch_res = engine.evaluate(vendors["V-006"])
        tch_plan = planner.plan_investigation(vendors["V-006"], criticality_input=tch_res)
        assert tch_plan.selected_depth_profile == "Critical"
        assert tch_plan.timebox.planned_minutes == 30
        assert len(tch_plan.required_source_classes) == 12

        # High: V-003 FSSI (2.00 -> High)
        fssi_res = engine.evaluate(vendors["V-003"])
        fssi_plan = planner.plan_investigation(vendors["V-003"], criticality_input=fssi_res)
        assert fssi_plan.selected_depth_profile == "High"
        assert fssi_plan.timebox.planned_minutes == 20

        # Low: V-004 Terrapin (1.00 partial -> Low)
        terrapin_res = engine.evaluate(vendors["V-004"])
        terrapin_plan = planner.plan_investigation(vendors["V-004"], criticality_input=terrapin_res)
        assert terrapin_plan.selected_depth_profile == "Low"
        assert terrapin_plan.timebox.planned_minutes == 5


# ─── 2. Canonical S1–S12 Taxonomy & Resource Mapping Tests ───────────────────

class TestCanonicalSourceTaxonomyAndResourceMappings:
    def test_canonical_s1_to_s12_definitions(self):
        reg = OSINTSourceRegistry()
        sources = reg.get_all_sources()
        assert len(sources) == 12

        # Verify canonical meanings for all S1-S12 classes
        assert "trust centers" in sources["S1"].description.lower()
        assert "subprocessor" in sources["S1"].description.lower()
        assert "dpas" in sources["S1"].description.lower()

        assert "product documentation" in sources["S2"].description.lower()
        assert "api references" in sources["S2"].description.lower()
        assert "sdks" in sources["S2"].description.lower()
        assert "public repositories" in sources["S2"].description.lower()

        assert "terms" in sources["S3"].description.lower()
        assert "privacy policies" in sources["S3"].description.lower()
        assert "security whitepapers" in sources["S3"].description.lower()

        # S4 is Engineering blogs, technical talks, open-source technical publications, and model cards
        assert "engineering blogs" in sources["S4"].description.lower()
        assert "technical talks" in sources["S4"].description.lower()
        assert "open-source technical publications" in sources["S4"].description.lower()
        assert "model cards" in sources["S4"].description.lower()
        assert "developer forums" not in sources["S4"].discovery_mode.lower()

        assert "filings" in sources["S5"].description.lower()
        assert "annual reports" in sources["S5"].description.lower()
        assert "certifications" in sources["S5"].description.lower()

        assert "job postings" in sources["S6"].description.lower()
        assert "hiring signals" in sources["S6"].description.lower()

        assert "executive statements" in sources["S7"].description.lower()
        assert "investor relations" in sources["S7"].description.lower()

        assert "marketplace" in sources["S8"].description.lower()
        assert "partner listings" in sources["S8"].description.lower()

        # S9 is Passive technical fingerprints (NOT developer forums)
        assert "passive technical fingerprints" in sources["S9"].name.lower()
        assert "passive technical fingerprints" in sources["S9"].description.lower()

        assert "patents" in sources["S10"].description.lower()
        assert "research papers" in sources["S10"].description.lower()

        assert "third-party reporting" in sources["S11"].description.lower()
        assert "news" in sources["S11"].description.lower()

        assert "marketing pages" in sources["S12"].description.lower()
        assert "press releases" in sources["S12"].description.lower()

    def test_resource_to_source_class_mappings_s2_s4_s9_s11(self):
        reg = OSINTSourceRegistry()
        sources = reg.get_all_sources()
        resources = reg.get_all_resources()
        assert len(resources) >= 20

        # GitHub and GitLab support S2 (public repos/SDKs), S4 (open-source technical publications), and S9 (passive fingerprints)
        assert set(resources["github"].associated_sources) == {"S2", "S4", "S9"}
        assert set(resources["gitlab"].associated_sources) == {"S2", "S4", "S9"}

        # Package registries (npm, PyPI, Maven Central) support both S2 (published vendor SDKs) and S9 (passive dependency manifests)
        for pkg_res in ("npm", "pypi", "maven_central"):
            assert set(resources[pkg_res].associated_sources) == {"S2", "S9"}
            assert pkg_res in sources["S2"].discovery_resources
            assert pkg_res in sources["S9"].discovery_resources

        # Docker Hub supports S9 (passive container image manifests)
        assert resources["docker_hub"].associated_sources == ["S9"]
        assert "docker_hub" in sources["S9"].discovery_resources

        # Hugging Face supports S4 (model cards / technical publications) and S9 (passive footprint)
        assert set(resources["hugging_face"].associated_sources) == {"S4", "S9"}
        assert "hugging_face" in sources["S4"].discovery_resources
        assert "hugging_face" in sources["S9"].discovery_resources

        # Reddit and Stack Overflow support S11 (third-party reporting / community discussions), NOT S4 or S9
        assert resources["reddit"].associated_sources == ["S11"]
        assert resources["stack_overflow"].associated_sources == ["S11"]
        assert "reddit" not in sources["S4"].discovery_resources
        assert "stack_overflow" not in sources["S4"].discovery_resources
        assert "q_reddit_ai" not in sources["S4"].query_template_ids
        assert "q_reddit_ai" in sources["S11"].query_template_ids

        # Bidirectional consistency: every resource's associated_sources matches its source's discovery_resources
        for r_id, res in resources.items():
            for s_id in res.associated_sources:
                assert r_id in sources[s_id].discovery_resources, f"{r_id} missing from {s_id}.discovery_resources"
        for s_id, src in sources.items():
            for r_id in src.discovery_resources:
                assert r_id in resources, f"{r_id} in {s_id} is not a registered resource"
                assert s_id in resources[r_id].associated_sources, f"{s_id} missing from {r_id}.associated_sources"

    def test_unified_policy_has_no_competing_policy_conflict_warnings(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        plan = planner.plan_investigation(sample_vendor, "Low")
        assert not hasattr(plan, "policy_conflicts")
        assert "S2" in plan.coverage_records
        assert plan.coverage_records["S2"].is_conditional is True
        assert plan.coverage_records["S2"].is_required is False
        assert "AI features" in plan.coverage_records["S2"].condition


# ─── 3. Query Generation & Missing Domain Handling ───────────────────────────

class TestQueryGeneration:
    def test_placeholder_substitution(self):
        qp = OSINTQueryPlanner()
        queries = qp.generate_queries_for_vendor(
            vendor_id="V-001",
            vendor_name="AutomWorx, Inc.",
            domain="https://www.automworx.com/portal",
            active_source_classes=["S1", "S2", "S3", "S6", "S12"],
        )
        assert len(queries) > 0
        rendered_texts = [q.rendered_query for q in queries]
        assert any('"AutomWorx" AI' in r for r in rendered_texts)
        assert any("site:automworx.com" in r for r in rendered_texts)
        assert not any("https://" in r for r in rendered_texts)

    def test_missing_domain_suppresses_vendor_domain_templates_without_broken_syntax(
        self, planner: OSINTInvestigationPlanner
    ):
        no_domain_vendor = MeridianVendor(
            vendor_id="V-999",
            vendor_name="NoDomain Analytics Corp.",
            domain="",
            service_product_provided="Portfolio analytics",
        )
        plan = planner.plan_investigation(no_domain_vendor, "Medium")
        assert len(plan.planned_queries) > 0
        assert any("domain" in issue.lower() for issue in plan.unresolved_planning_issues)
        for q in plan.planned_queries:
            assert "{domain}" not in q.rendered_query
            assert "site: " not in q.rendered_query
            assert not q.rendered_query.startswith("site: ")
            assert q.execution_status == SearchExecutionStatus.PLANNED

    def test_queries_deduplicated_preserving_source_associations(self):
        qp = OSINTQueryPlanner()
        queries = qp.generate_queries_for_vendor(
            vendor_id="V-002",
            vendor_name="Fiserv",
            domain="fiserv.com",
            active_source_classes=[f"S{i}" for i in range(1, 13)],
        )
        rendered_lower = [q.rendered_query.lower() for q in queries]
        assert len(rendered_lower) == len(set(rendered_lower))
        for q in queries:
            assert q.execution_status == SearchExecutionStatus.PLANNED

    def test_quotes_and_special_characters_handled_safely(self):
        qp = OSINTQueryPlanner()
        clean = qp.clean_vendor_name('Acme "AI" Solutions, LLC.')
        assert '"' not in clean
        assert clean.strip() == "Acme AI Solutions"


# ─── 4. Source Coverage Workflow & Evidence Integrity Tests ──────────────────

class TestCoverageTrackerAndWorkflow:
    def test_all_coverage_outcomes_and_evidence_references(self):
        records = {
            "S1": SourceCoverageRecord(source_id="S1", source_name="Trust Centers", is_required=True),
            "S2": SourceCoverageRecord(source_id="S2", source_name="Product Docs", is_required=True),
            "S3": SourceCoverageRecord(source_id="S3", source_name="Privacy Terms", is_required=True),
            "S5": SourceCoverageRecord(source_id="S5", source_name="Subprocessors", is_required=True),
            "S6": SourceCoverageRecord(source_id="S6", source_name="Job Postings", is_required=True),
        }
        tracker = OSINTCoverageTracker(records)

        tracker.record_evidence_found(
            "S1",
            evidence_references=["https://vendor.com/trust/soc2", "EV-101"],
            candidate_urls=4,
            reviewed_urls=4,
            produced_new_evidence=True,
            notes="SOC 2 Type II report confirms AWS Bedrock hosting.",
        )
        tracker.record_reviewed_no_evidence(
            "S2",
            candidate_urls=3,
            reviewed_urls=3,
            notes="Reviewed developer docs; no AI endpoints mentioned.",
        )
        tracker.record_search_attempted(
            "S3",
            candidate_urls=2,
            notes="Initial search executed; awaiting legal review.",
        )
        tracker.mark_unavailable(
            "S5",
            notes="Subprocessor list locked behind customer authentication portal.",
        )
        tracker.mark_requires_manual_review(
            "S6",
            candidate_urls=1,
            notes="Ambiguous ML Engineer posting; needs analyst verification.",
        )

        summary = tracker.get_summary()
        assert summary["total_required_sources"] == 5
        assert summary["completed"] == 2
        assert summary["evidence_recorded"] == 1
        assert summary["reviewed_no_evidence"] == 1
        assert summary["in_progress"] == 1
        assert summary["unavailable"] == 1
        assert summary["requires_manual_review"] == 1
        assert summary["not_started"] == 0
        assert summary["total_evidence_references"] == 2
        assert records["S1"].evidence_references == ["https://vendor.com/trust/soc2", "EV-101"]

    def test_coverage_outcomes_feed_into_stop_condition_evaluator(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        plan = planner.plan_investigation(sample_vendor, "Low")
        tracker = OSINTCoverageTracker(plan.coverage_records)

        tracker.record_reviewed_no_evidence("S1", reviewed_urls=2)
        tracker.record_reviewed_no_evidence("S3", reviewed_urls=1)
        tracker.record_search_attempted("S12", candidate_urls=3)
        tracker.sync_plan_stop_status(plan)

        assert plan.stop_status == AssessmentStopStatus.IN_PROGRESS
        assert "S12" in plan.stop_progress_notes

        tracker.mark_unavailable("S12", notes="Blocked")
        tracker.sync_plan_stop_status(plan)
        assert plan.stop_status == AssessmentStopStatus.IN_PROGRESS
        assert "S12" in plan.stop_progress_notes

        tracker.record_reviewed_no_evidence("S12", reviewed_urls=2)
        tracker.sync_plan_stop_status(plan)
        assert plan.stop_status == AssessmentStopStatus.COMPLETED_CONDITION_MET
        assert "completed" in plan.stop_progress_notes.lower()


# ─── 5. Active Effort Timer, Pause/Resume, Reruns & Persistence Tests ────────

class TestActiveEffortTimerAndPersistence:
    def test_timer_start_pause_resume_and_active_effort_governs_timebox(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        # High-depth assessment has a 20-minute target
        plan = planner.plan_investigation(sample_vendor, "High")
        assert plan.timebox.planned_minutes == 20
        assert plan.timebox.timer_state == TimerState.NOT_STARTED

        t0 = datetime(2026, 10, 10, 9, 0, 0, tzinfo=timezone.utc)

        # 1. Start timer at t=0
        OSINTCoverageTracker.start_timer(plan, now=t0)
        assert plan.timebox.timer_state == TimerState.RUNNING
        assert plan.timebox.actual_analyst_effort_minutes == 0.0
        assert plan.timebox.elapsed_wall_clock_minutes == 0.0

        # 2. Simulate Streamlit rerun at t=5m (must accumulate 5m without double counting)
        t5 = t0 + timedelta(minutes=5)
        OSINTCoverageTracker.refresh_timer(plan, now=t5)
        assert plan.timebox.actual_analyst_effort_minutes == pytest.approx(5.0)
        assert plan.timebox.elapsed_wall_clock_minutes == pytest.approx(5.0)

        # 3. Work until t=12m and pause
        t12 = t0 + timedelta(minutes=12)
        OSINTCoverageTracker.pause_timer(plan, now=t12)
        assert plan.timebox.timer_state == TimerState.PAUSED
        assert plan.timebox.actual_analyst_effort_minutes == pytest.approx(12.0)
        assert plan.timebox.elapsed_wall_clock_minutes == pytest.approx(12.0)
        assert plan.stop_status == AssessmentStopStatus.IN_PROGRESS

        # 4. Pause for 10 minutes (until t=22m) and resume:
        # Wall-clock is now 22m (> 20m target!), but active analyst effort is still 12m (< 20m target).
        # The pause MUST NOT consume the analyst's effort budget!
        t22 = t0 + timedelta(minutes=22)
        OSINTCoverageTracker.resume_timer(plan, now=t22)
        assert plan.timebox.timer_state == TimerState.RUNNING
        assert plan.timebox.actual_analyst_effort_minutes == pytest.approx(12.0)
        assert plan.timebox.elapsed_wall_clock_minutes == pytest.approx(22.0)
        assert plan.timebox.is_exhausted is False
        assert plan.stop_status == AssessmentStopStatus.IN_PROGRESS

        # 5. Simulate coverage & gate updates during active work at t=26m (4m into second session -> 16m effort)
        t26 = t0 + timedelta(minutes=26)
        OSINTCoverageTracker.refresh_timer(plan, now=t26)
        tracker = OSINTCoverageTracker(plan.coverage_records)
        tracker.record_search_attempted("S1", candidate_urls=2)
        tracker.sync_plan_stop_status(plan)
        OSINTCoverageTracker.update_plan_gate(plan, "G1", GateStatus.UNRESOLVED)
        assert plan.timebox.actual_analyst_effort_minutes == pytest.approx(16.0)
        assert plan.timebox.elapsed_wall_clock_minutes == pytest.approx(26.0)
        assert plan.timebox.timer_state == TimerState.RUNNING
        assert plan.stop_status == AssessmentStopStatus.IN_PROGRESS

        # 6. Work another 4 minutes until t=30m (total active effort = 12 + 8 = 20 minutes; wall-clock = 30 minutes)
        t30 = t0 + timedelta(minutes=30)
        OSINTCoverageTracker.stop_timer(plan, now=t30)
        assert plan.timebox.timer_state == TimerState.STOPPED
        assert plan.timebox.actual_analyst_effort_minutes == pytest.approx(20.0)
        assert plan.timebox.elapsed_wall_clock_minutes == pytest.approx(30.0)
        assert plan.timebox.is_exhausted is True
        assert plan.stop_status == AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE

    def test_persistence_across_reloads_and_process_restarts(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor, tmp_path: Path
    ):
        state_file = tmp_path / "osint_state.json"
        plan = planner.plan_investigation(sample_vendor, "Medium")
        t0 = datetime(2026, 10, 10, 10, 0, 0, tzinfo=timezone.utc)

        OSINTCoverageTracker.start_timer(plan, now=t0)
        OSINTCoverageTracker.refresh_timer(plan, now=t0 + timedelta(minutes=4))
        tracker = OSINTCoverageTracker(plan.coverage_records)
        tracker.record_evidence_found("S1", evidence_references=["https://automworx.com/trust"])
        tracker.sync_plan_stop_status(plan)

        # Save to disk
        OSINTInvestigationPlanner.save_plan_state(plan, state_path=state_file)

        # Simulate process restart (pause_if_running=True): effort is preserved (4.0m), not reset to 0!
        reloaded = OSINTInvestigationPlanner.load_plan_state(
            sample_vendor.vendor_id, state_path=state_file, pause_if_running=True
        )
        assert reloaded is not None
        assert reloaded.timebox.actual_analyst_effort_minutes == pytest.approx(4.0)
        assert reloaded.timebox.elapsed_wall_clock_minutes == pytest.approx(4.0)
        assert reloaded.timebox.timer_state == TimerState.PAUSED
        assert reloaded.coverage_records["S1"].status == CoverageState.REVIEWED_EVIDENCE_RECORDED
        assert reloaded.coverage_records["S1"].evidence_references == ["https://automworx.com/trust"]

    def test_legitimately_completed_investigation_not_downgraded_by_subsequent_timer_changes(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        plan = planner.plan_investigation(sample_vendor, "Low")
        t0 = datetime(2026, 10, 10, 11, 0, 0, tzinfo=timezone.utc)
        OSINTCoverageTracker.start_timer(plan, now=t0)
        OSINTCoverageTracker.refresh_timer(plan, now=t0 + timedelta(minutes=3))

        # Complete all required Low sources (S1, S3, S12) within 3 minutes (< 5m target)
        tracker = OSINTCoverageTracker(plan.coverage_records)
        tracker.record_reviewed_no_evidence("S1", reviewed_urls=1)
        tracker.record_reviewed_no_evidence("S3", reviewed_urls=1)
        tracker.record_reviewed_no_evidence("S12", reviewed_urls=1)
        tracker.sync_plan_stop_status(plan)
        assert plan.stop_status == AssessmentStopStatus.COMPLETED_CONDITION_MET

        # Later timer tick or manual update exceeds the 5-minute timebox (e.g. t=8m)
        OSINTCoverageTracker.stop_timer(plan, now=t0 + timedelta(minutes=8))
        assert plan.timebox.actual_analyst_effort_minutes == pytest.approx(8.0)
        # Must remain COMPLETED_CONDITION_MET and never downgrade to TIMEBOX_EXHAUSTED_INCOMPLETE
        assert plan.stop_status == AssessmentStopStatus.COMPLETED_CONDITION_MET


# ─── 6. Stopping Conditions & Timebox Expiry Tests ───────────────────────────

class TestStoppingConditions:
    def test_critical_stops_only_when_all_gates_resolved_and_required_sources_reviewed(self):
        gates = [
            GateChecklistItem(gate_id="G1", title="AI Use", question="Q1", status=GateStatus.QUALIFYING_POSITIVE_EVIDENCE, qualifying_criteria="Crit"),
            GateChecklistItem(gate_id="G2", title="Service Scope", question="Q2", status=GateStatus.QUALIFYING_POSITIVE_EVIDENCE, qualifying_criteria="Crit"),
            GateChecklistItem(gate_id="G3", title="Data Exposure", question="Q3", status=GateStatus.QUALIFYING_NEGATIVE_EVIDENCE, qualifying_criteria="Crit"),
            GateChecklistItem(gate_id="G4", title="Hyperscaler", question="Q4", status=GateStatus.UNRESOLVED, qualifying_criteria="Crit"),
        ]
        timebox = TimeboxBudget(planning_target="30 minutes", planned_minutes=30, min_hours=0.5, max_hours=0.5)
        records = {
            "S1": SourceCoverageRecord(source_id="S1", source_name="S1", is_required=True, status=CoverageState.REVIEWED_EVIDENCE_RECORDED),
        }

        status, reason = StopConditionEvaluator.evaluate("Critical", gates, records, timebox)
        assert status == AssessmentStopStatus.IN_PROGRESS
        assert "G4" in reason

        gates[3].status = GateStatus.QUALIFYING_NEGATIVE_EVIDENCE
        status, reason = StopConditionEvaluator.evaluate("Critical", gates, records, timebox)
        assert status == AssessmentStopStatus.COMPLETED_CONDITION_MET
        assert "resolved" in reason.lower()

    def test_high_stops_after_two_consecutive_non_novel_sources(self):
        gates = []
        records = {
            "S1": SourceCoverageRecord(source_id="S1", source_name="S1", is_required=True, status=CoverageState.NOT_STARTED),
        }
        timebox = TimeboxBudget(planning_target="20 minutes", planned_minutes=20, min_hours=0.33, max_hours=0.33)

        status, _ = StopConditionEvaluator.evaluate("High", gates, records, timebox, recent_source_novelty_history=[True, False])
        assert status == AssessmentStopStatus.IN_PROGRESS

        status, reason = StopConditionEvaluator.evaluate("High", gates, records, timebox, recent_source_novelty_history=[True, False, False])
        assert status == AssessmentStopStatus.COMPLETED_CONDITION_MET
        assert "Two consecutive" in reason

    def test_medium_stops_on_confirmed_signal(self):
        gates = [
            GateChecklistItem(gate_id="G1", title="AI Use", question="Q1", status=GateStatus.QUALIFYING_POSITIVE_EVIDENCE, qualifying_criteria="Crit"),
        ]
        records = {}
        timebox = TimeboxBudget(planning_target="10 minutes", planned_minutes=10, min_hours=0.17, max_hours=0.17)

        status, reason = StopConditionEvaluator.evaluate("Medium", gates, records, timebox)
        assert status == AssessmentStopStatus.COMPLETED_CONDITION_MET
        assert "Confirmed qualifying AI signal" in reason

    def test_poc_timebox_exhaustion_marks_incomplete_not_completed(self):
        gates = [
            GateChecklistItem(gate_id="G1", title="AI Use", question="Q1", status=GateStatus.UNRESOLVED, qualifying_criteria="Crit"),
        ]
        records = {
            "S1": SourceCoverageRecord(source_id="S1", source_name="S1", is_required=True, status=CoverageState.SEARCH_ATTEMPTED),
        }
        timebox = TimeboxBudget(
            planning_target="20 minutes",
            planned_minutes=20,
            elapsed_wall_clock_minutes=25.0,
            actual_analyst_effort_minutes=22.0,
            min_hours=0.3333,
            max_hours=0.3333,
        )

        status, reason = StopConditionEvaluator.evaluate("High", gates, records, timebox)
        assert status == AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE
        assert "20 minutes" in reason or "timebox" in reason.lower()
        assert "incomplete" in reason.lower()
