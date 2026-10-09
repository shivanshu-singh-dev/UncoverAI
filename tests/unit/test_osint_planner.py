"""Unit tests for the Meridian OSINT Depth Engine and Investigation Planner."""

import pytest
from pathlib import Path

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

    def test_o5_human_review_changes_criticality_depth_and_timebox(
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

        # Analyst applies O5 human review override -> High
        o5_override = HumanReviewRecord(
            user_decision="CHANGE_CRITICALITY",
            original_criticality="Low",
            final_criticality="High",
            rationale="O5 escalation due to wealth portfolio exposure.",
        )
        updated_plan = planner.plan_investigation(
            terrapin, criticality_input=crit_res, human_review=o5_override
        )
        assert updated_plan.approval_status == CriticalityApprovalStatus.APPROVED_BY_ANALYST
        assert updated_plan.input_criticality == "High"
        assert updated_plan.selected_depth_profile == "High"
        assert updated_plan.timebox.planning_target == "20 minutes"
        assert updated_plan.timebox.planned_minutes == 20
        assert set(updated_plan.required_source_classes) == {"S1", "S2", "S3", "S5", "S6", "S7", "S8"}

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


# ─── 2. Unified Source Registry & Policy Tests ───────────────────────────────

class TestUnifiedSourceRegistry:
    def test_all_twelve_sources_registered_with_nature_and_discovery_mode(self):
        reg = OSINTSourceRegistry()
        sources = reg.get_all_sources()
        assert len(sources) == 12
        for i in range(1, 13):
            s_id = f"S{i}"
            assert s_id in sources
            s_def = sources[s_id]
            assert s_def.name != ""
            assert s_def.reliability_rating in (1, 2, 3, 4, 5)
            assert len(s_def.evidence_targets) > 0
            assert s_def.source_nature in ("VENDOR_SPECIFIC", "FIXED_PLATFORM", "HYBRID")
            assert s_def.discovery_mode != ""

        # Verify vendor-specific vs fixed-platform vs hybrid designations
        assert sources["S1"].source_nature == "VENDOR_SPECIFIC"
        assert sources["S3"].source_nature == "VENDOR_SPECIFIC"
        assert sources["S4"].source_nature == "HYBRID"
        assert sources["S6"].source_nature == "FIXED_PLATFORM"
        assert sources["S8"].source_nature == "FIXED_PLATFORM"

    def test_resources_mapped_to_sources(self):
        reg = OSINTSourceRegistry()
        resources = reg.get_all_resources()
        assert len(resources) >= 20
        assert "github" in resources
        assert "google_search" in resources
        assert "linkedin_jobs" in resources
        assert "sec_edgar" in resources
        assert "wayback_machine" in resources

        gh = resources["github"]
        assert "S2" in gh.associated_sources
        assert gh.access_mode in ("AUTOMATED", "API_BASED", "MANUALLY_ASSISTED", "UNIMPLEMENTED")

    def test_unified_policy_has_no_competing_policy_conflict_warnings(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        plan = planner.plan_investigation(sample_vendor, "Low")
        assert not hasattr(plan, "policy_conflicts")
        # Conditional S2 is tracked in coverage_records with its trigger condition
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
            # Fixed-platform queries (e.g. site:github.com or site:linkedin.com/jobs) are allowed, but no empty site:
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

        # 1. REVIEWED_EVIDENCE_RECORDED with evidence_references
        tracker.record_evidence_found(
            "S1",
            evidence_references=["https://vendor.com/trust/soc2", "EV-101"],
            candidate_urls=4,
            reviewed_urls=4,
            produced_new_evidence=True,
            notes="SOC 2 Type II report confirms AWS Bedrock hosting.",
        )
        # 2. REVIEWED_NO_EVIDENCE
        tracker.record_reviewed_no_evidence(
            "S2",
            candidate_urls=3,
            reviewed_urls=3,
            notes="Reviewed developer docs; no AI endpoints mentioned.",
        )
        # 3. SEARCH_ATTEMPTED
        tracker.record_search_attempted(
            "S3",
            candidate_urls=2,
            notes="Initial search executed; awaiting legal review.",
        )
        # 4. UNAVAILABLE
        tracker.mark_unavailable(
            "S5",
            notes="Subprocessor list locked behind customer authentication portal.",
        )
        # 5. REQUIRES_MANUAL_REVIEW
        tracker.mark_requires_manual_review(
            "S6",
            candidate_urls=1,
            notes="Ambiguous ML Engineer posting; needs analyst verification.",
        )

        summary = tracker.get_summary()
        assert summary["total_required_sources"] == 5
        assert summary["completed"] == 2  # Only S1 and S2 are terminal reviewed states
        assert summary["evidence_recorded"] == 1
        assert summary["reviewed_no_evidence"] == 1
        assert summary["in_progress"] == 1  # S3 SEARCH_ATTEMPTED
        assert summary["unavailable"] == 1  # S5 UNAVAILABLE
        assert summary["requires_manual_review"] == 1  # S6 REQUIRES_MANUAL_REVIEW
        assert summary["not_started"] == 0
        assert summary["total_evidence_references"] == 2
        assert records["S1"].evidence_references == ["https://vendor.com/trust/soc2", "EV-101"]

    def test_coverage_outcomes_feed_into_stop_condition_evaluator(
        self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor
    ):
        plan = planner.plan_investigation(sample_vendor, "Low")
        tracker = OSINTCoverageTracker(plan.coverage_records)

        # Low requires S1, S3, S12. Mark S1 and S3 as REVIEWED_NO_EVIDENCE, S12 as SEARCH_ATTEMPTED
        tracker.record_reviewed_no_evidence("S1", reviewed_urls=2)
        tracker.record_reviewed_no_evidence("S3", reviewed_urls=1)
        tracker.record_search_attempted("S12", candidate_urls=3)
        tracker.sync_plan_stop_status(plan)

        # Because S12 is only SEARCH_ATTEMPTED (not reviewed), Low stop condition is NOT met
        assert plan.stop_status == AssessmentStopStatus.IN_PROGRESS
        assert "S12" in plan.stop_progress_notes

        # Even if S12 is marked UNAVAILABLE, Low requires all 3 to be reviewed with no signal (or G1 resolved)
        tracker.mark_unavailable("S12", notes="Blocked")
        tracker.sync_plan_stop_status(plan)
        assert plan.stop_status == AssessmentStopStatus.IN_PROGRESS
        assert "S12" in plan.stop_progress_notes

        # Once S12 is REVIEWED_NO_EVIDENCE, Low stop condition is satisfied
        tracker.record_reviewed_no_evidence("S12", reviewed_urls=2)
        tracker.sync_plan_stop_status(plan)
        assert plan.stop_status == AssessmentStopStatus.COMPLETED_CONDITION_MET
        assert "completed" in plan.stop_progress_notes.lower()


# ─── 5. Stopping Conditions & Timebox Expiry Tests ───────────────────────────

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

        # G4 is unresolved -> IN_PROGRESS
        status, reason = StopConditionEvaluator.evaluate("Critical", gates, records, timebox)
        assert status == AssessmentStopStatus.IN_PROGRESS
        assert "G4" in reason

        # Resolve G4 -> COMPLETED_CONDITION_MET
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
        assert "20m" in reason or "20 min" in reason or "timebox" in reason.lower()
        assert "incomplete" in reason.lower()
