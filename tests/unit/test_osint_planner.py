"""Unit tests for the Meridian OSINT Depth Engine and Investigation Planner."""

import pytest
from pathlib import Path

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


# ─── 1. Depth Selection Tests ─────────────────────────────────────────────────

class TestDepthSelection:
    def test_critical_selects_critical_profile(self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor):
        plan = planner.plan_investigation(sample_vendor, "Critical")
        assert plan.selected_depth_profile == "Critical"
        assert len(plan.required_source_classes) == 12  # S1-S12
        assert "S1" in plan.required_source_classes
        assert "S12" in plan.required_source_classes
        assert plan.reviewer_requirements.get("second_reviewer") is True

    def test_high_selects_high_profile(self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor):
        plan = planner.plan_investigation(sample_vendor, "High")
        assert plan.selected_depth_profile == "High"
        assert set(plan.required_source_classes) == {"S1", "S2", "S3", "S5", "S6", "S7", "S8"}
        assert plan.reviewer_requirements.get("analyst_validation") is True
        assert plan.reviewer_requirements.get("second_reviewer") is False

    def test_medium_selects_medium_profile(self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor):
        plan = planner.plan_investigation(sample_vendor, "Medium")
        assert plan.selected_depth_profile == "Medium"
        assert set(plan.required_source_classes) == {"S1", "S2", "S3", "S6"}

    def test_low_selects_low_profile(self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor):
        plan = planner.plan_investigation(sample_vendor, "Low")
        assert plan.selected_depth_profile == "Low"
        assert set(plan.required_source_classes) == {"S1", "S3", "S12"}

    def test_approval_status_governance_distinction(self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor):
        # Case A: Human review provided -> APPROVED_BY_ANALYST
        h_rev = HumanReviewRecord(
            user_decision="CHANGE_CRITICALITY",
            original_criticality="High",
            final_criticality="Critical",
            rationale="Analyst approved Critical classification.",
        )
        plan_approved = planner.plan_investigation(sample_vendor, human_review=h_rev)
        assert plan_approved.approval_status == CriticalityApprovalStatus.APPROVED_BY_ANALYST
        assert plan_approved.selected_depth_profile == "Critical"

        # Case B: Plain string input without review -> PROVISIONAL_CALCULATED
        plan_provisional = planner.plan_investigation(sample_vendor, "High")
        assert plan_provisional.approval_status == CriticalityApprovalStatus.PROVISIONAL_CALCULATED


# ─── 2. Source Coverage & Registry Tests ──────────────────────────────────────

class TestSourceRegistryAndCoverage:
    def test_all_twelve_sources_registered(self):
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

    def test_resources_mapped_to_sources(self):
        reg = OSINTSourceRegistry()
        resources = reg.get_all_resources()
        assert len(resources) >= 20
        # Check specific key discovery tools
        assert "github" in resources
        assert "google_search" in resources
        assert "linkedin_jobs" in resources
        assert "sec_edgar" in resources
        assert "wayback_machine" in resources

        gh = resources["github"]
        assert "S2" in gh.associated_sources
        assert gh.access_mode in ("AUTOMATED", "API_BASED", "MANUALLY_ASSISTED", "UNIMPLEMENTED")

    def test_fixed_source_policy_conflicts_surfaced(self, planner: OSINTInvestigationPlanner, sample_vendor: MeridianVendor):
        # Low profile only requires S1, S3, S12. Team policy mandates GitHub (S2/S4/S9) and Job boards (S6).
        plan = planner.plan_investigation(sample_vendor, "Low")
        assert len(plan.policy_conflicts) > 0
        conflict_sources = [c.source_id for c in plan.policy_conflicts]
        assert "S6" in conflict_sources  # Job boards omitted from Low profile


# ─── 3. Query Generation Tests ────────────────────────────────────────────────

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
        # Check clean substitution
        rendered_texts = [q.rendered_query for q in queries]
        # Should substitute cleaned vendor name without corporate suffix
        assert any('"AutomWorx" AI' in r for r in rendered_texts)
        assert any('site:automworx.com' in r for r in rendered_texts)
        assert not any('https://' in r for r in rendered_texts)

    def test_queries_deduplicated_preserving_source_associations(self):
        qp = OSINTQueryPlanner()
        queries = qp.generate_queries_for_vendor(
            vendor_id="V-002",
            vendor_name="Fiserv",
            domain="fiserv.com",
            active_source_classes=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"],
        )
        # Check that rendered query strings are strictly unique
        rendered_lower = [q.rendered_query.lower() for q in queries]
        assert len(rendered_lower) == len(set(rendered_lower))

        # Check execution status is PLANNED (never executed)
        for q in queries:
            assert q.execution_status == SearchExecutionStatus.PLANNED

    def test_quotes_and_special_characters_handled_safely(self):
        qp = OSINTQueryPlanner()
        clean = qp.clean_vendor_name('Acme "AI" Solutions, LLC.')
        assert '"' not in clean
        assert clean.strip() == "Acme AI Solutions"


# ─── 4. Coverage Tracking Tests ───────────────────────────────────────────────

class TestCoverageTracker:
    def test_state_distinctions(self):
        records = {
            "S1": SourceCoverageRecord(source_id="S1", source_name="Trust Centers", is_required=True),
            "S2": SourceCoverageRecord(source_id="S2", source_name="Product Docs", is_required=True),
            "S3": SourceCoverageRecord(source_id="S3", source_name="Privacy Terms", is_required=True),
        }
        tracker = OSINTCoverageTracker(records)

        # Update states
        tracker.update_source_status("S1", CoverageState.REVIEWED_EVIDENCE_RECORDED, candidate_urls=5, reviewed_urls=5, produced_new_evidence=True)
        tracker.update_source_status("S2", CoverageState.UNAVAILABLE, notes="Vendor developer portal behind enterprise firewall.")
        # S3 remains NOT_STARTED

        summary = tracker.get_summary()
        assert summary["total_required_sources"] == 3
        assert summary["completed"] == 1
        assert summary["unavailable"] == 1
        assert summary["not_started"] == 1
        assert summary["total_urls_found"] == 5

        # Unavailable does NOT equal completed with no evidence
        assert records["S2"].status == CoverageState.UNAVAILABLE
        assert records["S1"].status == CoverageState.REVIEWED_EVIDENCE_RECORDED


# ─── 5. Stopping Conditions Tests ─────────────────────────────────────────────

class TestStoppingConditions:
    def test_critical_stops_only_when_all_gates_resolved(self):
        gates = [
            GateChecklistItem(gate_id="G1", title="AI Use", question="Q1", status=GateStatus.QUALIFYING_POSITIVE_EVIDENCE, qualifying_criteria="Crit"),
            GateChecklistItem(gate_id="G2", title="Service Scope", question="Q2", status=GateStatus.QUALIFYING_POSITIVE_EVIDENCE, qualifying_criteria="Crit"),
            GateChecklistItem(gate_id="G3", title="Data Exposure", question="Q3", status=GateStatus.QUALIFYING_NEGATIVE_EVIDENCE, qualifying_criteria="Crit"),
            GateChecklistItem(gate_id="G4", title="Hyperscaler", question="Q4", status=GateStatus.UNRESOLVED, qualifying_criteria="Crit"),
        ]
        timebox = TimeboxBudget(planning_target="6–8 hours", min_hours=6.0, max_hours=8.0)
        records = {}

        # G4 is unresolved -> IN_PROGRESS
        status, reason = StopConditionEvaluator.evaluate("Critical", gates, records, timebox)
        assert status == AssessmentStopStatus.IN_PROGRESS
        assert "G4" in reason

        # Resolve G4
        gates[3].status = GateStatus.QUALIFYING_NEGATIVE_EVIDENCE
        status, reason = StopConditionEvaluator.evaluate("Critical", gates, records, timebox)
        assert status == AssessmentStopStatus.COMPLETED_CONDITION_MET
        assert "resolved" in reason.lower()

    def test_high_stops_after_two_consecutive_non_novel_sources(self):
        gates = []
        records = {
            "S1": SourceCoverageRecord(source_id="S1", source_name="S1", is_required=True, status=CoverageState.NOT_STARTED),
        }
        timebox = TimeboxBudget(planning_target="3–4 hours", min_hours=3.0, max_hours=4.0)

        # Novelty history [True, False] -> IN_PROGRESS
        status, _ = StopConditionEvaluator.evaluate("High", gates, records, timebox, recent_source_novelty_history=[True, False])
        assert status == AssessmentStopStatus.IN_PROGRESS

        # Novelty history [True, False, False] -> COMPLETED_CONDITION_MET
        status, reason = StopConditionEvaluator.evaluate("High", gates, records, timebox, recent_source_novelty_history=[True, False, False])
        assert status == AssessmentStopStatus.COMPLETED_CONDITION_MET
        assert "Two consecutive" in reason

    def test_medium_stops_on_confirmed_signal(self):
        gates = [
            GateChecklistItem(gate_id="G1", title="AI Use", question="Q1", status=GateStatus.QUALIFYING_POSITIVE_EVIDENCE, qualifying_criteria="Crit"),
        ]
        records = {}
        timebox = TimeboxBudget(planning_target="1–2 hours", min_hours=1.0, max_hours=2.0)

        status, reason = StopConditionEvaluator.evaluate("Medium", gates, records, timebox)
        assert status == AssessmentStopStatus.COMPLETED_CONDITION_MET
        assert "Confirmed qualifying AI signal" in reason

    def test_timebox_exhaustion_marks_incomplete_not_completed(self):
        gates = [
            GateChecklistItem(gate_id="G1", title="AI Use", question="Q1", status=GateStatus.UNRESOLVED, qualifying_criteria="Crit"),
        ]
        records = {}
        timebox = TimeboxBudget(
            planning_target="6–8 hours",
            min_hours=6.0,
            max_hours=8.0,
            actual_analyst_time_hours=8.5,  # Exceeded 8 hours
        )

        status, reason = StopConditionEvaluator.evaluate("Critical", gates, records, timebox)
        assert status == AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE
        assert "Timebox limit" in reason
