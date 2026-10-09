"""Source-class coverage tracking, analyst outcome recording, and stop-condition synchronization."""

from typing import Optional

from meridian_assessment.models.osint_plan import (
    CoverageState,
    GateStatus,
    OSINTInvestigationPlan,
    SourceCoverageRecord,
)
from meridian_assessment.services.osint.stop_conditions import StopConditionEvaluator


class OSINTCoverageTracker:
    """Tracks investigation progress across planned source classes and feeds outcomes into the stopping evaluator."""

    def __init__(self, records: dict[str, SourceCoverageRecord]) -> None:
        self.records = records
        self._novelty_sequence: list[bool] = []

    def update_source_status(
        self,
        source_id: str,
        new_state: CoverageState,
        candidate_urls: int = 0,
        reviewed_urls: int = 0,
        produced_new_evidence: bool = False,
        evidence_references: Optional[list[str]] = None,
        notes: str = "",
    ) -> None:
        """Update coverage audit state for a specific source class."""
        s_id = source_id.upper()
        if s_id not in self.records:
            raise KeyError(f"Source class '{s_id}' is not in the current investigation plan.")

        rec = self.records[s_id]
        rec.status = new_state
        rec.candidate_urls_count = candidate_urls
        rec.reviewed_urls_count = reviewed_urls

        # Enforce evidence integrity: only REVIEWED_EVIDENCE_RECORDED can mark produced_new_evidence=True
        if new_state == CoverageState.REVIEWED_EVIDENCE_RECORDED:
            rec.produced_new_evidence = produced_new_evidence
        else:
            rec.produced_new_evidence = False

        if evidence_references is not None:
            rec.evidence_references = [ref.strip() for ref in evidence_references if ref.strip()]

        if notes:
            rec.notes = notes

        if new_state in (CoverageState.REVIEWED_EVIDENCE_RECORDED, CoverageState.REVIEWED_NO_EVIDENCE):
            self._novelty_sequence.append(rec.produced_new_evidence)

    def record_search_attempted(
        self,
        source_id: str,
        candidate_urls: int = 0,
        notes: str = "",
    ) -> None:
        """Record that a search was attempted for a source class (not yet reviewed)."""
        self.update_source_status(
            source_id=source_id,
            new_state=CoverageState.SEARCH_ATTEMPTED,
            candidate_urls=candidate_urls,
            reviewed_urls=0,
            produced_new_evidence=False,
            notes=notes or "Search attempted; awaiting source review.",
        )

    def record_evidence_found(
        self,
        source_id: str,
        evidence_references: list[str],
        candidate_urls: int = 1,
        reviewed_urls: int = 1,
        produced_new_evidence: bool = True,
        notes: str = "",
    ) -> None:
        """Record that a source was reviewed and relevant evidence was found and referenced."""
        cleaned_refs = [r.strip() for r in evidence_references if r.strip()]
        if not cleaned_refs and not notes.strip():
            raise ValueError("Recording found evidence requires at least one evidence reference (URL/doc ID) or note.")
        self.update_source_status(
            source_id=source_id,
            new_state=CoverageState.REVIEWED_EVIDENCE_RECORDED,
            candidate_urls=max(candidate_urls, reviewed_urls, len(cleaned_refs)),
            reviewed_urls=max(1, reviewed_urls),
            produced_new_evidence=produced_new_evidence,
            evidence_references=cleaned_refs,
            notes=notes or f"Evidence recorded ({len(cleaned_refs)} reference(s)).",
        )

    def record_reviewed_no_evidence(
        self,
        source_id: str,
        candidate_urls: int = 1,
        reviewed_urls: int = 1,
        notes: str = "",
    ) -> None:
        """Record a negative finding: source was reviewed without relevant AI evidence."""
        self.update_source_status(
            source_id=source_id,
            new_state=CoverageState.REVIEWED_NO_EVIDENCE,
            candidate_urls=max(candidate_urls, reviewed_urls),
            reviewed_urls=max(1, reviewed_urls),
            produced_new_evidence=False,
            evidence_references=[],
            notes=notes or "Source reviewed; no relevant AI evidence identified (negative finding).",
        )

    def mark_unavailable(
        self,
        source_id: str,
        notes: str = "",
    ) -> None:
        """Mark a source as unavailable or inaccessible."""
        self.update_source_status(
            source_id=source_id,
            new_state=CoverageState.UNAVAILABLE,
            candidate_urls=0,
            reviewed_urls=0,
            produced_new_evidence=False,
            evidence_references=[],
            notes=notes or "Source unavailable or inaccessible during review.",
        )

    def mark_requires_manual_review(
        self,
        source_id: str,
        candidate_urls: int = 0,
        notes: str = "",
    ) -> None:
        """Mark a source as requiring manual analyst follow-up (e.g. gated portal or ambiguous claim)."""
        self.update_source_status(
            source_id=source_id,
            new_state=CoverageState.REQUIRES_MANUAL_REVIEW,
            candidate_urls=candidate_urls,
            reviewed_urls=0,
            produced_new_evidence=False,
            notes=notes or "Source flagged for manual analyst review.",
        )

    @staticmethod
    def update_plan_timebox(
        plan: OSINTInvestigationPlan,
        elapsed_wall_clock_minutes: Optional[float] = None,
        actual_analyst_effort_minutes: Optional[float] = None,
    ) -> OSINTInvestigationPlan:
        """Persist elapsed wall-clock and actual analyst effort time into the plan and re-evaluate stop status."""
        if elapsed_wall_clock_minutes is not None:
            plan.timebox.elapsed_wall_clock_minutes = max(0.0, float(elapsed_wall_clock_minutes))
        if actual_analyst_effort_minutes is not None:
            plan.timebox.actual_analyst_effort_minutes = max(0.0, float(actual_analyst_effort_minutes))
            plan.timebox.actual_analyst_time_hours = round(plan.timebox.actual_analyst_effort_minutes / 60.0, 4)

        plan.timebox.is_exhausted = StopConditionEvaluator.is_timebox_exhausted(plan.timebox)
        status, notes = StopConditionEvaluator.evaluate(
            profile_name=plan.selected_depth_profile,
            gate_checklist=plan.gate_checklist,
            coverage_records=plan.coverage_records,
            timebox=plan.timebox,
        )
        plan.stop_status = status
        plan.stop_progress_notes = notes
        return plan

    @staticmethod
    def update_plan_gate(
        plan: OSINTInvestigationPlan,
        gate_id: str,
        new_status: GateStatus,
        evidence_notes: str = "",
    ) -> OSINTInvestigationPlan:
        """Persist a G1–G4 gate status update and re-evaluate the plan's stopping condition."""
        g_id = gate_id.upper()
        for gate in plan.gate_checklist:
            if gate.gate_id.upper() == g_id:
                gate.status = new_status
                if evidence_notes:
                    gate.evidence_notes = evidence_notes
                break

        status, notes = StopConditionEvaluator.evaluate(
            profile_name=plan.selected_depth_profile,
            gate_checklist=plan.gate_checklist,
            coverage_records=plan.coverage_records,
            timebox=plan.timebox,
        )
        plan.stop_status = status
        plan.stop_progress_notes = notes
        return plan

    def sync_plan_stop_status(self, plan: OSINTInvestigationPlan) -> OSINTInvestigationPlan:
        """Persist tracker records into the plan and recompute the stopping condition status."""
        plan.coverage_records = self.records
        plan.timebox.is_exhausted = StopConditionEvaluator.is_timebox_exhausted(plan.timebox)
        history = self._novelty_sequence if self._novelty_sequence else None
        status, reason = StopConditionEvaluator.evaluate(
            profile_name=plan.selected_depth_profile,
            gate_checklist=plan.gate_checklist,
            coverage_records=plan.coverage_records,
            timebox=plan.timebox,
            recent_source_novelty_history=history,
        )
        plan.stop_status = status
        plan.stop_progress_notes = reason
        return plan

    def get_summary(self) -> dict[str, int]:
        """Compute coverage progress metrics while keeping negative, unavailable, and unsearched states distinct."""
        total_required = sum(1 for r in self.records.values() if r.is_required)
        not_started = sum(1 for r in self.records.values() if r.status == CoverageState.NOT_STARTED)
        in_progress = sum(
            1 for r in self.records.values()
            if r.status in (
                CoverageState.SEARCH_ATTEMPTED,
                CoverageState.RESULTS_FOUND,
                CoverageState.RELEVANT_SOURCE_IDENTIFIED,
            )
        )
        evidence_recorded = sum(
            1 for r in self.records.values() if r.status == CoverageState.REVIEWED_EVIDENCE_RECORDED
        )
        reviewed_no_evidence = sum(
            1 for r in self.records.values() if r.status == CoverageState.REVIEWED_NO_EVIDENCE
        )
        completed = evidence_recorded + reviewed_no_evidence
        unavailable = sum(1 for r in self.records.values() if r.status == CoverageState.UNAVAILABLE)
        manual_review = sum(1 for r in self.records.values() if r.status == CoverageState.REQUIRES_MANUAL_REVIEW)

        total_urls_found = sum(r.candidate_urls_count for r in self.records.values())
        total_urls_reviewed = sum(r.reviewed_urls_count for r in self.records.values())
        total_evidence_refs = sum(len(r.evidence_references) for r in self.records.values())

        return {
            "total_planned_sources": len(self.records),
            "total_required_sources": total_required,
            "not_started": not_started,
            "in_progress": in_progress,
            "completed": completed,
            "evidence_recorded": evidence_recorded,
            "reviewed_no_evidence": reviewed_no_evidence,
            "unavailable": unavailable,
            "requires_manual_review": manual_review,
            "total_urls_found": total_urls_found,
            "total_urls_reviewed": total_urls_reviewed,
            "total_evidence_references": total_evidence_refs,
        }
