"""Source-class coverage tracking, analyst outcome recording, active timer lifecycle, and stop-condition synchronization."""

from datetime import datetime, timezone
from typing import Optional

from meridian_assessment.models.osint_plan import (
    AssessmentStopStatus,
    CoverageState,
    GateStatus,
    OSINTInvestigationPlan,
    SourceCoverageRecord,
    TimerState,
)
from meridian_assessment.services.osint.stop_conditions import StopConditionEvaluator


def _ensure_utc(dt: Optional[datetime] = None) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_utc(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime."""
    parsed = datetime.fromisoformat(ts)
    return _ensure_utc(parsed)


class OSINTCoverageTracker:
    """Tracks investigation progress across planned source classes, manages the active effort timer, and feeds outcomes into the stopping evaluator."""

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

    # ─── Active Analyst Effort & Wall-Clock Timer Lifecycle ──────────────────

    @staticmethod
    def _recompute_stop_status_for_timer(plan: OSINTInvestigationPlan) -> OSINTInvestigationPlan:
        """Re-evaluate stopping condition after a timer update without downgrading an already-completed investigation."""
        was_completed = plan.stop_status == AssessmentStopStatus.COMPLETED_CONDITION_MET
        was_notes = plan.stop_progress_notes

        plan.timebox.is_exhausted = StopConditionEvaluator.is_timebox_exhausted(plan.timebox)
        status, notes = StopConditionEvaluator.evaluate(
            profile_name=plan.selected_depth_profile,
            gate_checklist=plan.gate_checklist,
            coverage_records=plan.coverage_records,
            timebox=plan.timebox,
        )

        # Requirement 2D: If the investigation is already legitimately complete, do not downgrade it
        # to incomplete merely because a timer value subsequently changes.
        if was_completed and status == AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE:
            plan.stop_status = AssessmentStopStatus.COMPLETED_CONDITION_MET
            plan.stop_progress_notes = was_notes
        else:
            plan.stop_status = status
            plan.stop_progress_notes = notes
        return plan

    @staticmethod
    def start_timer(
        plan: OSINTInvestigationPlan,
        now: Optional[datetime] = None,
    ) -> OSINTInvestigationPlan:
        """Start or resume the active analyst effort timer."""
        cur_dt = _ensure_utc(now)
        tb = plan.timebox

        if tb.timer_state == TimerState.RUNNING:
            # Already running: refresh checkpoint cleanly without double-counting
            return OSINTCoverageTracker.refresh_timer(plan, now=cur_dt)

        if tb.first_started_at is None:
            tb.first_started_at = cur_dt.isoformat()

        tb.active_segment_started_at = cur_dt.isoformat()
        tb.last_updated_at = cur_dt.isoformat()
        tb.timer_state = TimerState.RUNNING

        # Update wall-clock elapsed if restarting after a pause/stop
        first_dt = _parse_iso_utc(tb.first_started_at)
        wall_sec = max(0.0, (cur_dt - first_dt).total_seconds())
        tb.elapsed_wall_clock_minutes = max(tb.elapsed_wall_clock_minutes, round(wall_sec / 60.0, 4))

        return OSINTCoverageTracker._recompute_stop_status_for_timer(plan)

    @staticmethod
    def refresh_timer(
        plan: OSINTInvestigationPlan,
        now: Optional[datetime] = None,
    ) -> OSINTInvestigationPlan:
        """Advance active analyst effort and wall-clock time safely across Streamlit reruns.

        Advances `active_segment_started_at` to `now` so repeated reruns only accumulate incremental delta.
        """
        cur_dt = _ensure_utc(now)
        tb = plan.timebox

        if tb.timer_state == TimerState.RUNNING and tb.active_segment_started_at:
            seg_start = _parse_iso_utc(tb.active_segment_started_at)
            delta_sec = max(0.0, (cur_dt - seg_start).total_seconds())
            tb.accumulated_active_seconds += delta_sec
            tb.active_segment_started_at = cur_dt.isoformat()
            tb.last_updated_at = cur_dt.isoformat()
            tb.actual_analyst_effort_minutes = round(tb.accumulated_active_seconds / 60.0, 4)
            tb.actual_analyst_time_hours = round(tb.actual_analyst_effort_minutes / 60.0, 4)

        if tb.first_started_at and tb.timer_state in (TimerState.RUNNING, TimerState.PAUSED):
            first_dt = _parse_iso_utc(tb.first_started_at)
            wall_sec = max(0.0, (cur_dt - first_dt).total_seconds())
            tb.elapsed_wall_clock_minutes = max(tb.elapsed_wall_clock_minutes, round(wall_sec / 60.0, 4))

        return OSINTCoverageTracker._recompute_stop_status_for_timer(plan)

    @staticmethod
    def pause_timer(
        plan: OSINTInvestigationPlan,
        now: Optional[datetime] = None,
    ) -> OSINTInvestigationPlan:
        """Pause the active analyst effort timer.

        Flushes active effort accumulated up to `now` and clears `active_segment_started_at` so pause duration
        does not consume the analyst effort budget.
        """
        cur_dt = _ensure_utc(now)
        tb = plan.timebox

        if tb.timer_state == TimerState.RUNNING:
            OSINTCoverageTracker.refresh_timer(plan, now=cur_dt)

        if tb.first_started_at:
            first_dt = _parse_iso_utc(tb.first_started_at)
            wall_sec = max(0.0, (cur_dt - first_dt).total_seconds())
            tb.elapsed_wall_clock_minutes = max(tb.elapsed_wall_clock_minutes, round(wall_sec / 60.0, 4))

        tb.active_segment_started_at = None
        tb.last_updated_at = cur_dt.isoformat()
        tb.timer_state = TimerState.PAUSED
        return OSINTCoverageTracker._recompute_stop_status_for_timer(plan)

    @staticmethod
    def resume_timer(
        plan: OSINTInvestigationPlan,
        now: Optional[datetime] = None,
    ) -> OSINTInvestigationPlan:
        """Resume active analyst effort tracking after a pause.

        Pause duration contributes to `elapsed_wall_clock_minutes` but NOT `actual_analyst_effort_minutes`.
        """
        return OSINTCoverageTracker.start_timer(plan, now=now)

    @staticmethod
    def stop_timer(
        plan: OSINTInvestigationPlan,
        now: Optional[datetime] = None,
    ) -> OSINTInvestigationPlan:
        """Stop/finish the investigation timer and evaluate final stopping status."""
        cur_dt = _ensure_utc(now)
        tb = plan.timebox

        if tb.timer_state == TimerState.RUNNING:
            OSINTCoverageTracker.refresh_timer(plan, now=cur_dt)
        elif tb.first_started_at:
            first_dt = _parse_iso_utc(tb.first_started_at)
            wall_sec = max(0.0, (cur_dt - first_dt).total_seconds())
            tb.elapsed_wall_clock_minutes = max(tb.elapsed_wall_clock_minutes, round(wall_sec / 60.0, 4))

        tb.active_segment_started_at = None
        tb.last_updated_at = cur_dt.isoformat()
        tb.timer_state = TimerState.STOPPED
        return OSINTCoverageTracker._recompute_stop_status_for_timer(plan)

    @staticmethod
    def update_plan_timebox(
        plan: OSINTInvestigationPlan,
        elapsed_wall_clock_minutes: Optional[float] = None,
        actual_analyst_effort_minutes: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> OSINTInvestigationPlan:
        """Persist manual adjustments to elapsed wall-clock and actual analyst effort time and re-evaluate stop status."""
        cur_dt = _ensure_utc(now)
        tb = plan.timebox

        if elapsed_wall_clock_minutes is not None:
            tb.elapsed_wall_clock_minutes = max(0.0, float(elapsed_wall_clock_minutes))
        if actual_analyst_effort_minutes is not None:
            tb.actual_analyst_effort_minutes = max(0.0, float(actual_analyst_effort_minutes))
            tb.accumulated_active_seconds = round(tb.actual_analyst_effort_minutes * 60.0, 4)
            tb.actual_analyst_time_hours = round(tb.actual_analyst_effort_minutes / 60.0, 4)
            if tb.timer_state == TimerState.RUNNING:
                tb.active_segment_started_at = cur_dt.isoformat()
        tb.last_updated_at = cur_dt.isoformat()

        return OSINTCoverageTracker._recompute_stop_status_for_timer(plan)

    @staticmethod
    def update_plan_gate(
        plan: OSINTInvestigationPlan,
        gate_id: str,
        new_status: GateStatus,
        evidence_notes: str = "",
    ) -> OSINTInvestigationPlan:
        """Persist a G1–G4 gate status update and re-evaluate the plan's stopping condition without resetting the timer."""
        g_id = gate_id.upper()
        for gate in plan.gate_checklist:
            if gate.gate_id.upper() == g_id:
                gate.status = new_status
                if evidence_notes:
                    gate.evidence_notes = evidence_notes
                break

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

    def sync_plan_stop_status(self, plan: OSINTInvestigationPlan) -> OSINTInvestigationPlan:
        """Persist tracker records into the plan and recompute the stopping condition status without resetting the timer."""
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
