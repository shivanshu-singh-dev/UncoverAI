"""Deterministic stopping condition evaluators for OSINT depth profiles."""

from typing import Optional

from meridian_assessment.models.osint_plan import (
    AssessmentStopStatus,
    CoverageState,
    GateChecklistItem,
    GateStatus,
    SourceCoverageRecord,
    TimeboxBudget,
)

# States that represent a completed review of a source class
_COMPLETED_REVIEW_STATES = (
    CoverageState.REVIEWED_EVIDENCE_RECORDED,
    CoverageState.REVIEWED_NO_EVIDENCE,
)


class StopConditionEvaluator:
    """Evaluates rulebook stopping conditions and POC timebox tracking."""

    @staticmethod
    def is_timebox_exhausted(timebox: TimeboxBudget) -> bool:
        """Determine whether the planned timebox has been reached or exceeded."""
        if timebox.is_exhausted:
            return True

        # Check minute-based POC budget (planned_minutes vs elapsed or analyst effort)
        legacy_extra_minutes = (
            (timebox.actual_analyst_time_hours * 60.0)
            + timebox.actual_automated_execution_time_minutes
            + timebox.time_spent_searching_minutes
            + timebox.time_spent_reviewing_minutes
            + timebox.time_spent_resolving_ambiguous_minutes
            + timebox.time_spent_validating_minutes
        )
        effective_effort_mins = max(timebox.actual_analyst_effort_minutes, legacy_extra_minutes)
        effective_wall_mins = timebox.elapsed_wall_clock_minutes

        if timebox.planned_minutes > 0:
            if effective_effort_mins >= timebox.planned_minutes or effective_wall_mins >= timebox.planned_minutes:
                return True

        if timebox.max_hours > 0 and (effective_effort_mins / 60.0) >= timebox.max_hours:
            return True

        return False

    @staticmethod
    def evaluate(
        profile_name: str,
        gate_checklist: list[GateChecklistItem],
        coverage_records: dict[str, SourceCoverageRecord],
        timebox: TimeboxBudget,
        recent_source_novelty_history: Optional[list[bool]] = None,
    ) -> tuple[AssessmentStopStatus, str]:
        """Evaluate whether stopping condition is met or timebox is exhausted.

        Returns:
            (AssessmentStopStatus, reason_message)
        """
        timebox_is_exhausted = StopConditionEvaluator.is_timebox_exhausted(timebox)
        prof = profile_name.capitalize()

        # 1. Critical Profile: Stop when G1-G4 resolved with qualifying positive/negative evidence
        if prof == "Critical":
            unresolved_gates = [
                g.gate_id
                for g in gate_checklist
                if g.status not in (
                    GateStatus.QUALIFYING_POSITIVE_EVIDENCE,
                    GateStatus.QUALIFYING_NEGATIVE_EVIDENCE,
                )
            ]
            if not unresolved_gates:
                return (
                    AssessmentStopStatus.COMPLETED_CONDITION_MET,
                    "All G1–G4 investigative gates resolved with qualifying evidence or formal negative findings logged.",
                )
            if timebox_is_exhausted:
                return (
                    AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE,
                    f"Timebox limit ({timebox.planning_target}) reached while gates remain unresolved: {', '.join(unresolved_gates)}.",
                )
            return (
                AssessmentStopStatus.IN_PROGRESS,
                f"Investigation in progress. Outstanding gates: {', '.join(unresolved_gates)}.",
            )

        # 2. High Profile: Stop after two consecutive completed source classes add no new information
        elif prof == "High":
            if recent_source_novelty_history is None:
                # Derive novelty sequence from completed source coverage records in order
                completed_recs = [
                    s for s in coverage_records.values()
                    if s.status in _COMPLETED_REVIEW_STATES
                ]
                recent_source_novelty_history = [s.produced_new_evidence for s in completed_recs]

            if len(recent_source_novelty_history) >= 2:
                last_two = recent_source_novelty_history[-2:]
                if last_two == [False, False]:
                    return (
                        AssessmentStopStatus.COMPLETED_CONDITION_MET,
                        "Stopping condition met: Two consecutive completed source classes yielded no novel information.",
                    )

            # Check if all required sources have been reviewed
            uncompleted = [
                s.source_id
                for s in coverage_records.values()
                if s.is_required and s.status not in _COMPLETED_REVIEW_STATES
            ]
            if not uncompleted and any(s.is_required for s in coverage_records.values()):
                return (
                    AssessmentStopStatus.COMPLETED_CONDITION_MET,
                    "All required source classes in High depth profile have been reviewed and completed.",
                )

            if timebox_is_exhausted:
                return (
                    AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE,
                    f"Timebox limit ({timebox.planning_target}) reached with {len(uncompleted)} required sources incomplete: {', '.join(uncompleted)}.",
                )

            return (
                AssessmentStopStatus.IN_PROGRESS,
                f"High investigation in progress ({len(uncompleted)} required sources remaining: {', '.join(uncompleted)}).",
            )

        # 3. Medium Profile: Stop after one CONFIRMED signal or required source exhaustion
        elif prof == "Medium":
            has_confirmed_signal = any(
                g.status == GateStatus.QUALIFYING_POSITIVE_EVIDENCE
                for g in gate_checklist
            )
            if has_confirmed_signal:
                return (
                    AssessmentStopStatus.COMPLETED_CONDITION_MET,
                    "Stopping condition met: Confirmed qualifying AI signal obtained. Outstanding gates logged as unresolved.",
                )

            uncompleted_med = [
                s.source_id
                for s in coverage_records.values()
                if s.is_required and s.status not in _COMPLETED_REVIEW_STATES
            ]
            if not uncompleted_med and any(s.is_required for s in coverage_records.values()):
                return (
                    AssessmentStopStatus.COMPLETED_CONDITION_MET,
                    "Required Medium source classes reviewed and exhausted without confirming positive signal.",
                )

            if timebox_is_exhausted:
                return (
                    AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE,
                    f"Timebox limit ({timebox.planning_target}) reached with {len(uncompleted_med)} required sources incomplete: {', '.join(uncompleted_med)}.",
                )

            return (
                AssessmentStopStatus.IN_PROGRESS,
                f"Medium investigation in progress ({len(uncompleted_med)} required sources remaining: {', '.join(uncompleted_med)}).",
            )

        # 4. Low Profile: Complete lightweight pass
        elif prof == "Low":
            uncompleted_low = [
                s.source_id
                for s in coverage_records.values()
                if s.is_required and s.status not in _COMPLETED_REVIEW_STATES
            ]
            if not uncompleted_low and any(s.is_required for s in coverage_records.values()):
                return (
                    AssessmentStopStatus.COMPLETED_CONDITION_MET,
                    "Configured lightweight source pass (S1, S3, S12) completed. Reassessment schedule logged.",
                )

            if timebox_is_exhausted:
                return (
                    AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE,
                    f"Timebox limit ({timebox.planning_target}) reached before finishing lightweight pass ({', '.join(uncompleted_low)} remaining).",
                )

            return (
                AssessmentStopStatus.IN_PROGRESS,
                f"Low pass in progress ({len(uncompleted_low)} required sources remaining: {', '.join(uncompleted_low)}).",
            )

        return AssessmentStopStatus.IN_PROGRESS, "Investigation status undetermined."
