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


class StopConditionEvaluator:
    """Evaluates rulebook stopping conditions and timebox tracking."""

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
        # Calculate elapsed time
        elapsed_hours = (
            timebox.actual_analyst_time_hours
            + (timebox.actual_automated_execution_time_minutes / 60.0)
            + (timebox.time_spent_searching_minutes / 60.0)
            + (timebox.time_spent_reviewing_minutes / 60.0)
            + (timebox.time_spent_resolving_ambiguous_minutes / 60.0)
            + (timebox.time_spent_validating_minutes / 60.0)
        )
        timebox_is_exhausted = elapsed_hours >= timebox.max_hours

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
                    f"Timebox limit of {timebox.max_hours}h reached, but gates remain unresolved: {', '.join(unresolved_gates)}.",
                )
            return (
                AssessmentStopStatus.IN_PROGRESS,
                f"Investigation in progress. Outstanding gates: {', '.join(unresolved_gates)}.",
            )

        # 2. High Profile: Stop after two consecutive source classes add no new information
        elif prof == "High":
            # Check novelty history if provided
            if recent_source_novelty_history and len(recent_source_novelty_history) >= 2:
                last_two = recent_source_novelty_history[-2:]
                if last_two == [False, False]:
                    return (
                        AssessmentStopStatus.COMPLETED_CONDITION_MET,
                        "Stopping condition met: Two consecutive completed source classes yielded no novel information.",
                    )

            # Check if all sources exhausted
            uncompleted = [
                s.source_id
                for s in coverage_records.values()
                if s.is_required
                and s.status in (CoverageState.NOT_STARTED, CoverageState.SEARCH_ATTEMPTED, CoverageState.RESULTS_FOUND)
            ]
            if not uncompleted:
                return (
                    AssessmentStopStatus.COMPLETED_CONDITION_MET,
                    "All required source classes in High depth profile have been completed.",
                )

            if timebox_is_exhausted:
                return (
                    AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE,
                    f"Timebox limit of {timebox.max_hours}h reached with {len(uncompleted)} required sources incomplete.",
                )

            return (
                AssessmentStopStatus.IN_PROGRESS,
                f"High investigation in progress ({len(uncompleted)} required sources remaining).",
            )

        # 3. Medium Profile: Stop after one CONFIRMED signal or source exhaustion
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
                if s.is_required
                and s.status in (CoverageState.NOT_STARTED, CoverageState.SEARCH_ATTEMPTED, CoverageState.RESULTS_FOUND)
            ]
            if not uncompleted_med:
                return (
                    AssessmentStopStatus.COMPLETED_CONDITION_MET,
                    "Applicable Medium source classes exhausted without confirming positive signal.",
                )

            if timebox_is_exhausted:
                return (
                    AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE,
                    f"Timebox limit of {timebox.max_hours}h reached with {len(uncompleted_med)} sources incomplete.",
                )

            return (
                AssessmentStopStatus.IN_PROGRESS,
                f"Medium investigation in progress ({len(uncompleted_med)} sources remaining).",
            )

        # 4. Low Profile: Complete lightweight pass
        elif prof == "Low":
            uncompleted_low = [
                s.source_id
                for s in coverage_records.values()
                if s.is_required
                and s.status in (CoverageState.NOT_STARTED, CoverageState.SEARCH_ATTEMPTED, CoverageState.RESULTS_FOUND)
            ]
            if not uncompleted_low:
                return (
                    AssessmentStopStatus.COMPLETED_CONDITION_MET,
                    "Configured lightweight source pass (S1, S3, S12) completed. Reassessment schedule logged.",
                )

            if timebox_is_exhausted:
                return (
                    AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE,
                    f"Timebox limit of {timebox.max_hours}h reached before finishing lightweight pass.",
                )

            return (
                AssessmentStopStatus.IN_PROGRESS,
                f"Low pass in progress ({len(uncompleted_low)} sources remaining).",
            )

        return AssessmentStopStatus.IN_PROGRESS, "Investigation status undetermined."
