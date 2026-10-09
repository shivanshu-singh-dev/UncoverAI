"""Source-class coverage tracking and audit status management."""

from meridian_assessment.models.osint_plan import (
    CoverageState,
    SourceCoverageRecord,
)


class OSINTCoverageTracker:
    """Tracks investigation progress across planned source classes."""

    def __init__(self, records: dict[str, SourceCoverageRecord]) -> None:
        self.records = records

    def update_source_status(
        self,
        source_id: str,
        new_state: CoverageState,
        candidate_urls: int = 0,
        reviewed_urls: int = 0,
        produced_new_evidence: bool = False,
        notes: str = "",
    ) -> None:
        """Update coverage audit state for a specific source class."""
        s_id = source_id.upper()
        if s_id in self.records:
            rec = self.records[s_id]
            rec.status = new_state
            rec.candidate_urls_count = candidate_urls
            rec.reviewed_urls_count = reviewed_urls
            rec.produced_new_evidence = produced_new_evidence
            if notes:
                rec.notes = notes

    def get_summary(self) -> dict[str, int]:
        """Compute coverage progress metrics."""
        total_required = sum(1 for r in self.records.values() if r.is_required)
        not_started = sum(1 for r in self.records.values() if r.status == CoverageState.NOT_STARTED)
        in_progress = sum(
            1 for r in self.records.values()
            if r.status in (CoverageState.SEARCH_ATTEMPTED, CoverageState.RESULTS_FOUND, CoverageState.RELEVANT_SOURCE_IDENTIFIED)
        )
        completed = sum(
            1 for r in self.records.values()
            if r.status in (CoverageState.REVIEWED_EVIDENCE_RECORDED, CoverageState.REVIEWED_NO_EVIDENCE)
        )
        unavailable = sum(1 for r in self.records.values() if r.status == CoverageState.UNAVAILABLE)
        manual_review = sum(1 for r in self.records.values() if r.status == CoverageState.REQUIRES_MANUAL_REVIEW)

        total_urls_found = sum(r.candidate_urls_count for r in self.records.values())
        total_urls_reviewed = sum(r.reviewed_urls_count for r in self.records.values())

        return {
            "total_planned_sources": len(self.records),
            "total_required_sources": total_required,
            "not_started": not_started,
            "in_progress": in_progress,
            "completed": completed,
            "unavailable": unavailable,
            "requires_manual_review": manual_review,
            "total_urls_found": total_urls_found,
            "total_urls_reviewed": total_urls_reviewed,
        }
