"""Durable local SQLite + artifact Evidence Repository for the Meridian POC (Level 4.4 & Section 23).

Provides:
  - Schema creation and migration version tracking (`schema_migrations`).
  - Durable storage and retrieval of `ExtractedDocument`, `EvidenceRecord`,
    `HumanValidationRecord`, and `VendorEndToEndAssessmentReport`.
  - Explicit separation between collected/unreviewed evidence (`is_reviewed_and_accepted = False`)
    and analyst-reviewed/accepted evidence (`is_reviewed_and_accepted = True`).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional

from meridian_assessment.models.ai_risk import (
    EvidenceRecord,
    ExtractedDocument,
    HumanValidationRecord,
    VendorEndToEndAssessmentReport,
)

DEFAULT_EVIDENCE_DB_PATH = Path("data/output/evidence_store/meridian_evidence.db")
DEFAULT_ARTIFACT_STORE_DIR = Path("data/output/evidence_store/artifacts")
SCHEMA_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceRepository:
    """SQLite-backed repository for extracted OSINT documents, evidence records, validations, and reports."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_EVIDENCE_DB_PATH,
        artifact_dir: Optional[Path] = DEFAULT_ARTIFACT_STORE_DIR,
    ) -> None:
        self.db_path_str = str(db_path)
        self.is_memory = self.db_path_str == ":memory:"
        if not self.is_memory:
            Path(self.db_path_str).parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = artifact_dir
        if self.artifact_dir is not None and not self.is_memory:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._persistent_conn: Optional[sqlite3.Connection] = (
            sqlite3.connect(":memory:", check_same_thread=False) if self.is_memory else None
        )
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        return sqlite3.connect(self.db_path_str)

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS extracted_documents (
                    document_id TEXT PRIMARY KEY,
                    vendor_id TEXT NOT NULL,
                    canonical_url TEXT NOT NULL,
                    source_class TEXT NOT NULL,
                    extraction_status TEXT NOT NULL,
                    content_sha256 TEXT,
                    retrieved_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_records (
                    evidence_id TEXT PRIMARY KEY,
                    vendor_id TEXT NOT NULL,
                    document_id TEXT,
                    source_class TEXT NOT NULL,
                    canonical_url TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    evidence_strength TEXT NOT NULL,
                    is_reviewed_and_accepted INTEGER NOT NULL DEFAULT 0,
                    is_synthetic_test_fixture INTEGER NOT NULL DEFAULT 0,
                    retrieved_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS human_validations (
                    validation_id TEXT PRIMARY KEY,
                    vendor_id TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vendor_reports (
                    report_id TEXT PRIMARY KEY,
                    vendor_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    is_synthetic_evidence_run INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cur.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _utc_now_iso()),
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    def get_schema_version(self) -> str:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1"
            ).fetchone()
            return str(row[0]) if row else "UNKNOWN"
        finally:
            if self._persistent_conn is None:
                conn.close()

    # ─── Extracted Documents ──────────────────────────────────────────────────

    def save_document(self, doc: ExtractedDocument) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO extracted_documents
                (document_id, vendor_id, canonical_url, source_class, extraction_status, content_sha256, retrieved_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.document_id,
                    doc.vendor_id,
                    doc.canonical_url,
                    doc.source_class,
                    doc.extraction_status.value,
                    doc.content_sha256,
                    doc.retrieved_at,
                    json.dumps(doc.model_dump(mode="json")),
                ),
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    def list_documents(self, vendor_id: str) -> list[ExtractedDocument]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM extracted_documents WHERE vendor_id = ? ORDER BY document_id",
                (vendor_id,),
            ).fetchall()
            return [ExtractedDocument.model_validate(json.loads(r[0])) for r in rows]
        finally:
            if self._persistent_conn is None:
                conn.close()

    # ─── Evidence Records (Collected vs Reviewed/Accepted) ────────────────────

    def save_evidence(self, record: EvidenceRecord) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO evidence_records
                (evidence_id, vendor_id, document_id, source_class, canonical_url, content_sha256,
                 evidence_strength, is_reviewed_and_accepted, is_synthetic_test_fixture, retrieved_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.evidence_id,
                    record.vendor_id,
                    record.document_id,
                    record.source_class,
                    record.canonical_url,
                    record.content_sha256,
                    record.evidence_strength.value,
                    1 if record.is_reviewed_and_accepted else 0,
                    1 if record.is_synthetic_test_fixture else 0,
                    record.retrieved_at,
                    json.dumps(record.model_dump(mode="json")),
                ),
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    def save_evidence_batch(self, records: list[EvidenceRecord]) -> None:
        for rec in records:
            self.save_evidence(rec)

    def get_evidence(self, evidence_id: str) -> Optional[EvidenceRecord]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM evidence_records WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            return EvidenceRecord.model_validate(json.loads(row[0])) if row else None
        finally:
            if self._persistent_conn is None:
                conn.close()

    def list_evidence(
        self,
        vendor_id: str,
        *,
        only_reviewed_and_accepted: bool = False,
    ) -> list[EvidenceRecord]:
        conn = self._connect()
        try:
            if only_reviewed_and_accepted:
                rows = conn.execute(
                    """
                    SELECT payload_json FROM evidence_records
                    WHERE vendor_id = ? AND is_reviewed_and_accepted = 1
                    ORDER BY evidence_id
                    """,
                    (vendor_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload_json FROM evidence_records WHERE vendor_id = ? ORDER BY evidence_id",
                    (vendor_id,),
                ).fetchall()
            return [EvidenceRecord.model_validate(json.loads(r[0])) for r in rows]
        finally:
            if self._persistent_conn is None:
                conn.close()

    def mark_evidence_reviewed(
        self,
        evidence_id: str,
        *,
        accepted: bool = True,
        reviewer_notes: Optional[str] = None,
    ) -> Optional[EvidenceRecord]:
        """Update review acceptance status without altering the original captured text or hash."""
        rec = self.get_evidence(evidence_id)
        if rec is None:
            return None
        rec.is_reviewed_and_accepted = accepted
        if reviewer_notes is not None:
            rec.reviewer_notes = reviewer_notes
        self.save_evidence(rec)
        return rec

    # ─── Human Validations & Reports ──────────────────────────────────────────

    def save_human_validation(self, validation: HumanValidationRecord) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO human_validations
                (validation_id, vendor_id, target_type, target_id, reviewed_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    validation.validation_id,
                    validation.vendor_id,
                    validation.target_type,
                    validation.target_id,
                    validation.reviewed_at,
                    json.dumps(validation.model_dump(mode="json")),
                ),
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    def list_human_validations(self, vendor_id: str) -> list[HumanValidationRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM human_validations WHERE vendor_id = ? ORDER BY reviewed_at",
                (vendor_id,),
            ).fetchall()
            return [HumanValidationRecord.model_validate(json.loads(r[0])) for r in rows]
        finally:
            if self._persistent_conn is None:
                conn.close()

    def save_vendor_report(self, report: VendorEndToEndAssessmentReport) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO vendor_reports
                (report_id, vendor_id, generated_at, is_synthetic_evidence_run, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.vendor_id,
                    report.generated_at,
                    1 if report.is_synthetic_evidence_run else 0,
                    json.dumps(report.model_dump(mode="json")),
                ),
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    def get_latest_vendor_report(self, vendor_id: str) -> Optional[VendorEndToEndAssessmentReport]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM vendor_reports WHERE vendor_id = ? ORDER BY generated_at DESC LIMIT 1",
                (vendor_id,),
            ).fetchone()
            return VendorEndToEndAssessmentReport.model_validate(json.loads(row[0])) if row else None
        finally:
            if self._persistent_conn is None:
                conn.close()

    def load_latest_reports_by_vendor(self) -> dict[str, VendorEndToEndAssessmentReport]:
        """Return a mapping of `vendor_id -> latest VendorEndToEndAssessmentReport` across all persisted reports."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT vendor_id, payload_json FROM vendor_reports ORDER BY generated_at ASC"
            ).fetchall()
            latest: dict[str, VendorEndToEndAssessmentReport] = {}
            for vid, pjson in rows:
                try:
                    latest[str(vid)] = VendorEndToEndAssessmentReport.model_validate(json.loads(pjson))
                except Exception:
                    continue
            return latest
        finally:
            if self._persistent_conn is None:
                conn.close()
