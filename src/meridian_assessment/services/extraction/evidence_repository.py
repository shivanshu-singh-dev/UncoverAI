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
import hashlib
import json
from pathlib import Path
import re
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
CLEANING_LOGIC_VERSION = "2.0"
CLASSIFIER_PROMPT_VERSION = "2.0"
EMBEDDING_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
SYNTHETIC_FIXTURE_HASH_PREFIXES = frozenset({"e3cf1468698b", "02b5b6376a51", "9f4122cb1170", "a84f39c02d11"})
_ARTIFACT_FILENAME_RE = re.compile(r"^DOC_(?P<vendor_id>V-\d+)_(?P<seq>\d+)_(?P<sha12>[0-9a-f]{12})\.(?P<ext>html|pdf|txt)$")



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
                """
                CREATE TABLE IF NOT EXISTS chunk_embeddings_cache (
                    cache_key TEXT PRIMARY KEY,
                    chunk_sha256 TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    cleaning_version TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS classification_cache (
                    cache_key TEXT PRIMARY KEY,
                    vendor_id TEXT NOT NULL,
                    passage_hash TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    classifier_version TEXT NOT NULL,
                    analysis_state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS assessment_checkpoints (
                    vendor_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cleaning_version TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    classifier_version TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
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

    def get_latest_vendor_report(
        self,
        vendor_id: str,
        *,
        only_live: bool = False,
    ) -> Optional[VendorEndToEndAssessmentReport]:
        conn = self._connect()
        try:
            if only_live:
                row = conn.execute(
                    "SELECT payload_json FROM vendor_reports WHERE vendor_id = ? AND is_synthetic_evidence_run = 0 ORDER BY generated_at DESC LIMIT 1",
                    (vendor_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT payload_json FROM vendor_reports WHERE vendor_id = ? ORDER BY generated_at DESC LIMIT 1",
                    (vendor_id,),
                ).fetchone()
            return VendorEndToEndAssessmentReport.model_validate(json.loads(row[0])) if row else None
        finally:
            if self._persistent_conn is None:
                conn.close()

    def load_latest_reports_by_vendor(
        self,
        *,
        only_live: bool = False,
    ) -> dict[str, VendorEndToEndAssessmentReport]:
        """Return a mapping of `vendor_id -> latest VendorEndToEndAssessmentReport` across all persisted reports."""
        conn = self._connect()
        try:
            if only_live:
                rows = conn.execute(
                    "SELECT vendor_id, payload_json FROM vendor_reports WHERE is_synthetic_evidence_run = 0 ORDER BY generated_at ASC"
                ).fetchall()
            else:
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

    # ─── Artifact Reuse & Integrity Verification (Level 4) ────────────────────

    def verify_artifact_integrity(
        self,
        vendor_id: str,
        artifact_path: Path | str,
        *,
        expected_sha256: Optional[str] = None,
    ) -> tuple[bool, Optional[bytes], str]:
        """Verify an artifact exists, belongs to vendor_id, is non-empty, and matches its SHA-256."""
        path = Path(artifact_path)
        if not path.exists() or not path.is_file():
            return False, None, "Artifact file does not exist on disk."
        m = _ARTIFACT_FILENAME_RE.match(path.name)
        if m is not None and m.group("vendor_id") != vendor_id:
            return False, None, f"Artifact vendor mismatch: expected {vendor_id}, got {m.group('vendor_id')}."
        try:
            raw_bytes = path.read_bytes()
        except Exception as exc:
            return False, None, f"Failed to read artifact bytes: {exc}"
        if not raw_bytes:
            return False, None, "Artifact file is empty (0 bytes)."
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if expected_sha256 and actual_sha != expected_sha256:
            return False, None, f"Artifact SHA-256 mismatch: expected {expected_sha256[:12]}, got {actual_sha[:12]}."
        if m is not None and not actual_sha.startswith(m.group("sha12")):
            return (
                False,
                None,
                f"Artifact filename prefix mismatch: expected {m.group('sha12')}, got {actual_sha[:12]}.",
            )
        return True, raw_bytes, actual_sha

    def find_cached_document(
        self,
        vendor_id: str,
        canonical_url: Optional[str] = None,
        *,
        document_id: Optional[str] = None,
        cleaning_version: str = CLEANING_LOGIC_VERSION,
        allow_synthetic: bool = False,
    ) -> Optional[ExtractedDocument]:
        """Look up a previously extracted document for (vendor_id, canonical_url or document_id) with verified integrity."""
        conn = self._connect()
        try:
            rows = []
            if canonical_url:
                rows = conn.execute(
                    """
                    SELECT payload_json FROM extracted_documents
                    WHERE vendor_id = ? AND canonical_url = ?
                    ORDER BY retrieved_at DESC
                    """,
                    (vendor_id, canonical_url),
                ).fetchall()
            if not rows and document_id:
                rows = conn.execute(
                    """
                    SELECT payload_json FROM extracted_documents
                    WHERE vendor_id = ? AND document_id = ?
                    ORDER BY retrieved_at DESC
                    """,
                    (vendor_id, document_id),
                ).fetchall()
        finally:
            if self._persistent_conn is None:
                conn.close()

        for (pjson,) in rows:
            try:
                doc = ExtractedDocument.model_validate(json.loads(pjson))
            except Exception:
                continue
            if not allow_synthetic and doc.content_sha256 and doc.content_sha256[:12] in SYNTHETIC_FIXTURE_HASH_PREFIXES:
                continue
            if doc.extraction_status.value != "EXTRACTED" or not doc.cleaned_text:
                continue
            if doc.cleaning_version != cleaning_version:
                # If raw artifact exists and passes verification, caller can re-extract with new cleaning version
                continue
            if doc.artifact_path:
                ok, _, _ = self.verify_artifact_integrity(
                    vendor_id,
                    doc.artifact_path,
                    expected_sha256=doc.content_sha256,
                )
                if not ok:
                    continue
            doc.reused_from_cache = True
            return doc
        return None

    def find_cached_artifact_bytes_for_document(
        self,
        vendor_id: str,
        canonical_url: str,
        *,
        allow_synthetic: bool = False,
    ) -> Optional[tuple[ExtractedDocument, bytes]]:
        """Find a stored document record whose raw artifact bytes on disk pass integrity verification."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT payload_json FROM extracted_documents
                WHERE vendor_id = ? AND canonical_url = ?
                ORDER BY retrieved_at DESC
                """,
                (vendor_id, canonical_url),
            ).fetchall()
        finally:
            if self._persistent_conn is None:
                conn.close()

        for (pjson,) in rows:
            try:
                doc = ExtractedDocument.model_validate(json.loads(pjson))
            except Exception:
                continue
            if not allow_synthetic and doc.content_sha256 and doc.content_sha256[:12] in SYNTHETIC_FIXTURE_HASH_PREFIXES:
                continue
            if not doc.artifact_path:
                continue
            ok, raw_bytes, _ = self.verify_artifact_integrity(
                vendor_id,
                doc.artifact_path,
                expected_sha256=doc.content_sha256,
            )
            if ok and raw_bytes is not None:
                return doc, raw_bytes
        return None


    def discover_verified_vendor_artifacts(
        self,
        vendor_id: str,
        *,
        exclude_synthetic: bool = True,
        include_synthetic: Optional[bool] = None,
    ) -> list[tuple[Path, bytes, str, int, str]]:
        """Discover and integrity-verify all local artifact files on disk for `vendor_id`.

        Returns a sorted list of tuples:
            (artifact_path, raw_bytes, full_sha256, sequence_number, file_extension)
        """
        if include_synthetic is not None:
            exclude_synthetic = not bool(include_synthetic)
        if self.artifact_dir is None or not self.artifact_dir.exists():
            return []
        verified: list[tuple[Path, bytes, str, int, str]] = []
        for path in sorted(self.artifact_dir.glob(f"DOC_{vendor_id}_*")):
            m = _ARTIFACT_FILENAME_RE.match(path.name)
            if m is None or m.group("vendor_id") != vendor_id:
                continue
            sha12 = m.group("sha12")
            if exclude_synthetic and sha12 in SYNTHETIC_FIXTURE_HASH_PREFIXES:
                continue
            ok, raw_bytes, full_sha = self.verify_artifact_integrity(vendor_id, path)
            if not ok or raw_bytes is None:
                continue
            verified.append((path, raw_bytes, full_sha, int(m.group("seq")), m.group("ext")))
        verified.sort(key=lambda item: (item[3], item[0].name))
        return verified

    # ─── Level 5 Embedding Cache ──────────────────────────────────────────────

    @staticmethod
    def _embedding_cache_key(chunk_sha256: str, model_name: str, cleaning_version: str) -> str:
        raw = f"{chunk_sha256}|{model_name}|{cleaning_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cached_embeddings(
        self,
        chunk_hashes: list[str],
        model_name: str,
        *,
        cleaning_version: str = CLEANING_LOGIC_VERSION,
    ) -> dict[str, list[float]]:
        """Return cached embedding vectors keyed by chunk_sha256 for the given model and cleaning version."""
        if not chunk_hashes:
            return {}
        unique_hashes = list(dict.fromkeys(chunk_hashes))
        key_to_hash = {
            self._embedding_cache_key(ch, model_name, cleaning_version): ch for ch in unique_hashes
        }
        keys = list(key_to_hash.keys())
        result: dict[str, list[float]] = {}
        conn = self._connect()
        try:
            # Query in batches of 500 to stay below SQLite parameter limits
            for i in range(0, len(keys), 500):
                batch_keys = keys[i : i + 500]
                placeholders = ",".join("?" for _ in batch_keys)
                rows = conn.execute(
                    f"SELECT cache_key, vector_json FROM chunk_embeddings_cache WHERE cache_key IN ({placeholders})",
                    batch_keys,
                ).fetchall()
                for ck, vec_json in rows:
                    ch = key_to_hash.get(str(ck))
                    if ch is not None:
                        try:
                            vec = json.loads(vec_json)
                            if isinstance(vec, list) and vec:
                                result[ch] = [float(x) for x in vec]
                        except Exception:
                            continue
            return result
        finally:
            if self._persistent_conn is None:
                conn.close()

    def save_cached_embeddings(
        self,
        embeddings_by_hash: dict[str, list[float]],
        model_name: str,
        *,
        cleaning_version: str = CLEANING_LOGIC_VERSION,
    ) -> None:
        """Persist embedding vectors keyed by chunk_sha256."""
        if not embeddings_by_hash:
            return
        now = _utc_now_iso()
        rows_to_insert = []
        for ch, vec in embeddings_by_hash.items():
            if not vec:
                continue
            ck = self._embedding_cache_key(ch, model_name, cleaning_version)
            rows_to_insert.append(
                (ck, ch, model_name, cleaning_version, len(vec), json.dumps(vec), now)
            )
        if not rows_to_insert:
            return
        conn = self._connect()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_embeddings_cache
                (cache_key, chunk_sha256, model_name, cleaning_version, dimension, vector_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    # ─── Level 6 Classification Cache ─────────────────────────────────────────

    @staticmethod
    def _classification_cache_key(
        vendor_id: str,
        passage_hash: str,
        source_class: str,
        service_scope: str,
        model_name: str,
        classifier_version: str,
    ) -> str:
        raw = f"{vendor_id}|{passage_hash}|{source_class}|{service_scope}|{model_name}|{classifier_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cached_classification(
        self,
        vendor_id: str,
        passage_hash: str,
        source_class: str,
        service_scope: str,
        model_name: str,
        *,
        classifier_version: str = CLASSIFIER_PROMPT_VERSION,
    ) -> Optional[dict[str, Any]]:
        """Retrieve a cached classification result for a normalized passage hash."""
        ck = self._classification_cache_key(
            vendor_id, passage_hash, source_class, service_scope, model_name, classifier_version
        )
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT analysis_state, payload_json FROM classification_cache WHERE cache_key = ?",
                (ck,),
            ).fetchone()
            if not row:
                return None
            payload = json.loads(row[1])
            if isinstance(payload, dict):
                payload["analysis_state"] = str(row[0])
                return payload
            return None
        except Exception:
            return None
        finally:
            if self._persistent_conn is None:
                conn.close()

    def save_cached_classification(
        self,
        vendor_id: str,
        passage_hash: str,
        source_class: str,
        service_scope: str,
        model_name: str,
        analysis_state: str,
        payload: dict[str, Any],
        *,
        classifier_version: str = CLASSIFIER_PROMPT_VERSION,
    ) -> None:
        """Save a classification result to cache."""
        ck = self._classification_cache_key(
            vendor_id, passage_hash, source_class, service_scope, model_name, classifier_version
        )
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO classification_cache
                (cache_key, vendor_id, passage_hash, model_name, classifier_version, analysis_state, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ck,
                    vendor_id,
                    passage_hash,
                    model_name,
                    classifier_version,
                    analysis_state,
                    json.dumps(payload),
                    _utc_now_iso(),
                ),
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    # ─── Resumable Assessment Checkpointing (Section 7) ───────────────────────

    def save_checkpoint(
        self,
        vendor_id: str,
        run_id: str,
        status: str = "IN_PROGRESS",
        payload: Optional[dict[str, Any]] = None,
        *,
        completed_level: Optional[int] = None,
        checkpoint_state: Optional[dict[str, Any]] = None,
        cleaning_version: str = CLEANING_LOGIC_VERSION,
        embedding_model: str = EMBEDDING_MODEL_DEFAULT,
        classifier_version: str = CLASSIFIER_PROMPT_VERSION,
    ) -> None:
        """Persist a resumable assessment checkpoint for `vendor_id`."""
        resolved_payload: dict[str, Any] = dict(payload or checkpoint_state or {})
        if completed_level is not None:
            resolved_payload["completed_level"] = completed_level
            if status == "IN_PROGRESS" and "status" not in resolved_payload:
                status = f"LEVEL_{completed_level}_CHECKPOINT"
            elif "status" in resolved_payload:
                status = str(resolved_payload["status"])
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO assessment_checkpoints
                (vendor_id, run_id, status, cleaning_version, embedding_model, classifier_version, updated_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vendor_id,
                    run_id,
                    status,
                    cleaning_version,
                    embedding_model,
                    classifier_version,
                    _utc_now_iso(),
                    json.dumps(resolved_payload),
                ),
            )
            conn.commit()
        finally:
            if self._persistent_conn is None:
                conn.close()

    def load_checkpoint(
        self,
        vendor_id: str,
        *,
        cleaning_version: str = CLEANING_LOGIC_VERSION,
        embedding_model: str = EMBEDDING_MODEL_DEFAULT,
        classifier_version: str = CLASSIFIER_PROMPT_VERSION,
    ) -> Optional[dict[str, Any]]:
        """Load a persisted checkpoint for `vendor_id` if processing versions match."""
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT run_id, status, cleaning_version, embedding_model, classifier_version, updated_at, payload_json
                FROM assessment_checkpoints
                WHERE vendor_id = ?
                """,
                (vendor_id,),
            ).fetchone()
            if not row:
                return None
            run_id, status, c_ver, e_mod, clf_ver, updated_at, pjson = row
            if c_ver != cleaning_version or e_mod != embedding_model or clf_ver != classifier_version:
                return None
            data = json.loads(pjson)
            if not isinstance(data, dict):
                return None
            data["run_id"] = str(run_id)
            data["status"] = str(status)
            data["cleaning_version"] = str(c_ver)
            data["embedding_model"] = str(e_mod)
            data["classifier_version"] = str(clf_ver)
            data["updated_at"] = str(updated_at)
            return data
        except Exception:
            return None
        finally:
            if self._persistent_conn is None:
                conn.close()

