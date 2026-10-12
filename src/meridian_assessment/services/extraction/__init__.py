"""Level 4 Web/PDF extraction and durable evidence repository package."""

from meridian_assessment.services.extraction.extractor import (
    ContentFetchAdapter,
    DisabledOfflineFetchAdapter,
    FetchResponse,
    HTMLTextExtractor,
    OSINTExtractionService,
    PDFDocumentExtractor,
    ResilientPublicFetchAdapter,
    ScraperAPIFetchAdapter,
    StubFetchAdapter,
    UrllibPublicFetchAdapter,
    build_fetch_adapter_from_env,
    compute_sha256,
    validate_safe_public_url,
    validate_source_integrity,
)
from meridian_assessment.services.extraction.evidence_repository import (
    DEFAULT_ARTIFACT_STORE_DIR,
    DEFAULT_EVIDENCE_DB_PATH,
    SCHEMA_VERSION,
    EvidenceRepository,
)

__all__ = [
    "ContentFetchAdapter",
    "DisabledOfflineFetchAdapter",
    "FetchResponse",
    "HTMLTextExtractor",
    "OSINTExtractionService",
    "PDFDocumentExtractor",
    "ResilientPublicFetchAdapter",
    "ScraperAPIFetchAdapter",
    "StubFetchAdapter",
    "UrllibPublicFetchAdapter",
    "build_fetch_adapter_from_env",
    "compute_sha256",
    "validate_safe_public_url",
    "validate_source_integrity",
    "DEFAULT_ARTIFACT_STORE_DIR",
    "DEFAULT_EVIDENCE_DB_PATH",
    "SCHEMA_VERSION",
    "EvidenceRepository",
]
