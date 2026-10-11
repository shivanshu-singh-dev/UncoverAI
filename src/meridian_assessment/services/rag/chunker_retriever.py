"""Level 5 — Semantic Filtering & RAG Chunking Pipeline.

Implements:
  - Configurable token-aware document/evidence chunking (~500 tokens, 50-token overlap by default)
    preserving parent `document_id`, `evidence_id`, `vendor_id`, `url`, `source_class`,
    `retrieved_at`, `chunk_index`, exact character offsets (`char_start`, `char_end`), and SHA-256 hash.
  - Configurable `EmbeddingProvider` interface compatible with `sentence-transformers/all-MiniLM-L6-v2`
    that never downloads weights automatically and remains completely separate from criticality scoring.
  - Explicit `MODEL_UNAVAILABLE_NOT_CONFIGURED` state when the embedding model is not loaded.
  - Injectable `DeterministicFakeEmbeddingProvider` for offline testing of cosine similarity,
    thresholds, and top-k concept retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from pathlib import Path
import re
from typing import Any, Optional, Sequence
import yaml

from meridian_assessment.models.ai_risk import (
    DocumentChunk,
    EmbeddingStatus,
    EvidenceRecord,
    ExtractedDocument,
    RetrievedChunkMatch,
    SemanticRetrievalSummary,
)
from meridian_assessment.services.extraction.extractor import compute_sha256

_DEFAULT_AI_RISK_CONFIG_PATH = Path("config/ai_risk_scoring.yaml")


# ─── Embedding Provider Abstraction ───────────────────────────────────────────


class EmbeddingProvider(ABC):
    """Abstract interface for Level 5 RAG chunk & query embeddings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Return `(True, info)` if the embedding model is loaded and ready, else `(False, reason)`."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a sequence of texts into numeric vectors."""


class UnconfiguredEmbeddingProvider(EmbeddingProvider):
    """Default offline embedding provider when no local embedding model is activated."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reason: str = (
            "Embedding model 'sentence-transformers/all-MiniLM-L6-v2' is not activated in offline mode "
            "(auto_load_model=false). Set MERIDIAN_ENABLE_RAG_EMBEDDINGS=true after local model installation."
        ),
    ) -> None:
        self.model_name = model_name
        self.reason = reason

    def is_available(self) -> tuple[bool, str]:
        return False, self.reason

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError(self.reason)


class DeterministicFakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic keyword-vector embedding provider for offline unit and integration tests.

    Never downloads weights or calls external services.
    """

    VOCAB_DIMENSIONS: tuple[tuple[str, ...], ...] = (
        ("train", "training", "fine-tune", "fine-tuning", "improve", "dataset"),
        ("subprocessor", "openai", "anthropic", "bedrock", "vertex", "cohere", "mistral"),
        ("inference", "endpoint", "completions", "embeddings", "vector", "rag", "llm", "genai"),
        ("retention", "zdr", "zero-data-retention", "zero", "delete", "deletion", "30-day", "logs"),
        ("pii", "ssn", "financial", "account", "credentials", "customer", "sensitive", "regulated"),
        ("autonomous", "agent", "agentic", "tool-calling", "approval", "automated", "decision"),
        ("governance", "isolation", "tenant", "injection", "guardrails", "soc2", "iso42001"),
    )

    def __init__(self, model_name: str = "test-fake/all-MiniLM-L6-v2-fixture") -> None:
        self.model_name = model_name

    def is_available(self) -> tuple[bool, str]:
        return True, f"Deterministic test embedding provider ({self.model_name}) active."

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for txt in texts:
            words = re.findall(r"[a-z0-9-]+", (txt or "").lower())
            raw_vec: list[float] = []
            for dim_terms in self.VOCAB_DIMENSIONS:
                score = sum(1.0 for w in words if any(t in w for t in dim_terms))
                raw_vec.append(score)
            norm = math.sqrt(sum(v * v for v in raw_vec))
            if norm > 0:
                vectors.append([v / norm for v in raw_vec])
            else:
                vectors.append([0.0 for _ in raw_vec])
        return vectors


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Configurable wrapper for `sentence-transformers/all-MiniLM-L6-v2`.

    By default `allow_download=False` so it never triggers an unapproved network model download.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        allow_download: bool = False,
        loaded_model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.allow_download = allow_download
        self._model = loaded_model
        self._load_error: Optional[str] = None

    def is_available(self) -> tuple[bool, str]:
        if self._model is not None:
            return True, f"Loaded {self.model_name}."
        if not self.allow_download:
            return (
                False,
                f"Model '{self.model_name}' not pre-loaded and network downloads are disabled.",
            )
        return True, f"Model '{self.model_name}' configured for lazy loading."

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ok, msg = self.is_available()
        if not ok:
            raise RuntimeError(msg)
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, local_files_only=not self.allow_download)
        arr = self._model.encode(list(texts))
        return [[float(x) for x in row] for row in arr]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_prod = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(dot_prod / (norm_a * norm_b))


# ─── RAG Chunker & Semantic Retriever ─────────────────────────────────────────


class RAGChunkerAndRetriever:
    """Chunks extracted evidence/documents and retrieves semantically relevant passages for risk concepts."""

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        *,
        config_path: Path = _DEFAULT_AI_RISK_CONFIG_PATH,
        chunk_size_tokens: Optional[int] = None,
        chunk_overlap_tokens: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        top_k_per_concept: Optional[int] = None,
    ) -> None:
        self.config_path = config_path
        cfg = self._load_rag_config(config_path)
        self.chunk_size_tokens = (
            int(chunk_size_tokens)
            if chunk_size_tokens is not None
            else int(cfg.get("chunk_size_tokens", 500))
        )
        self.chunk_overlap_tokens = (
            int(chunk_overlap_tokens)
            if chunk_overlap_tokens is not None
            else int(cfg.get("chunk_overlap_tokens", 50))
        )
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            self.chunk_overlap_tokens = max(0, self.chunk_size_tokens // 5)

        self.similarity_threshold = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else float(cfg.get("similarity_threshold", 0.45))
        )
        self.top_k_per_concept = (
            int(top_k_per_concept)
            if top_k_per_concept is not None
            else int(cfg.get("top_k_per_concept", 3))
        )
        self.model_name: str = str(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
        self.risk_concepts: list[dict[str, Any]] = list(cfg.get("risk_query_concepts", []))
        self.embedding_provider: EmbeddingProvider = embedding_provider or UnconfiguredEmbeddingProvider(
            model_name=self.model_name
        )

    @staticmethod
    def _load_rag_config(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    return data.get("rag_chunking", {})
            except Exception:
                return {}
        return {}

    def chunk_evidence_records(
        self,
        evidence_records: Sequence[EvidenceRecord],
    ) -> list[DocumentChunk]:
        """Chunk a list of EvidenceRecord objects while preserving parent IDs and exact character offsets."""
        chunks: list[DocumentChunk] = []
        for ev in evidence_records:
            text = ev.verbatim_excerpt or ""
            if not text.strip():
                continue

            word_matches = list(re.finditer(r"\S+", text))
            if not word_matches:
                continue

            step = max(1, self.chunk_size_tokens - self.chunk_overlap_tokens)
            c_idx = 0
            for start_w in range(0, len(word_matches), step):
                end_w = min(len(word_matches), start_w + self.chunk_size_tokens)
                rel_char_start = word_matches[start_w].start()
                rel_char_end = word_matches[end_w - 1].end()
                chunk_text = text[rel_char_start:rel_char_end]

                base_offset = ev.char_start or 0
                abs_start = base_offset + rel_char_start
                abs_end = base_offset + rel_char_end
                token_est = end_w - start_w

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"CHK_{ev.evidence_id}_{c_idx:03d}",
                        document_id=ev.document_id or f"DOC_{ev.vendor_id}",
                        evidence_id=ev.evidence_id,
                        vendor_id=ev.vendor_id,
                        url=ev.canonical_url,
                        source_class=ev.source_class,
                        retrieved_at=ev.retrieved_at,
                        chunk_index=c_idx,
                        text=chunk_text,
                        char_start=abs_start,
                        char_end=abs_end,
                        token_estimate=token_est,
                        content_sha256=compute_sha256(chunk_text),
                        section_heading=ev.section_heading,
                        page_number=ev.page_number,
                        embedding_status=EmbeddingStatus.UNAVAILABLE_NOT_CONFIGURED,
                        embedding_model_version=None,
                    )
                )
                c_idx += 1
                if end_w >= len(word_matches):
                    break

        return chunks

    def retrieve_relevant_chunks(
        self,
        evidence_records: Sequence[EvidenceRecord],
    ) -> tuple[list[DocumentChunk], SemanticRetrievalSummary]:
        """Chunk evidence and run semantic retrieval if the embedding provider is available."""
        chunks = self.chunk_evidence_records(evidence_records)
        if not chunks:
            avail, msg = self.embedding_provider.is_available()
            return [], SemanticRetrievalSummary(
                status="NO_DOCUMENTS",
                model_name=self.embedding_provider.model_name,
                model_available=avail,
                chunk_size_tokens=self.chunk_size_tokens,
                chunk_overlap_tokens=self.chunk_overlap_tokens,
                chunks_created=0,
                chunks_embedded=0,
                matches=[],
                limitations=["No extracted evidence text available for chunking."] + ([] if avail else [msg]),
            )

        avail, avail_msg = self.embedding_provider.is_available()
        if not avail:
            for chk in chunks:
                chk.embedding_status = EmbeddingStatus.UNAVAILABLE_NOT_CONFIGURED
            return chunks, SemanticRetrievalSummary(
                status="MODEL_UNAVAILABLE_NOT_CONFIGURED",
                model_name=self.embedding_provider.model_name,
                model_available=False,
                chunk_size_tokens=self.chunk_size_tokens,
                chunk_overlap_tokens=self.chunk_overlap_tokens,
                chunks_created=len(chunks),
                chunks_embedded=0,
                matches=[],
                limitations=[avail_msg],
            )

        try:
            chunk_vectors = self.embedding_provider.encode([c.text for c in chunks])
            concept_queries = [str(rc.get("query", "")) for rc in self.risk_concepts]
            concept_vectors = self.embedding_provider.encode(concept_queries) if concept_queries else []
        except Exception as exc:
            for chk in chunks:
                chk.embedding_status = EmbeddingStatus.FAILED
            return chunks, SemanticRetrievalSummary(
                status="MODEL_UNAVAILABLE_NOT_CONFIGURED",
                model_name=self.embedding_provider.model_name,
                model_available=False,
                chunk_size_tokens=self.chunk_size_tokens,
                chunk_overlap_tokens=self.chunk_overlap_tokens,
                chunks_created=len(chunks),
                chunks_embedded=0,
                matches=[],
                limitations=[f"Embedding execution failed: {exc}"],
            )

        for chk in chunks:
            chk.embedding_status = EmbeddingStatus.EMBEDDED
            chk.embedding_model_version = self.embedding_provider.model_name

        matches: list[RetrievedChunkMatch] = []
        for rc, q_vec in zip(self.risk_concepts, concept_vectors):
            c_id = str(rc.get("id", "RC_UNKNOWN"))
            q_txt = str(rc.get("query", ""))
            scored: list[tuple[float, DocumentChunk]] = []
            for chk, c_vec in zip(chunks, chunk_vectors):
                sim = cosine_similarity(q_vec, c_vec)
                if sim >= self.similarity_threshold:
                    scored.append((sim, chk))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            for sim, chk in scored[: self.top_k_per_concept]:
                matches.append(
                    RetrievedChunkMatch(
                        chunk=chk,
                        concept_id=c_id,
                        concept_query=q_txt,
                        similarity_score=round(sim, 4),
                        retrieval_method="SEMANTIC_EMBEDDING",
                    )
                )

        return chunks, SemanticRetrievalSummary(
            status="COMPLETED",
            model_name=self.embedding_provider.model_name,
            model_available=True,
            chunk_size_tokens=self.chunk_size_tokens,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
            chunks_created=len(chunks),
            chunks_embedded=len(chunks),
            matches=matches,
            limitations=[],
        )
