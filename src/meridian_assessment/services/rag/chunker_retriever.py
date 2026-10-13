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

    By default `allow_download=False` and `try_local_cache=True` so it loads from the local
    HuggingFace cache (`local_files_only=True`) without ever triggering an unapproved network download.
    Defaults to `device="cpu"` so embedding execution never competes with Ollama on 6 GB VRAM GPUs.
    """

    _MODEL_CACHE: dict[tuple[str, str], Any] = {}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        allow_download: bool = False,
        try_local_cache: bool = True,
        auto_load: Optional[bool] = None,
        device: str = "cpu",
        batch_size: int = 64,
        loaded_model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.allow_download = allow_download
        self.try_local_cache = bool(auto_load) if auto_load is not None else try_local_cache
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._model = loaded_model
        self._load_error: Optional[str] = None


    def _ensure_loaded(self) -> tuple[bool, str]:
        if self._model is not None:
            return True, f"Loaded {self.model_name} ({self.device})."
        cache_key = (self.model_name, self.device)
        if cache_key in self._MODEL_CACHE:
            self._model = self._MODEL_CACHE[cache_key]
            return True, f"Loaded {self.model_name} from process cache ({self.device})."
        if self._load_error is not None:
            return False, self._load_error
        if not self.try_local_cache and not self.allow_download:
            self._load_error = f"Model '{self.model_name}' not pre-loaded and network downloads are disabled."
            return False, self._load_error
        try:
            from sentence_transformers import SentenceTransformer

            model_inst = SentenceTransformer(
                self.model_name,
                local_files_only=not self.allow_download,
                device=self.device,
            )
            self._MODEL_CACHE[cache_key] = model_inst
            self._model = model_inst
            return True, f"Loaded {self.model_name} from local cache ({self.device})."
        except Exception as exc:
            self._load_error = (
                f"Embedding model '{self.model_name}' could not be loaded from local cache: {exc}"
            )
            return False, self._load_error

    def is_available(self) -> tuple[bool, str]:
        return self._ensure_loaded()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ok, msg = self._ensure_loaded()
        if not ok or self._model is None:
            raise RuntimeError(msg)
        if not texts:
            return []
        arr = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in arr]


def build_embedding_provider(
    *,
    enable_embeddings: bool = True,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    allow_download: bool = False,
    device: str = "cpu",
) -> EmbeddingProvider:
    """Build a local-cache SentenceTransformerEmbeddingProvider when enabled, else UnconfiguredEmbeddingProvider."""
    if not enable_embeddings:
        return UnconfiguredEmbeddingProvider(
            model_name=model_name,
            reason=f"Semantic retrieval disabled by configuration for '{model_name}'.",
        )
    provider = SentenceTransformerEmbeddingProvider(
        model_name=model_name,
        allow_download=allow_download,
        try_local_cache=True,
        device=device,
    )
    ok, reason = provider.is_available()
    if ok:
        return provider
    return UnconfiguredEmbeddingProvider(model_name=model_name, reason=reason)


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
        candidate_similarity_threshold: Optional[float] = None,
        top_k_per_concept: Optional[int] = None,
        max_total_candidates: Optional[int] = None,
        min_per_document_quota: Optional[int] = None,
        broad_recall_min_candidates: Optional[int] = None,
        evidence_repository: Optional[Any] = None,
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
        self.candidate_similarity_threshold = (
            float(candidate_similarity_threshold)
            if candidate_similarity_threshold is not None
            else float(cfg.get("candidate_similarity_threshold", 0.24))
        )
        self.top_k_per_concept = (
            int(top_k_per_concept)
            if top_k_per_concept is not None
            else int(cfg.get("top_k_per_concept", 5))
        )
        self.max_total_candidates = (
            int(max_total_candidates)
            if max_total_candidates is not None
            else int(cfg.get("max_total_candidates", 80))
        )
        self.min_per_document_quota = (
            int(min_per_document_quota)
            if min_per_document_quota is not None
            else int(cfg.get("min_per_document_quota", 2))
        )
        self.broad_recall_min_candidates = (
            int(broad_recall_min_candidates)
            if broad_recall_min_candidates is not None
            else int(cfg.get("broad_recall_min_candidates", 10))
        )
        self.model_name: str = str(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
        self.risk_concepts: list[dict[str, Any]] = list(cfg.get("risk_query_concepts", []))
        self.embedding_provider: EmbeddingProvider = embedding_provider or UnconfiguredEmbeddingProvider(
            model_name=self.model_name
        )
        self.evidence_repository = evidence_repository

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
        from meridian_assessment.services.extraction.extractor import OSINTExtractionService

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
                norm_hash = OSINTExtractionService.compute_normalized_passage_hash(chunk_text)
                if c_idx == 0 and not ev.normalized_hash:
                    ev.normalized_hash = norm_hash

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
                        normalized_hash=norm_hash,
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

    def _encode_chunks_with_cache(
        self,
        chunks: list[DocumentChunk],
    ) -> tuple[list[list[float]], int, int]:
        """Encode chunks while deduplicating by normalized_hash and reusing SQLite embedding cache when available.

        Returns `(chunk_vectors, unique_chunks_count, cached_hits_count)`.
        """
        hash_to_text: dict[str, str] = {}
        for chk in chunks:
            h_key = chk.normalized_hash or chk.content_sha256
            if h_key not in hash_to_text:
                hash_to_text[h_key] = chk.text

        unique_hashes = list(hash_to_text.keys())
        cached_map: dict[str, list[float]] = {}
        if self.evidence_repository is not None and hasattr(self.evidence_repository, "get_cached_embeddings"):
            try:
                cached_map = self.evidence_repository.get_cached_embeddings(
                    unique_hashes,
                    self.embedding_provider.model_name,
                )
            except Exception:
                cached_map = {}

        missing_hashes = [h for h in unique_hashes if h not in cached_map]
        if missing_hashes:
            missing_texts = [hash_to_text[h] for h in missing_hashes]
            new_vecs = self.embedding_provider.encode(missing_texts)
            newly_encoded: dict[str, list[float]] = {}
            for h, vec in zip(missing_hashes, new_vecs):
                cached_map[h] = vec
                newly_encoded[h] = vec
            if self.evidence_repository is not None and hasattr(self.evidence_repository, "save_cached_embeddings"):
                try:
                    self.evidence_repository.save_cached_embeddings(
                        newly_encoded,
                        self.embedding_provider.model_name,
                    )
                except Exception:
                    pass

        chunk_vectors = [cached_map[chk.normalized_hash or chk.content_sha256] for chk in chunks]
        return chunk_vectors, len(unique_hashes), len(unique_hashes) - len(missing_hashes)

    def retrieve_relevant_chunks(
        self,
        evidence_records: Sequence[EvidenceRecord],
        *,
        legal_entity: str = "",
    ) -> tuple[list[DocumentChunk], SemanticRetrievalSummary]:
        """Chunk evidence, compute or reuse embeddings, and execute 5-part hybrid candidate selection."""
        from meridian_assessment.services.extraction.extractor import OSINTExtractionService

        ev_by_id: dict[str, EvidenceRecord] = {ev.evidence_id: ev for ev in evidence_records}
        chunks = self.chunk_evidence_records(evidence_records)
        if not chunks:
            avail, msg = self.embedding_provider.is_available()
            return [], SemanticRetrievalSummary(
                status="NO_DOCUMENTS",
                model_name=self.embedding_provider.model_name,
                model_available=avail,
                retrieval_mode="NO_DOCUMENTS",
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
            selected_ids = self._apply_deterministic_fallback_selection(
                chunks, ev_by_id, legal_entity=legal_entity
            )
            screened_cnt = max(0, len(ev_by_id) - len(selected_ids))
            return chunks, SemanticRetrievalSummary(
                status="MODEL_UNAVAILABLE_NOT_CONFIGURED",
                model_name=self.embedding_provider.model_name,
                model_available=False,
                retrieval_mode="DETERMINISTIC_FALLBACK",
                chunk_size_tokens=self.chunk_size_tokens,
                chunk_overlap_tokens=self.chunk_overlap_tokens,
                chunks_created=len(chunks),
                chunks_embedded=0,
                unique_chunks_after_dedup=len({c.normalized_hash or c.content_sha256 for c in chunks}),
                deterministic_candidates_retained=len(selected_ids),
                deterministic_screened_out=screened_cnt,
                candidates_selected_for_llm=len(selected_ids),
                candidates_screened_out_semantic=screened_cnt,
                chunks_screened_out_semantic=screened_cnt,
                selected_candidate_evidence_ids=sorted(selected_ids),
                broad_recall_fallback_triggered=True,
                matches=[],
                limitations=[avail_msg],
            )

        try:
            chunk_vectors, unique_cnt, cache_hits = self._encode_chunks_with_cache(chunks)
            concept_queries = [str(rc.get("query", "")) for rc in self.risk_concepts]
            concept_vectors = self.embedding_provider.encode(concept_queries) if concept_queries else []
        except Exception as exc:
            for chk in chunks:
                chk.embedding_status = EmbeddingStatus.FAILED
            selected_ids = self._apply_deterministic_fallback_selection(
                chunks, ev_by_id, legal_entity=legal_entity
            )
            screened_cnt = max(0, len(ev_by_id) - len(selected_ids))
            return chunks, SemanticRetrievalSummary(
                status="MODEL_UNAVAILABLE_NOT_CONFIGURED",
                model_name=self.embedding_provider.model_name,
                model_available=False,
                retrieval_mode="DETERMINISTIC_FALLBACK",
                chunk_size_tokens=self.chunk_size_tokens,
                chunk_overlap_tokens=self.chunk_overlap_tokens,
                chunks_created=len(chunks),
                chunks_embedded=0,
                unique_chunks_after_dedup=len({c.normalized_hash or c.content_sha256 for c in chunks}),
                deterministic_candidates_retained=len(selected_ids),
                deterministic_screened_out=screened_cnt,
                candidates_selected_for_llm=len(selected_ids),
                candidates_screened_out_semantic=screened_cnt,
                chunks_screened_out_semantic=screened_cnt,
                selected_candidate_evidence_ids=sorted(selected_ids),
                broad_recall_fallback_triggered=True,
                matches=[],
                limitations=[f"Embedding execution failed: {exc}"],
            )

        for chk in chunks:
            chk.embedding_status = EmbeddingStatus.EMBEDDED
            chk.embedding_model_version = self.embedding_provider.model_name

        # Score every chunk against every risk concept query
        chunk_max_sim: dict[str, float] = {chk.chunk_id: 0.0 for chk in chunks}
        chunk_concepts: dict[str, list[str]] = {chk.chunk_id: [] for chk in chunks}
        matches: list[RetrievedChunkMatch] = []
        top_per_concept_ids: set[str] = set()

        for rc, q_vec in zip(self.risk_concepts, concept_vectors):
            c_id = str(rc.get("id", "RC_UNKNOWN"))
            q_txt = str(rc.get("query", ""))
            scored_all: list[tuple[float, DocumentChunk]] = []
            scored_above_thresh: list[tuple[float, DocumentChunk]] = []
            for chk, c_vec in zip(chunks, chunk_vectors):
                sim = cosine_similarity(q_vec, c_vec)
                if sim > chunk_max_sim[chk.chunk_id]:
                    chunk_max_sim[chk.chunk_id] = sim
                if sim >= self.candidate_similarity_threshold and c_id not in chunk_concepts[chk.chunk_id]:
                    chunk_concepts[chk.chunk_id].append(c_id)
                scored_all.append((sim, chk))
                if sim >= self.similarity_threshold:
                    scored_above_thresh.append((sim, chk))

            scored_above_thresh.sort(key=lambda pair: pair[0], reverse=True)
            for sim, chk in scored_above_thresh[: self.top_k_per_concept]:
                matches.append(
                    RetrievedChunkMatch(
                        chunk=chk,
                        concept_id=c_id,
                        concept_query=q_txt,
                        similarity_score=round(sim, 4),
                        retrieval_method="SEMANTIC_EMBEDDING",
                    )
                )

            scored_all.sort(key=lambda pair: pair[0], reverse=True)
            for sim, chk in scored_all[: self.top_k_per_concept]:
                if sim >= self.candidate_similarity_threshold:
                    top_per_concept_ids.add(chk.chunk_id)

        # Attach max similarity and matched concepts to chunks and parent EvidenceRecords
        for chk in chunks:
            msim = round(chunk_max_sim.get(chk.chunk_id, 0.0), 4)
            mconcepts = chunk_concepts.get(chk.chunk_id, [])
            chk.max_similarity_score = msim
            chk.matched_concepts = list(mconcepts)
            if chk.evidence_id and chk.evidence_id in ev_by_id:
                parent_ev = ev_by_id[chk.evidence_id]
                if parent_ev.semantic_score is None or msim > parent_ev.semantic_score:
                    parent_ev.semantic_score = msim
                for cid in mconcepts:
                    if cid not in parent_ev.matched_concepts:
                        parent_ev.matched_concepts.append(cid)

        # Execute 5-part hybrid candidate selection across parent EvidenceRecords
        selected_ev_ids, dup_suppressed, broad_fallback = self._select_hybrid_evidence_candidates(
            chunks=chunks,
            ev_by_id=ev_by_id,
            top_per_concept_chunk_ids=top_per_concept_ids,
            legal_entity=legal_entity,
        )

        mode_label = (
            "HYBRID_BROAD_RECALL_FALLBACK"
            if broad_fallback
            else "HYBRID_SEMANTIC_MINILM"
        )
        screened_sem_cnt = max(0, len(ev_by_id) - len(selected_ev_ids))
        return chunks, SemanticRetrievalSummary(
            status="COMPLETED",
            model_name=self.embedding_provider.model_name,
            model_available=True,
            retrieval_mode=mode_label,
            chunk_size_tokens=self.chunk_size_tokens,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
            chunks_created=len(chunks),
            chunks_embedded=len(chunks),
            chunks_from_cache=cache_hits,
            embedding_cache_hits=cache_hits,
            unique_chunks_after_dedup=unique_cnt,
            duplicates_suppressed=dup_suppressed,
            semantic_candidates_retained=len(selected_ev_ids),
            semantic_screened_out=screened_sem_cnt,
            candidates_selected_for_llm=len(selected_ev_ids),
            candidates_screened_out_semantic=screened_sem_cnt,
            chunks_screened_out_semantic=screened_sem_cnt,
            selected_candidate_evidence_ids=sorted(selected_ev_ids),
            broad_recall_fallback_triggered=broad_fallback,
            matches=matches,
            limitations=[],
        )

    def _apply_deterministic_fallback_selection(
        self,
        chunks: list[DocumentChunk],
        ev_by_id: dict[str, EvidenceRecord],
        *,
        legal_entity: str = "",
    ) -> set[str]:
        """Select top candidate EvidenceRecords deterministically when semantic embeddings are unavailable."""
        from meridian_assessment.services.extraction.extractor import OSINTExtractionService

        seen_norm: set[str] = set()
        scored_evs: list[tuple[int, int, str]] = []
        for idx, (ev_id, ev) in enumerate(ev_by_id.items()):
            norm_h = ev.normalized_hash or OSINTExtractionService.compute_normalized_passage_hash(ev.verbatim_excerpt)
            if norm_h in seen_norm:
                ev.screening_status = "SCREENED_OUT_DETERMINISTIC"
                ev.analysis_state = "SCREENED_OUT_DETERMINISTIC"
                ev.screening_reason = "DUPLICATE_NORMALIZED_HASH"
                continue
            seen_norm.add(norm_h)
            rel = OSINTExtractionService._score_excerpt_relevance(ev.verbatim_excerpt, legal_entity=legal_entity)
            if OSINTExtractionService.contains_protected_governance_disclosure(ev.verbatim_excerpt):
                rel += 25
            scored_evs.append((rel, idx, ev_id))

        scored_evs.sort(key=lambda item: (-item[0], item[1]))
        selected_ids = {item[2] for item in scored_evs[: self.max_total_candidates]}

        for ev_id, ev in ev_by_id.items():
            if ev_id in selected_ids:
                ev.screening_status = "SELECTED_FOR_LLM"
            elif ev.screening_status != "SCREENED_OUT_DETERMINISTIC":
                ev.screening_status = "SCREENED_OUT_SEMANTIC"
                ev.analysis_state = "SCREENED_OUT_SEMANTIC"
                ev.screening_reason = "BELOW_DETERMINISTIC_FALLBACK_CAP"

        for chk in chunks:
            if chk.evidence_id in selected_ids:
                chk.screening_status = "SELECTED_FOR_LLM"
            else:
                chk.screening_status = "SCREENED_OUT_SEMANTIC"
                chk.screening_reason = "BELOW_DETERMINISTIC_FALLBACK_CAP"

        return selected_ids

    def _select_hybrid_evidence_candidates(
        self,
        *,
        chunks: list[DocumentChunk],
        ev_by_id: dict[str, EvidenceRecord],
        top_per_concept_chunk_ids: set[str],
        legal_entity: str = "",
    ) -> tuple[set[str], int, bool]:
        """Execute the 5-part hybrid candidate selection across EvidenceRecords.

        Returns `(selected_evidence_ids, duplicates_suppressed_count, broad_recall_fallback_triggered)`.
        """
        from meridian_assessment.services.extraction.extractor import OSINTExtractionService

        # 1. Deduplicate EvidenceRecords by normalized_hash
        seen_norm: dict[str, str] = {}
        unique_ev_ids: list[str] = []
        dup_suppressed = 0

        for ev_id, ev in ev_by_id.items():
            norm_h = ev.normalized_hash or OSINTExtractionService.compute_normalized_passage_hash(ev.verbatim_excerpt)
            ev.normalized_hash = norm_h
            if norm_h in seen_norm:
                dup_suppressed += 1
                ev.screening_status = "SCREENED_OUT_DETERMINISTIC"
                ev.analysis_state = "SCREENED_OUT_DETERMINISTIC"
                ev.screening_reason = "DUPLICATE_NORMALIZED_HASH"
                primary_ev = ev_by_id[seen_norm[norm_h]]
                if ev.document_id and ev.document_id != primary_ev.document_id:
                    if ev.document_id not in primary_ev.corroborating_document_ids:
                        primary_ev.corroborating_document_ids.append(ev.document_id)
                if ev.canonical_url and ev.canonical_url != primary_ev.canonical_url:
                    if ev.canonical_url not in primary_ev.corroborating_urls:
                        primary_ev.corroborating_urls.append(ev.canonical_url)
                continue
            seen_norm[norm_h] = ev_id
            unique_ev_ids.append(ev_id)

        # Small corpus fast-path: if total unique records <= 20, select all unique records
        if len(unique_ev_ids) <= 20 and len(unique_ev_ids) <= self.max_total_candidates:
            selected_set = set(unique_ev_ids)
            for ev_id in selected_set:
                ev_by_id[ev_id].screening_status = "SELECTED_FOR_LLM"
            for chk in chunks:
                chk.screening_status = (
                    "SELECTED_FOR_LLM" if chk.evidence_id in selected_set else "SCREENED_OUT_SEMANTIC"
                )
            return selected_set, dup_suppressed, False

        # Compute composite ranking features per unique EvidenceRecord
        top_concept_ev_ids = {
            chk.evidence_id for chk in chunks if chk.chunk_id in top_per_concept_chunk_ids and chk.evidence_id
        }
        ev_features: dict[str, dict[str, Any]] = {}
        for idx, ev_id in enumerate(unique_ev_ids):
            ev = ev_by_id[ev_id]
            sim = float(ev.semantic_score or 0.0)
            kw_score = OSINTExtractionService._score_excerpt_relevance(ev.verbatim_excerpt, legal_entity=legal_entity)
            has_protected = OSINTExtractionService.contains_protected_governance_disclosure(ev.verbatim_excerpt)
            in_top_concept = ev_id in top_concept_ev_ids
            # Composite score combines semantic similarity (0..1) + keyword signal bonus
            composite = sim + min(0.35, kw_score * 0.015) + (0.20 if has_protected else 0.0)
            ev_features[ev_id] = {
                "idx": idx,
                "sim": sim,
                "kw_score": kw_score,
                "has_protected": has_protected,
                "in_top_concept": in_top_concept,
                "composite": composite,
                "doc_id": ev.document_id or "UNKNOWN_DOC",
                "source_class": ev.source_class or "S12",
            }

        selected_ids: list[str] = []
        selected_set: set[str] = set()

        def _try_add(eid: str, reason: str) -> bool:
            if len(selected_ids) >= self.max_total_candidates or eid in selected_set:
                return False
            selected_ids.append(eid)
            selected_set.add(eid)
            ev_by_id[eid].screening_status = "SELECTED_FOR_LLM"
            ev_by_id[eid].screening_reason = reason
            return True

        # Tier A: Protected governance/AI disclosures and top-k concept matches above candidate_similarity_threshold
        tier_a = sorted(
            [
                eid
                for eid in unique_ev_ids
                if ev_features[eid]["has_protected"]
                or ev_features[eid]["in_top_concept"]
                or ev_features[eid]["sim"] >= self.candidate_similarity_threshold
            ],
            key=lambda eid: (-ev_features[eid]["composite"], ev_features[eid]["idx"]),
        )

        # Enforce source-class diversity first (at least 1 top candidate from each available source class)
        seen_source_classes: set[str] = set()
        for eid in tier_a:
            sc = ev_features[eid]["source_class"]
            if sc not in seen_source_classes:
                if _try_add(eid, f"SOURCE_CLASS_DIVERSITY_{sc}"):
                    seen_source_classes.add(sc)

        # Enforce per-document diversity quota (`min_per_document_quota`) for documents with viable signal
        doc_counts: dict[str, int] = {}
        for eid in selected_ids:
            d_id = ev_features[eid]["doc_id"]
            doc_counts[d_id] = doc_counts.get(d_id, 0) + 1

        by_composite_all = sorted(
            unique_ev_ids,
            key=lambda eid: (-ev_features[eid]["composite"], ev_features[eid]["idx"]),
        )
        for eid in by_composite_all:
            d_id = ev_features[eid]["doc_id"]
            if doc_counts.get(d_id, 0) < self.min_per_document_quota:
                if (
                    ev_features[eid]["sim"] >= (self.candidate_similarity_threshold * 0.85)
                    or ev_features[eid]["kw_score"] > 0
                    or ev_features[eid]["has_protected"]
                ):
                    if _try_add(eid, "DOCUMENT_DIVERSITY_QUOTA"):
                        doc_counts[d_id] = doc_counts.get(d_id, 0) + 1

        # Fill remaining slots from Tier A (above semantic threshold or protected governance signal)
        for eid in tier_a:
            _try_add(eid, "SEMANTIC_AND_GOVERNANCE_RANKING")

        # Part 5: Broad-recall fallback if fewer than `broad_recall_min_candidates` were selected
        broad_fallback = False
        target_min = min(self.broad_recall_min_candidates, len(unique_ev_ids), self.max_total_candidates)
        if len(selected_ids) < target_min:
            broad_fallback = True
            for eid in by_composite_all:
                if len(selected_ids) >= target_min:
                    break
                _try_add(eid, "BROAD_RECALL_FALLBACK")

        # Mark remaining unique EvidenceRecords as SCREENED_OUT_SEMANTIC (retained in store, skipped by Mistral)
        for eid in unique_ev_ids:
            if eid not in selected_set:
                ev = ev_by_id[eid]
                ev.screening_status = "SCREENED_OUT_SEMANTIC"
                ev.analysis_state = "SCREENED_OUT_SEMANTIC"
                ev.screening_reason = (
                    f"BELOW_SEMANTIC_THRESHOLD_OR_CAP (sim={ev_features[eid]['sim']:.3f})"
                )

        for chk in chunks:
            if chk.evidence_id in selected_set:
                chk.screening_status = "SELECTED_FOR_LLM"
                chk.screening_reason = ev_by_id[chk.evidence_id].screening_reason
            else:
                chk.screening_status = "SCREENED_OUT_SEMANTIC"
                chk.screening_reason = "BELOW_SEMANTIC_THRESHOLD_OR_CAP"

        return selected_set, dup_suppressed, broad_fallback

