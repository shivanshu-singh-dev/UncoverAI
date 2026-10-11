"""Level 4 — Modular Web & PDF Extraction Pipeline with Point-of-Capture Traceability.

Features:
  - Strict public-URL safety validation (`validate_safe_public_url`) rejecting non-HTTP/HTTPS
    schemes, localhost, private/loopback/link-local IPs, and unsafe redirect targets.
  - Readable HTML extraction (`HTMLTextExtractor`) removing `<script>`, `<style>`, `<nav>`,
    `<footer>`, `<header>`, and cookie-banner noise while preserving `<title>` and headings.
  - PDF extraction adapter (`PDFDocumentExtractor`) preserving page numbers and explicitly
    reporting password-protected (`PASSWORD_PROTECTED`), scanned/image-only (`SCANNED_IMAGE_ONLY`),
    malformed, or empty PDFs without fabricating text.
  - Point-of-capture SHA-256 hashing and verbatim `EvidenceRecord` creation with exact
    character offsets (`char_start`, `char_end`) and page references.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import ipaddress
from pathlib import Path
import re
from typing import Any, Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request
import zlib

from meridian_assessment.models.ai_risk import (
    EvidenceRecord,
    EvidenceScope,
    ExtractedDocument,
    ExtractionStatus,
    URLTriageAction,
    URLTriageDecision,
)
from meridian_assessment.models.osint_plan import CandidateURL
from meridian_assessment.services.osint.search_collector import canonicalize_url
from meridian_assessment.services.osint.source_registry import OSINTSourceRegistry


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_sha256(content: bytes | str) -> str:
    """Compute hex SHA-256 digest for bytes or UTF-8 string."""
    data = content.encode("utf-8", errors="replace") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


# ─── URL Safety & SSRF Prevention ─────────────────────────────────────────────


def validate_safe_public_url(url: str) -> tuple[bool, str]:
    """Validate that a URL uses http/https and does not target localhost or private networks."""
    if not url or not isinstance(url, str):
        return False, "Empty or non-string URL."

    cleaned = url.strip()
    try:
        parsed = urllib.parse.urlsplit(cleaned)
    except Exception as exc:
        return False, f"Malformed URL syntax: {exc}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, f"Disallowed URL scheme '{scheme}'; only http and https are permitted."

    if parsed.username or parsed.password:
        return False, "URLs containing embedded credentials are not permitted."

    host = (parsed.hostname or "").lower().strip()
    if not host:
        return False, "URL is missing a hostname."

    blocked_hosts = {"localhost", "localhost.localdomain", "ip6-localhost", "0.0.0.0"}
    if host in blocked_hosts or host.endswith((".local", ".internal", ".localhost", ".lan", ".corp")):
        return False, f"Blocked local/internal hostname '{host}'."

    try:
        ip_obj = ipaddress.ip_address(host.strip("[]"))
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            return False, f"Blocked private/non-routable IP address '{ip_obj}'."
    except ValueError:
        # Host is a domain name, must contain at least one dot
        if "." not in host:
            return False, f"Unqualified single-label hostname '{host}' is not permitted."

    return True, "Valid public HTTP/HTTPS URL."


# ─── HTML Readable Text Extractor ─────────────────────────────────────────────


_IGNORED_HTML_TAGS = frozenset({
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "svg",
    "canvas",
    "iframe",
    "form",
    "button",
    "aside",
})

_COOKIE_NOISE_PATTERNS = re.compile(
    r"(we use cookies|accept all cookies|cookie preferences|manage cookie settings)",
    re.IGNORECASE,
)


class _ReadableHTMLParser(HTMLParser):
    """Standard-library HTML parser extracting title, section headings, and clean body paragraphs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._in_title = False
        self._current_heading_tag: Optional[str] = None
        self.title_parts: list[str] = []
        self.sections: list[tuple[Optional[str], str]] = []
        self._current_heading: Optional[str] = None
        self._heading_buf: list[str] = []
        self._text_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        attr_dict = {k.lower(): (v or "").lower() for k, v in attrs}
        cls_id = f"{attr_dict.get('class', '')} {attr_dict.get('id', '')} {attr_dict.get('role', '')}"

        if t in _IGNORED_HTML_TAGS or any(
            noise in cls_id for noise in ("cookie-banner", "cookie-consent", "gdpr-banner", "navbar", "site-footer")
        ):
            self._ignore_depth += 1
            return

        if self._ignore_depth > 0:
            return

        if t == "title":
            self._in_title = True
        elif t in ("h1", "h2", "h3", "h4"):
            self._flush_section()
            self._current_heading_tag = t
            self._heading_buf = []
        elif t in ("p", "li", "div", "section", "article", "tr", "br"):
            self._text_buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if self._ignore_depth > 0:
            if t in _IGNORED_HTML_TAGS or t in ("div", "section"):
                self._ignore_depth = max(0, self._ignore_depth - 1)
            return

        if t == "title":
            self._in_title = False
        elif t in ("h1", "h2", "h3", "h4") and self._current_heading_tag == t:
            heading_text = re.sub(r"\s+", " ", "".join(self._heading_buf)).strip()
            if heading_text:
                self._current_heading = heading_text
            self._current_heading_tag = None
            self._heading_buf = []
        elif t in ("p", "li", "section", "article", "div"):
            self._text_buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._current_heading_tag is not None:
            self._heading_buf.append(data)
            return
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", data)
        if cleaned.strip() and not _COOKIE_NOISE_PATTERNS.search(cleaned):
            self._text_buf.append(cleaned)

    def _flush_section(self) -> None:
        raw_block = "".join(self._text_buf)
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_block.splitlines()]
        lines = [ln for ln in lines if ln]
        body = "\n".join(lines).strip()
        if body:
            self.sections.append((self._current_heading, body))
        self._text_buf = []

    def finalize(self) -> tuple[Optional[str], str, list[tuple[Optional[str], str]]]:
        self._flush_section()
        title = re.sub(r"\s+", " ", "".join(self.title_parts)).strip() or None
        full_blocks: list[str] = []
        for heading, body in self.sections:
            if heading:
                full_blocks.append(f"[{heading}]\n{body}")
            else:
                full_blocks.append(body)
        return title, "\n\n".join(full_blocks).strip(), self.sections


class HTMLTextExtractor:
    """Extracts readable text, page title, and section structure from HTML content."""

    @staticmethod
    def extract(html_content: str | bytes) -> tuple[Optional[str], str, list[tuple[Optional[str], str]], ExtractionStatus]:
        raw_str = (
            html_content.decode("utf-8", errors="replace")
            if isinstance(html_content, bytes)
            else html_content
        )
        if not raw_str or not raw_str.strip():
            return None, "", [], ExtractionStatus.EMPTY_CONTENT

        parser = _ReadableHTMLParser()
        try:
            parser.feed(raw_str)
            title, cleaned_text, sections = parser.finalize()
        except Exception:
            # Fallback simple tag strip if HTML is severely broken
            title = None
            cleaned_text = re.sub(r"<[^>]+>", " ", raw_str)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
            sections = [(None, cleaned_text)] if cleaned_text else []

        if not cleaned_text:
            return title, "", [], ExtractionStatus.EMPTY_CONTENT
        return title, cleaned_text, sections, ExtractionStatus.EXTRACTED


# ─── PDF Document Extractor ───────────────────────────────────────────────────


class PDFDocumentExtractor:
    """Extracts text and page references from PDF bytes while detecting encrypted or scanned PDFs."""

    @staticmethod
    def extract(pdf_bytes: bytes) -> tuple[Optional[str], str, dict[int, str], ExtractionStatus, list[str]]:
        if not pdf_bytes:
            return None, "", {}, ExtractionStatus.EMPTY_CONTENT, ["PDF payload is empty (0 bytes)."]

        if not pdf_bytes.lstrip().startswith(b"%PDF-"):
            return (
                None,
                "",
                {},
                ExtractionStatus.UNSUPPORTED_FORMAT,
                ["Invalid or malformed PDF header (missing '%PDF-')."],
            )

        # Check for password / encryption dictionary
        if re.search(rb"/Encrypt\b", pdf_bytes):
            return (
                None,
                "",
                {},
                ExtractionStatus.PASSWORD_PROTECTED,
                ["PDF document is encrypted / password-protected; text cannot be extracted without credentials."],
            )

        # Extract Optional Title from Info dictionary if present
        title: Optional[str] = None
        title_match = re.search(rb"/Title\s*\(([^)]{1,200})\)", pdf_bytes)
        if title_match:
            title = title_match.group(1).decode("latin-1", errors="ignore").strip() or None

        # Parse streams and page text operators deterministically
        page_texts: dict[int, str] = {}
        stream_pattern = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
        streams = stream_pattern.findall(pdf_bytes)

        extracted_chunks: list[str] = []
        for idx, raw_stream in enumerate(streams, start=1):
            decoded_bytes = raw_stream
            # Attempt zlib decompression if Flate-encoded
            try:
                decoded_bytes = zlib.decompress(raw_stream)
            except Exception:
                pass

            text_fragments = PDFDocumentExtractor._extract_pdf_text_operators(decoded_bytes)
            if text_fragments:
                page_str = " ".join(text_fragments).strip()
                if page_str:
                    page_num = len(page_texts) + 1
                    page_texts[page_num] = page_str
                    extracted_chunks.append(page_str)

        # Also check uncompressed top-level text blocks if no stream text was found
        if not extracted_chunks:
            top_fragments = PDFDocumentExtractor._extract_pdf_text_operators(pdf_bytes)
            if top_fragments:
                page_str = " ".join(top_fragments).strip()
                if page_str:
                    page_texts[1] = page_str
                    extracted_chunks.append(page_str)

        full_text = "\n\n".join(
            f"[Page {p_no}]\n{p_txt}" for p_no, p_txt in sorted(page_texts.items())
        ).strip()

        if full_text:
            return title, full_text, page_texts, ExtractionStatus.EXTRACTED, []

        # No extractable text: distinguish scanned/image-only PDF from empty PDF
        if re.search(rb"/Subtype\s*/Image\b", pdf_bytes) or re.search(rb"/XObject\b", pdf_bytes):
            return (
                title,
                "",
                {},
                ExtractionStatus.SCANNED_IMAGE_ONLY,
                ["PDF contains embedded images without extractable text layer (scanned PDF; OCR is optional and not enabled)."],
            )

        return (
            title,
            "",
            {},
            ExtractionStatus.EMPTY_CONTENT,
            ["PDF contains no extractable text streams."],
        )

    @staticmethod
    def _extract_pdf_text_operators(data: bytes) -> list[str]:
        """Extract readable strings from PDF `(...) Tj` and `[...] TJ` text operators."""
        text_out: list[str] = []
        # Match BT ... ET text blocks or direct (...) Tj / [...] TJ operators
        for match in re.finditer(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)\s*Tj", data):
            s = PDFDocumentExtractor._unescape_pdf_string(match.group(1))
            if s:
                text_out.append(s)
        for tj_arr in re.finditer(rb"\[(.*?)\]\s*TJ", data, re.DOTALL):
            inner = tj_arr.group(1)
            parts = [
                PDFDocumentExtractor._unescape_pdf_string(m.group(1))
                for m in re.finditer(rb"\(([^()\\]*(?:\\.[^()\\]*)*)\)", inner)
            ]
            joined = "".join(p for p in parts if p).strip()
            if joined:
                text_out.append(joined)
        return text_out

    @staticmethod
    def _unescape_pdf_string(raw: bytes) -> str:
        s = raw.decode("latin-1", errors="ignore")
        s = s.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\").replace(r"\n", " ")
        s = re.sub(r"\s+", " ", s).strip()
        # Keep strings that have printable characters
        if any(ch.isalnum() for ch in s):
            return s
        return ""


# ─── Content Fetch Adapters ───────────────────────────────────────────────────


class FetchResponse:
    """Raw HTTP fetch payload with content type and final URL."""

    def __init__(
        self,
        *,
        url: str,
        status_code: int = 200,
        content_type: str = "text/html",
        content_bytes: bytes = b"",
        error_status: Optional[ExtractionStatus] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.content_type = content_type
        self.content_bytes = content_bytes
        self.error_status = error_status
        self.error_message = error_message


class ContentFetchAdapter(ABC):
    """Abstract interface for fetching public web pages and PDFs."""

    @abstractmethod
    def is_configured(self) -> tuple[bool, str]:
        """Return `(True, info)` if live or stub fetching is active."""

    @abstractmethod
    def fetch(self, url: str, *, timeout_seconds: float = 10.0, max_bytes: int = 2_000_000) -> FetchResponse:
        """Fetch content from a validated public URL."""


class DisabledOfflineFetchAdapter(ContentFetchAdapter):
    """Default offline fetch adapter that never makes network calls."""

    def is_configured(self) -> tuple[bool, str]:
        return False, "Live web/PDF fetching is disabled in offline mode; use StubFetchAdapter or pre-captured artifacts."

    def fetch(self, url: str, *, timeout_seconds: float = 10.0, max_bytes: int = 2_000_000) -> FetchResponse:
        return FetchResponse(
            url=url,
            status_code=0,
            content_type="",
            content_bytes=b"",
            error_status=ExtractionStatus.UNAVAILABLE,
            error_message="Live network extraction is disabled in offline/unconfigured mode.",
        )


class StubFetchAdapter(ContentFetchAdapter):
    """Offline stub fetch adapter for deterministic tests without network calls."""

    def __init__(self, fixtures: Optional[dict[str, FetchResponse | dict[str, Any]]] = None) -> None:
        self.fixtures = fixtures or {}
        self.fetched_urls: list[str] = []

    def is_configured(self) -> tuple[bool, str]:
        return True, "Configured offline StubFetchAdapter."

    def fetch(self, url: str, *, timeout_seconds: float = 10.0, max_bytes: int = 2_000_000) -> FetchResponse:
        ok, reason = validate_safe_public_url(url)
        if not ok:
            return FetchResponse(
                url=url,
                status_code=0,
                error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                error_message=reason,
            )

        self.fetched_urls.append(url)
        canon = canonicalize_url(url) or url
        entry = self.fixtures.get(url) or self.fixtures.get(canon)
        if entry is None:
            return FetchResponse(
                url=url,
                status_code=404,
                error_status=ExtractionStatus.FETCH_FAILED,
                error_message=f"No offline fixture registered for URL '{url}'.",
            )
        if isinstance(entry, FetchResponse):
            if len(entry.content_bytes) > max_bytes:
                return FetchResponse(
                    url=entry.url,
                    status_code=entry.status_code,
                    content_type=entry.content_type,
                    content_bytes=entry.content_bytes[:max_bytes],
                )
            return entry

        raw_body = entry.get("body", b"")
        body_bytes = raw_body.encode("utf-8") if isinstance(raw_body, str) else bytes(raw_body)
        return FetchResponse(
            url=entry.get("final_url", url),
            status_code=int(entry.get("status_code", 200)),
            content_type=str(entry.get("content_type", "text/html")),
            content_bytes=body_bytes[:max_bytes],
            error_status=entry.get("error_status"),
            error_message=entry.get("error_message"),
        )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validates every HTTP redirect destination against SSRF/private-network rules."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        ok, reason = validate_safe_public_url(newurl)
        if not ok:
            raise urllib.error.URLError(f"Blocked unsafe redirect target '{newurl}': {reason}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibPublicFetchAdapter(ContentFetchAdapter):
    """Bounded HTTP/HTTPS fetcher enforcing public URL validation and response-size limits."""

    def is_configured(self) -> tuple[bool, str]:
        return True, "UrllibPublicFetchAdapter ready."

    def fetch(self, url: str, *, timeout_seconds: float = 10.0, max_bytes: int = 2_000_000) -> FetchResponse:
        ok, reason = validate_safe_public_url(url)
        if not ok:
            return FetchResponse(
                url=url,
                status_code=0,
                error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                error_message=reason,
            )

        opener = urllib.request.build_opener(_SafeRedirectHandler())
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MeridianRiskAssessmentPOC/1.0"},
            method="GET",
        )
        try:
            with opener.open(req, timeout=timeout_seconds) as resp:
                final_url = resp.geturl()
                ok_final, reason_final = validate_safe_public_url(final_url)
                if not ok_final:
                    return FetchResponse(
                        url=final_url,
                        status_code=0,
                        error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                        error_message=reason_final,
                    )
                ctype = (resp.headers.get("Content-Type") or "text/html").split(";")[0].strip().lower()
                data = resp.read(max_bytes)
                return FetchResponse(
                    url=final_url,
                    status_code=getattr(resp, "status", 200),
                    content_type=ctype,
                    content_bytes=data,
                )
        except TimeoutError as exc:
            return FetchResponse(
                url=url,
                status_code=0,
                error_status=ExtractionStatus.TIMEOUT,
                error_message=f"Fetch timed out after {timeout_seconds}s: {exc}",
            )
        except urllib.error.HTTPError as exc:
            return FetchResponse(
                url=url,
                status_code=exc.code,
                error_status=ExtractionStatus.FETCH_FAILED,
                error_message=f"HTTP {exc.code}: {exc.reason}",
            )
        except urllib.error.URLError as exc:
            if "Blocked unsafe redirect" in str(exc):
                return FetchResponse(
                    url=url,
                    status_code=0,
                    error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                    error_message=str(exc),
                )
            if "timed out" in str(exc).lower():
                return FetchResponse(
                    url=url,
                    status_code=0,
                    error_status=ExtractionStatus.TIMEOUT,
                    error_message=str(exc),
                )
            return FetchResponse(
                url=url,
                status_code=0,
                error_status=ExtractionStatus.FETCH_FAILED,
                error_message=f"Network error: {exc.reason}",
            )


# ─── Extraction & Evidence Record Construction Service ────────────────────────


class OSINTExtractionService:
    """Orchestrates web/PDF extraction and constructs traceable EvidenceRecord objects."""

    def __init__(
        self,
        fetch_adapter: Optional[ContentFetchAdapter] = None,
        source_registry: Optional[OSINTSourceRegistry] = None,
        artifact_dir: Optional[Path] = None,
        timeout_seconds: float = 10.0,
        max_bytes: int = 2_000_000,
    ) -> None:
        self.fetch_adapter: ContentFetchAdapter = fetch_adapter or DisabledOfflineFetchAdapter()
        self.source_registry: OSINTSourceRegistry = source_registry or OSINTSourceRegistry()
        self.artifact_dir = artifact_dir
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes

    def extract_from_url(
        self,
        *,
        vendor_id: str,
        url: str,
        source_class: str = "S2",
        query_ids: Optional[list[str]] = None,
        triage_action: Optional[str] = None,
        document_index: int = 1,
        is_synthetic_fixture: bool = False,
    ) -> ExtractedDocument:
        """Validate, fetch, and extract a single webpage or PDF URL into an ExtractedDocument."""
        now_iso = _utc_now_iso()
        doc_id = f"DOC_{vendor_id}_{document_index:03d}"

        ok, safety_msg = validate_safe_public_url(url)
        canon_url = canonicalize_url(url) or url
        if not ok:
            return ExtractedDocument(
                document_id=doc_id,
                vendor_id=vendor_id,
                url=url,
                canonical_url=canon_url,
                source_class=source_class,
                content_type="unknown",
                retrieved_at=now_iso,
                extraction_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                extraction_limitations=[safety_msg],
                query_ids=list(query_ids or []),
                triage_action=triage_action,
                is_synthetic_fixture=is_synthetic_fixture,
            )

        resp = self.fetch_adapter.fetch(url, timeout_seconds=self.timeout_seconds, max_bytes=self.max_bytes)
        if resp.error_status is not None:
            return ExtractedDocument(
                document_id=doc_id,
                vendor_id=vendor_id,
                url=url,
                canonical_url=canonicalize_url(resp.url) or canon_url,
                source_class=source_class,
                content_type=resp.content_type or "unknown",
                retrieved_at=now_iso,
                extraction_status=resp.error_status,
                extraction_limitations=[resp.error_message or f"Extraction failed with status {resp.error_status.value}."],
                query_ids=list(query_ids or []),
                triage_action=triage_action,
                is_synthetic_fixture=is_synthetic_fixture,
            )

        content_bytes = resp.content_bytes
        sha256_hex = compute_sha256(content_bytes)
        ctype = (resp.content_type or "").lower()
        is_pdf = ctype == "application/pdf" or url.lower().split("?")[0].endswith(".pdf") or content_bytes.lstrip().startswith(b"%PDF-")

        artifact_path_str: Optional[str] = None
        if self.artifact_dir is not None and content_bytes:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            ext = ".pdf" if is_pdf else ".html"
            art_file = self.artifact_dir / f"{doc_id}_{sha256_hex[:12]}{ext}"
            art_file.write_bytes(content_bytes)
            artifact_path_str = str(art_file)

        if is_pdf:
            title, cleaned_text, page_texts, status, limits = PDFDocumentExtractor.extract(content_bytes)
            return ExtractedDocument(
                document_id=doc_id,
                vendor_id=vendor_id,
                url=url,
                canonical_url=canonicalize_url(resp.url) or canon_url,
                source_class=source_class,
                content_type="application/pdf",
                title=title,
                raw_text="",
                cleaned_text=cleaned_text,
                page_texts=page_texts,
                content_sha256=sha256_hex,
                byte_size=len(content_bytes),
                retrieved_at=now_iso,
                extraction_status=status,
                extraction_limitations=limits,
                artifact_path=artifact_path_str,
                query_ids=list(query_ids or []),
                triage_action=triage_action,
                is_synthetic_fixture=is_synthetic_fixture,
            )

        if ctype and not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml+xml", "text/markdown")):
            return ExtractedDocument(
                document_id=doc_id,
                vendor_id=vendor_id,
                url=url,
                canonical_url=canonicalize_url(resp.url) or canon_url,
                source_class=source_class,
                content_type=ctype,
                content_sha256=sha256_hex,
                byte_size=len(content_bytes),
                retrieved_at=now_iso,
                extraction_status=ExtractionStatus.UNSUPPORTED_FORMAT,
                extraction_limitations=[f"Unsupported content type '{ctype}'."],
                artifact_path=artifact_path_str,
                query_ids=list(query_ids or []),
                triage_action=triage_action,
                is_synthetic_fixture=is_synthetic_fixture,
            )

        raw_html = content_bytes.decode("utf-8", errors="replace")
        title, cleaned_text, _, status = HTMLTextExtractor.extract(raw_html)
        limits = [] if status == ExtractionStatus.EXTRACTED else ["HTML document contained no extractable body text."]
        return ExtractedDocument(
            document_id=doc_id,
            vendor_id=vendor_id,
            url=url,
            canonical_url=canonicalize_url(resp.url) or canon_url,
            source_class=source_class,
            content_type=ctype or "text/html",
            title=title,
            raw_text=raw_html,
            cleaned_text=cleaned_text,
            page_texts={},
            content_sha256=sha256_hex,
            byte_size=len(content_bytes),
            retrieved_at=now_iso,
            extraction_status=status,
            extraction_limitations=limits,
            artifact_path=artifact_path_str,
            query_ids=list(query_ids or []),
            triage_action=triage_action,
            is_synthetic_fixture=is_synthetic_fixture,
        )

    def build_evidence_records_from_document(
        self,
        doc: ExtractedDocument,
        *,
        legal_entity: str,
        domain: str,
        starting_seq: int = 1,
    ) -> list[EvidenceRecord]:
        """Construct traceable EvidenceRecord items from an ExtractedDocument.

        Every `verbatim_excerpt` is an exact substring of `doc.cleaned_text` with recorded
        `char_start` and `char_end` offsets. Never fabricates text for failed/empty extractions.
        """
        s_def = self.source_registry.get_source(doc.source_class)
        grade = s_def.default_evidence_grade if s_def else "C"
        rel_rating = s_def.reliability_rating if s_def else 2

        if doc.extraction_status != ExtractionStatus.EXTRACTED or not doc.cleaned_text.strip():
            return []

        records: list[EvidenceRecord] = []
        text = doc.cleaned_text
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

        search_cursor = 0
        seq = starting_seq
        current_heading: Optional[str] = None
        current_page: Optional[int] = None

        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue

            # Detect [Page N] or [Heading] prefix lines
            while lines and lines[0].startswith("[") and lines[0].endswith("]"):
                tag_inner = lines[0][1:-1].strip()
                if tag_inner.lower().startswith("page "):
                    try:
                        current_page = int(tag_inner.split()[1])
                    except ValueError:
                        pass
                else:
                    current_heading = tag_inner
                lines.pop(0)

            excerpt = "\n".join(lines).strip()
            if not excerpt:
                continue

            idx = text.find(excerpt, search_cursor)
            if idx == -1:
                idx = text.find(excerpt)
            char_start = idx if idx != -1 else 0
            char_end = char_start + len(excerpt)
            if idx != -1:
                search_cursor = char_end

            ev_id = f"EV_{doc.vendor_id}_{seq:03d}"
            seq += 1

            records.append(
                EvidenceRecord(
                    evidence_id=ev_id,
                    document_id=doc.document_id,
                    vendor_id=doc.vendor_id,
                    legal_entity=legal_entity,
                    domain=domain,
                    source_class=doc.source_class,
                    canonical_url=doc.canonical_url,
                    original_url=doc.url,
                    retrieved_at=doc.retrieved_at,
                    page_title=doc.title,
                    section_heading=current_heading,
                    verbatim_excerpt=excerpt,
                    page_number=current_page,
                    char_start=char_start,
                    char_end=char_end,
                    content_sha256=doc.content_sha256 or compute_sha256(text),
                    capture_artifact_ref=doc.artifact_path,
                    source_reliability_grade=grade,
                    source_reliability_rating=rel_rating,
                    query_ids=list(doc.query_ids),
                    triage_decision=doc.triage_action,
                    extraction_status=doc.extraction_status,
                    extraction_limitations=list(doc.extraction_limitations),
                    is_reviewed_and_accepted=False,
                    is_synthetic_test_fixture=doc.is_synthetic_fixture,
                )
            )

        return records
