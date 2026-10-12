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
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Optional, Sequence
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


# ─── Source-Integrity & Identity Validation ───────────────────────────────────


def _normalize_host_identity(url_or_host: Optional[str]) -> str:
    """Extract normalized lowercase hostname without leading 'www.'."""
    if not url_or_host or not isinstance(url_or_host, str):
        return ""
    cleaned = url_or_host.strip()
    if not cleaned or cleaned.upper() == "UNKNOWN":
        return ""
    if "://" in cleaned:
        try:
            host = (urllib.parse.urlsplit(cleaned).hostname or "").lower().strip()
        except Exception:
            host = ""
    else:
        host = cleaned.split("/")[0].split(":")[0].lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _hosts_belong_to_same_identity(host_a: str, host_b: str) -> bool:
    """Return True if two normalized hostnames belong to the same domain family."""
    if not host_a or not host_b:
        return True
    if host_a == host_b:
        return True
    if host_a.endswith("." + host_b) or host_b.endswith("." + host_a):
        return True
    parts_a = host_a.split(".")
    parts_b = host_b.split(".")
    if len(parts_a) >= 2 and len(parts_b) >= 2 and parts_a[-2:] == parts_b[-2:]:
        # Avoid matching shared multi-part public suffixes like co.uk
        if parts_a[-2] not in ("co", "com", "org", "net", "gov", "edu"):
            return True
    return False


def validate_source_integrity(
    *,
    requested_url: str,
    final_response_url: Optional[str] = None,
    content_type: str = "text/html",
    declared_canonical_url: Optional[str] = None,
    declared_og_url: Optional[str] = None,
    page_title: Optional[str] = None,
) -> tuple[bool, str, list[str]]:
    """Validate that a fetched webpage/document matches the requested source identity.

    Returns `(is_valid, integrity_status, warnings_or_reasons)`.
      - `is_valid == False` (`integrity_status == "SOURCE_IDENTITY_MISMATCH"`) when the final
        response URL, declared canonical/OG URL, or page title indicates an unrelated domain
        (such as requesting `https://httpbin.org/json/` and receiving `https://saas.group/` HTML).
      - Same-host canonical path differences produce `is_valid == True` with
        `integrity_status == "WARNING_SAME_HOST_PATH_DIFF"` so valid vendor documentation is not discarded.
    """
    warnings: list[str] = []
    req_host = _normalize_host_identity(requested_url)
    if not req_host:
        return True, "VERIFIED", warnings

    try:
        req_parsed = urllib.parse.urlsplit(requested_url.strip())
        req_path = (req_parsed.path or "/").rstrip("/") or "/"
    except Exception:
        req_path = "/"

    # 1. Check final response URL host if known
    if final_response_url and final_response_url.upper() != "UNKNOWN":
        final_host = _normalize_host_identity(final_response_url)
        if final_host and not _hosts_belong_to_same_identity(req_host, final_host):
            warnings.append(
                f"Final response URL '{final_response_url}' resolved to unrelated host '{final_host}' "
                f"(requested '{requested_url}', expected host '{req_host}')."
            )
            return False, "SOURCE_IDENTITY_MISMATCH", warnings

    # 2. Check declared <link rel="canonical"> and <meta property="og:url">
    for meta_label, meta_url in (
        ("canonical", declared_canonical_url),
        ("og:url", declared_og_url),
    ):
        if not meta_url or not isinstance(meta_url, str):
            continue
        cleaned_meta = meta_url.strip()
        if not cleaned_meta.lower().startswith(("http://", "https://")):
            continue
        meta_host = _normalize_host_identity(cleaned_meta)
        if not meta_host:
            continue
        if not _hosts_belong_to_same_identity(req_host, meta_host):
            warnings.append(
                f"Declared HTML {meta_label} '{cleaned_meta}' points to unrelated host '{meta_host}' "
                f"instead of requested host '{req_host}' ('{requested_url}')."
            )
            return False, "SOURCE_IDENTITY_MISMATCH", warnings

        try:
            meta_path = (urllib.parse.urlsplit(cleaned_meta).path or "/").rstrip("/") or "/"
        except Exception:
            meta_path = "/"
        if meta_path != req_path:
            msg = (
                f"Declared same-host {meta_label} path '{meta_path}' differs from requested path '{req_path}'; "
                "preserving actual requested/response URL."
            )
            if msg not in warnings:
                warnings.append(msg)

    # 3. Check if page title explicitly advertises a completely different domain (e.g. "saas.group - ...")
    if page_title:
        title_domain_match = re.match(r"^\s*([a-z0-9-]+(?:\.[a-z]{2,10})+)\b", page_title.strip(), re.IGNORECASE)
        if title_domain_match:
            title_host = _normalize_host_identity(title_domain_match.group(1))
            if title_host and not _hosts_belong_to_same_identity(req_host, title_host):
                warnings.append(
                    f"Page title '{page_title}' identifies unrelated domain '{title_host}' "
                    f"for requested URL '{requested_url}' (expected '{req_host}')."
                )
                return False, "SOURCE_IDENTITY_MISMATCH", warnings

    # 4. Check if a JSON endpoint returned HTML with an unrelated title
    if (req_path.endswith("/json") or req_path.endswith(".json")) and "html" in (content_type or "").lower():
        if page_title and req_host.split(".")[0] not in page_title.lower():
            warnings.append(
                f"Requested JSON endpoint '{requested_url}' returned HTML document titled '{page_title}'."
            )
            return False, "SOURCE_IDENTITY_MISMATCH", warnings

    if warnings:
        return True, "WARNING_SAME_HOST_PATH_DIFF", warnings
    return True, "VERIFIED", warnings


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
    """Standard-library HTML parser extracting title, metadata URLs, section headings, and clean body paragraphs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._in_title = False
        self._current_heading_tag: Optional[str] = None
        self.title_parts: list[str] = []
        self.declared_canonical_url: Optional[str] = None
        self.declared_og_url: Optional[str] = None
        self.sections: list[tuple[Optional[str], str]] = []
        self._current_heading: Optional[str] = None
        self._heading_buf: list[str] = []
        self._text_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        raw_attr_dict = {k.lower(): (v or "") for k, v in attrs}
        attr_dict = {k: v.lower() for k, v in raw_attr_dict.items()}

        if t == "link":
            rel_tokens = attr_dict.get("rel", "").split()
            href_val = raw_attr_dict.get("href", "").strip()
            if "canonical" in rel_tokens and href_val and not self.declared_canonical_url:
                self.declared_canonical_url = href_val
            return

        if t == "meta":
            prop_name = (attr_dict.get("property") or attr_dict.get("name") or "").strip()
            content_val = raw_attr_dict.get("content", "").strip()
            if prop_name == "og:url" and content_val and not self.declared_og_url:
                self.declared_og_url = content_val
            return

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
    """Extracts readable text, page title, metadata URLs, and section structure from HTML content."""

    @staticmethod
    def extract_with_metadata(
        html_content: str | bytes,
    ) -> tuple[
        Optional[str],
        str,
        list[tuple[Optional[str], str]],
        ExtractionStatus,
        Optional[str],
        Optional[str],
    ]:
        raw_str = (
            html_content.decode("utf-8", errors="replace")
            if isinstance(html_content, bytes)
            else html_content
        )
        if not raw_str or not raw_str.strip():
            return None, "", [], ExtractionStatus.EMPTY_CONTENT, None, None

        parser = _ReadableHTMLParser()
        try:
            parser.feed(raw_str)
            title, cleaned_text, sections = parser.finalize()
            decl_canon = parser.declared_canonical_url
            decl_og = parser.declared_og_url
        except Exception:
            title = None
            decl_canon = None
            decl_og = None
            cleaned_text = re.sub(r"<[^>]+>", " ", raw_str)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
            sections = [(None, cleaned_text)] if cleaned_text else []

        if not cleaned_text:
            return title, "", [], ExtractionStatus.EMPTY_CONTENT, decl_canon, decl_og
        return title, cleaned_text, sections, ExtractionStatus.EXTRACTED, decl_canon, decl_og

    @staticmethod
    def extract(html_content: str | bytes) -> tuple[Optional[str], str, list[tuple[Optional[str], str]], ExtractionStatus]:
        title, cleaned_text, sections, status, _, _ = HTMLTextExtractor.extract_with_metadata(html_content)
        return title, cleaned_text, sections, status


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
    """Raw HTTP fetch payload with content type, final URL, redirect chain, and retrieval method."""

    def __init__(
        self,
        *,
        url: str,
        status_code: int = 200,
        content_type: str = "text/html",
        content_bytes: bytes = b"",
        error_status: Optional[ExtractionStatus] = None,
        error_message: Optional[str] = None,
        requested_url: Optional[str] = None,
        final_url: Optional[str] = None,
        redirect_chain: Optional[list[str]] = None,
        retrieval_method: str = "DIRECT_HTTP",
    ) -> None:
        self.url = url
        self.requested_url = requested_url or url
        self.final_url = final_url if final_url is not None else url
        self.redirect_chain = list(redirect_chain or [])
        self.retrieval_method = retrieval_method
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
            requested_url=url,
            final_url="UNKNOWN",
            retrieval_method="OFFLINE_DISABLED",
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
                requested_url=url,
                final_url="UNKNOWN",
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
                requested_url=url,
                final_url="UNKNOWN",
                status_code=404,
                error_status=ExtractionStatus.FETCH_FAILED,
                error_message=f"No offline fixture registered for URL '{url}'.",
            )
        if isinstance(entry, FetchResponse):
            if len(entry.content_bytes) > max_bytes:
                return FetchResponse(
                    url=entry.url,
                    requested_url=entry.requested_url or url,
                    final_url=entry.final_url,
                    redirect_chain=entry.redirect_chain,
                    retrieval_method=entry.retrieval_method,
                    status_code=entry.status_code,
                    content_type=entry.content_type,
                    content_bytes=entry.content_bytes[:max_bytes],
                )
            return entry

        raw_body = entry.get("body", b"")
        body_bytes = raw_body.encode("utf-8") if isinstance(raw_body, str) else bytes(raw_body)
        final_u = entry.get("final_url", url)
        return FetchResponse(
            url=final_u if final_u and final_u != "UNKNOWN" else url,
            requested_url=url,
            final_url=final_u,
            redirect_chain=list(entry.get("redirect_chain", [])),
            retrieval_method=str(entry.get("retrieval_method", "DIRECT_HTTP")),
            status_code=int(entry.get("status_code", 200)),
            content_type=str(entry.get("content_type", "text/html")),
            content_bytes=body_bytes[:max_bytes],
            error_status=entry.get("error_status"),
            error_message=entry.get("error_message"),
        )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validates every HTTP redirect destination against SSRF/private-network rules and records hops."""

    def __init__(self) -> None:
        super().__init__()
        self.redirect_chain: list[str] = []

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        ok, reason = validate_safe_public_url(newurl)
        if not ok:
            raise urllib.error.URLError(f"Blocked unsafe redirect target '{newurl}': {reason}")
        self.redirect_chain.append(newurl)
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
                requested_url=url,
                final_url="UNKNOWN",
                retrieval_method="DIRECT_HTTP",
                status_code=0,
                error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                error_message=reason,
            )

        redirect_handler = _SafeRedirectHandler()
        opener = urllib.request.build_opener(redirect_handler)
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
                        requested_url=url,
                        final_url=final_url,
                        redirect_chain=redirect_handler.redirect_chain,
                        retrieval_method="DIRECT_HTTP",
                        status_code=0,
                        error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                        error_message=reason_final,
                    )
                ctype = (resp.headers.get("Content-Type") or "text/html").split(";")[0].strip().lower()
                data = resp.read(max_bytes)
                method_label = "DIRECT_PDF" if ctype == "application/pdf" else "DIRECT_HTTP"
                return FetchResponse(
                    url=final_url,
                    requested_url=url,
                    final_url=final_url,
                    redirect_chain=redirect_handler.redirect_chain,
                    retrieval_method=method_label,
                    status_code=getattr(resp, "status", 200),
                    content_type=ctype,
                    content_bytes=data,
                )
        except TimeoutError as exc:
            return FetchResponse(
                url=url,
                requested_url=url,
                final_url="UNKNOWN",
                redirect_chain=redirect_handler.redirect_chain,
                retrieval_method="DIRECT_HTTP",
                status_code=0,
                error_status=ExtractionStatus.TIMEOUT,
                error_message=f"Fetch timed out after {timeout_seconds}s: {exc}",
            )
        except urllib.error.HTTPError as exc:
            return FetchResponse(
                url=url,
                requested_url=url,
                final_url="UNKNOWN",
                redirect_chain=redirect_handler.redirect_chain,
                retrieval_method="DIRECT_HTTP",
                status_code=exc.code,
                error_status=ExtractionStatus.FETCH_FAILED,
                error_message=f"HTTP {exc.code}: {exc.reason}",
            )
        except urllib.error.URLError as exc:
            if "Blocked unsafe redirect" in str(exc):
                return FetchResponse(
                    url=url,
                    requested_url=url,
                    final_url="UNKNOWN",
                    redirect_chain=redirect_handler.redirect_chain,
                    retrieval_method="DIRECT_HTTP",
                    status_code=0,
                    error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                    error_message=str(exc),
                )
            if "timed out" in str(exc).lower():
                return FetchResponse(
                    url=url,
                    requested_url=url,
                    final_url="UNKNOWN",
                    redirect_chain=redirect_handler.redirect_chain,
                    retrieval_method="DIRECT_HTTP",
                    status_code=0,
                    error_status=ExtractionStatus.TIMEOUT,
                    error_message=str(exc),
                )
            return FetchResponse(
                url=url,
                requested_url=url,
                final_url="UNKNOWN",
                redirect_chain=redirect_handler.redirect_chain,
                retrieval_method="DIRECT_HTTP",
                status_code=0,
                error_status=ExtractionStatus.FETCH_FAILED,
                error_message=f"Network error: {exc.reason}",
            )


class ScraperAPIFetchAdapter(ContentFetchAdapter):
    """ScraperAPI webpage retrieval adapter (`https://api.scraperapi.com`).

    Enforces:
      - Public URL validation before forwarding to ScraperAPI.
      - Concurrency cap (`max_concurrent_requests=5` by default).
      - Bounded retries and timeouts.
      - `render=true` disabled by default unless explicitly requested.
      - Complete redaction of `api_key` from any error message or diagnostic string.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.scraperapi.com",
        *,
        render_js: bool = False,
        max_concurrent_requests: int = 5,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.5,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.endpoint = endpoint
        self.render_js = render_js
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._sleep_fn = sleep_fn
        self._semaphore = threading.BoundedSemaphore(self.max_concurrent_requests)

    def _redact(self, message: str) -> str:
        if not self.api_key or not message:
            return message
        return message.replace(self.api_key, "[REDACTED]")

    def is_configured(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ScraperAPI fallback requires MERIDIAN_SCRAPERAPI_KEY in .env."
        return True, "ScraperAPI retrieval fallback configured."

    def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float = 20.0,
        max_bytes: int = 2_000_000,
        render_js: Optional[bool] = None,
    ) -> FetchResponse:
        ok_url, url_reason = validate_safe_public_url(url)
        if not ok_url:
            return FetchResponse(
                url=url,
                requested_url=url,
                final_url="UNKNOWN",
                retrieval_method="SCRAPERAPI",
                status_code=0,
                error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                error_message=url_reason,
            )

        ok_cfg, cfg_reason = self.is_configured()
        if not ok_cfg:
            return FetchResponse(
                url=url,
                requested_url=url,
                final_url="UNKNOWN",
                retrieval_method="SCRAPERAPI",
                status_code=0,
                error_status=ExtractionStatus.UNAVAILABLE,
                error_message=cfg_reason,
            )

        params: dict[str, str] = {
            "api_key": self.api_key,
            "url": url,
        }
        use_render = self.render_js if render_js is None else bool(render_js)
        if use_render:
            params["render"] = "true"

        query_str = urllib.parse.urlencode(params)
        request_url = f"{self.endpoint}?{query_str}"
        max_attempts = 1 + self.max_retries
        last_resp: Optional[FetchResponse] = None

        with self._semaphore:
            for attempt in range(1, max_attempts + 1):
                req = urllib.request.Request(
                    request_url,
                    headers={"Accept": "text/html,application/xhtml+xml,application/pdf,*/*"},
                    method="GET",
                )
                try:
                    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                        ctype = (resp.headers.get("Content-Type") or "text/html").split(";")[0].strip().lower()
                        resolved_hdr = (
                            resp.headers.get("Sa-Final-Url")
                            or resp.headers.get("X-Resolved-Url")
                            or "UNKNOWN"
                        ).strip()
                        if resolved_hdr != "UNKNOWN":
                            ok_res, reason_res = validate_safe_public_url(resolved_hdr)
                            if not ok_res:
                                return FetchResponse(
                                    url=url,
                                    requested_url=url,
                                    final_url=resolved_hdr,
                                    retrieval_method="SCRAPERAPI",
                                    status_code=0,
                                    error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                                    error_message=reason_res,
                                )
                        data = resp.read(max_bytes)
                        method_label = "SCRAPERAPI_PDF" if ctype == "application/pdf" else "SCRAPERAPI"
                        return FetchResponse(
                            url=resolved_hdr if resolved_hdr != "UNKNOWN" else url,
                            requested_url=url,
                            final_url=resolved_hdr,
                            retrieval_method=method_label,
                            status_code=getattr(resp, "status", 200),
                            content_type=ctype,
                            content_bytes=data,
                        )
                except TimeoutError as exc:
                    last_resp = FetchResponse(
                        url=url,
                        requested_url=url,
                        final_url="UNKNOWN",
                        retrieval_method="SCRAPERAPI",
                        status_code=0,
                        error_status=ExtractionStatus.TIMEOUT,
                        error_message=self._redact(f"ScraperAPI request timed out after {timeout_seconds}s: {exc}"),
                    )
                    if attempt < max_attempts and self.retry_backoff_seconds > 0:
                        self._sleep_fn(self.retry_backoff_seconds * attempt)
                        continue
                    return last_resp
                except urllib.error.HTTPError as exc:
                    code = exc.code
                    reason_str = self._redact(str(exc.reason))
                    if code in (401, 403):
                        return FetchResponse(
                            url=url,
                            requested_url=url,
                            final_url="UNKNOWN",
                            retrieval_method="SCRAPERAPI",
                            status_code=code,
                            error_status=ExtractionStatus.UNAVAILABLE,
                            error_message=f"ScraperAPI authentication/quota failed (HTTP {code}: {reason_str}).",
                        )
                    last_resp = FetchResponse(
                        url=url,
                        requested_url=url,
                        final_url="UNKNOWN",
                        retrieval_method="SCRAPERAPI",
                        status_code=code,
                        error_status=ExtractionStatus.FETCH_FAILED,
                        error_message=f"ScraperAPI HTTP {code}: {reason_str}",
                    )
                    if (code == 429 or 500 <= code < 600) and attempt < max_attempts:
                        if self.retry_backoff_seconds > 0:
                            self._sleep_fn(self.retry_backoff_seconds * attempt)
                        continue
                    return last_resp
                except urllib.error.URLError as exc:
                    err_str = self._redact(str(exc.reason))
                    is_timeout = "timed out" in err_str.lower()
                    last_resp = FetchResponse(
                        url=url,
                        requested_url=url,
                        final_url="UNKNOWN",
                        retrieval_method="SCRAPERAPI",
                        status_code=0,
                        error_status=ExtractionStatus.TIMEOUT if is_timeout else ExtractionStatus.FETCH_FAILED,
                        error_message=f"ScraperAPI network error: {err_str}",
                    )
                    if attempt < max_attempts and self.retry_backoff_seconds > 0:
                        self._sleep_fn(self.retry_backoff_seconds * attempt)
                        continue
                    return last_resp

        return last_resp or FetchResponse(
            url=url,
            requested_url=url,
            final_url="UNKNOWN",
            retrieval_method="SCRAPERAPI",
            status_code=0,
            error_status=ExtractionStatus.FETCH_FAILED,
            error_message="ScraperAPI request failed.",
        )


_BOT_BLOCK_MARKERS = (
    "just a moment...",
    "enable javascript and cookies to continue",
    "attention required! | cloudflare",
    "cf-browser-verification",
    "access denied - GoDaddy",
    "please verify you are a human",
)


def _is_usable_fetch_response(resp: FetchResponse) -> bool:
    """Return True if a direct HTTP FetchResponse succeeded and contains usable non-challenge content."""
    if resp.error_status is not None:
        return False
    if not (200 <= resp.status_code < 300):
        return False
    if not resp.content_bytes or not resp.content_bytes.strip():
        return False

    ctype = (resp.content_type or "").lower()
    if ctype == "application/pdf" or resp.content_bytes.lstrip().startswith(b"%PDF-"):
        return True

    if "html" in ctype or not ctype:
        sample_lower = resp.content_bytes[:8192].decode("utf-8", errors="ignore").lower()
        if any(marker.lower() in sample_lower for marker in _BOT_BLOCK_MARKERS):
            return False
        _, cleaned_text, _, status = HTMLTextExtractor.extract(resp.content_bytes)
        if status != ExtractionStatus.EXTRACTED or len(cleaned_text.strip()) < 20:
            return False

    return True


class ResilientPublicFetchAdapter(ContentFetchAdapter):
    """Two-tier public webpage/PDF fetcher: direct HTTP first, ScraperAPI fallback when needed."""

    def __init__(
        self,
        direct_adapter: Optional[ContentFetchAdapter] = None,
        scraperapi_adapter: Optional[ScraperAPIFetchAdapter] = None,
        *,
        enable_cache: bool = True,
    ) -> None:
        self.direct_adapter: ContentFetchAdapter = direct_adapter or UrllibPublicFetchAdapter()
        self.scraperapi_adapter: Optional[ScraperAPIFetchAdapter] = scraperapi_adapter
        self.enable_cache = enable_cache
        self._cache: dict[str, FetchResponse] = {}
        self.telemetry: dict[str, int] = {
            "direct_attempts": 0,
            "direct_successes": 0,
            "scraperapi_attempts": 0,
            "scraperapi_successes": 0,
            "cache_hits": 0,
            "failures": 0,
        }

    def is_configured(self) -> tuple[bool, str]:
        direct_ok, direct_msg = self.direct_adapter.is_configured()
        scraper_ok = False
        if self.scraperapi_adapter is not None:
            scraper_ok, _ = self.scraperapi_adapter.is_configured()
        if direct_ok and scraper_ok:
            return True, "Direct HTTP fetch + ScraperAPI fallback configured."
        if direct_ok:
            return True, "Direct HTTP fetch ready (ScraperAPI fallback not configured)."
        if scraper_ok:
            return True, "ScraperAPI retrieval ready."
        return False, direct_msg

    def fetch(self, url: str, *, timeout_seconds: float = 10.0, max_bytes: int = 2_000_000) -> FetchResponse:
        ok, reason = validate_safe_public_url(url)
        if not ok:
            return FetchResponse(
                url=url,
                requested_url=url,
                final_url="UNKNOWN",
                retrieval_method="DIRECT_HTTP",
                status_code=0,
                error_status=ExtractionStatus.BLOCKED_UNSAFE_URL,
                error_message=reason,
            )

        cache_key = canonicalize_url(url) or url
        if self.enable_cache and cache_key in self._cache:
            self.telemetry["cache_hits"] += 1
            return self._cache[cache_key]

        # 1. Try direct HTTP fetch first
        self.telemetry["direct_attempts"] += 1
        direct_resp = self.direct_adapter.fetch(url, timeout_seconds=timeout_seconds, max_bytes=max_bytes)
        if direct_resp.error_status == ExtractionStatus.BLOCKED_UNSAFE_URL:
            self.telemetry["failures"] += 1
            return direct_resp

        if _is_usable_fetch_response(direct_resp):
            self.telemetry["direct_successes"] += 1
            if self.enable_cache:
                self._cache[cache_key] = direct_resp
            return direct_resp

        # 2. Fall back to ScraperAPI if configured
        if self.scraperapi_adapter is not None:
            scraper_ok, _ = self.scraperapi_adapter.is_configured()
            if scraper_ok:
                self.telemetry["scraperapi_attempts"] += 1
                scraper_resp = self.scraperapi_adapter.fetch(
                    url,
                    timeout_seconds=max(timeout_seconds, 20.0),
                    max_bytes=max_bytes,
                )
                if scraper_resp.error_status is None and scraper_resp.content_bytes.strip():
                    self.telemetry["scraperapi_successes"] += 1
                    if self.enable_cache:
                        self._cache[cache_key] = scraper_resp
                    return scraper_resp
                self.telemetry["failures"] += 1
                return scraper_resp

        self.telemetry["failures"] += 1
        return direct_resp


def build_fetch_adapter_from_env(
    *,
    env: Optional[dict[str, str]] = None,
    allow_live_direct: bool = True,
    load_dotenv: bool = True,
) -> ContentFetchAdapter:
    """Instantiate a ContentFetchAdapter using environment settings (direct HTTP + optional ScraperAPI fallback)."""
    if env is None:
        if load_dotenv:
            from meridian_assessment.config.env_loader import load_dotenv_if_present

            load_dotenv_if_present()
        environ = os.environ
    else:
        environ = env

    scraper_key = (environ.get("MERIDIAN_SCRAPERAPI_KEY") or "").strip()
    scraper_adapter = ScraperAPIFetchAdapter(api_key=scraper_key) if scraper_key else None

    if not allow_live_direct and scraper_adapter is None:
        return DisabledOfflineFetchAdapter()

    direct_adapter: ContentFetchAdapter = (
        UrllibPublicFetchAdapter() if allow_live_direct else DisabledOfflineFetchAdapter()
    )
    return ResilientPublicFetchAdapter(
        direct_adapter=direct_adapter,
        scraperapi_adapter=scraper_adapter,
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
                requested_url=url,
                final_response_url="UNKNOWN",
                retrieval_method="BLOCKED_UNSAFE_URL",
                source_integrity_status="BLOCKED_UNSAFE_URL",
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
        final_resp_url = getattr(resp, "final_url", None) or (resp.url if resp.url != url else "UNKNOWN")
        retrieval_method = (
            "SYNTHETIC_SAMPLE"
            if is_synthetic_fixture
            else getattr(resp, "retrieval_method", "DIRECT_HTTP")
        )
        redirect_chain = list(getattr(resp, "redirect_chain", []) or [])
        resolved_canon = (
            canonicalize_url(final_resp_url)
            if final_resp_url and final_resp_url != "UNKNOWN"
            else canon_url
        ) or canon_url

        if resp.error_status is not None:
            return ExtractedDocument(
                document_id=doc_id,
                vendor_id=vendor_id,
                url=url,
                canonical_url=resolved_canon,
                requested_url=url,
                final_response_url=final_resp_url,
                http_status=resp.status_code,
                redirect_chain=redirect_chain,
                retrieval_method=retrieval_method,
                source_integrity_status="FETCH_ERROR",
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
            integrity_ok, integrity_status, integrity_warnings = validate_source_integrity(
                requested_url=url,
                final_response_url=final_resp_url,
                content_type="application/pdf",
                page_title=title,
            )
            if not integrity_ok:
                status = ExtractionStatus.SOURCE_IDENTITY_MISMATCH
                limits = list(limits) + list(integrity_warnings)
            return ExtractedDocument(
                document_id=doc_id,
                vendor_id=vendor_id,
                url=url,
                canonical_url=resolved_canon if integrity_ok else canon_url,
                requested_url=url,
                final_response_url=final_resp_url,
                http_status=resp.status_code,
                redirect_chain=redirect_chain,
                retrieval_method=retrieval_method,
                source_integrity_status=integrity_status,
                source_integrity_warnings=integrity_warnings,
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
                canonical_url=resolved_canon,
                requested_url=url,
                final_response_url=final_resp_url,
                http_status=resp.status_code,
                redirect_chain=redirect_chain,
                retrieval_method=retrieval_method,
                source_integrity_status="UNSUPPORTED_FORMAT",
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
        title, cleaned_text, _, status, decl_canon, decl_og = HTMLTextExtractor.extract_with_metadata(raw_html)
        limits = [] if status == ExtractionStatus.EXTRACTED else ["HTML document contained no extractable body text."]

        integrity_ok, integrity_status, integrity_warnings = validate_source_integrity(
            requested_url=url,
            final_response_url=final_resp_url,
            content_type=ctype or "text/html",
            declared_canonical_url=decl_canon,
            declared_og_url=decl_og,
            page_title=title,
        )
        if not integrity_ok:
            status = ExtractionStatus.SOURCE_IDENTITY_MISMATCH
            limits = list(limits) + list(integrity_warnings)

        return ExtractedDocument(
            document_id=doc_id,
            vendor_id=vendor_id,
            url=url,
            canonical_url=resolved_canon if integrity_ok else canon_url,
            requested_url=url,
            final_response_url=final_resp_url,
            http_status=resp.status_code,
            redirect_chain=redirect_chain,
            declared_canonical_url=decl_canon,
            declared_og_url=decl_og,
            retrieval_method=retrieval_method,
            source_integrity_status=integrity_status,
            source_integrity_warnings=integrity_warnings,
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
        `char_start` and `char_end` offsets. Never fabricates text for failed/empty extractions
        or documents flagged with `SOURCE_IDENTITY_MISMATCH`.
        """
        s_def = self.source_registry.get_source(doc.source_class)
        grade = s_def.default_evidence_grade if s_def else "C"
        rel_rating = s_def.reliability_rating if s_def else 2

        if (
            doc.extraction_status != ExtractionStatus.EXTRACTED
            or doc.source_integrity_status == "SOURCE_IDENTITY_MISMATCH"
            or not doc.cleaned_text.strip()
        ):
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
                    final_url=doc.final_response_url,
                    retrieval_method=doc.retrieval_method,
                    source_integrity_status=doc.source_integrity_status,
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
