"""Unit and regression tests for Serper search integration, ScraperAPI retrieval fallback,
and source-integrity validation (including the httpbin.org/json vs saas.group mismatch).

All tests use deterministic mocks and make zero live network requests.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse

import pytest

from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.models.ai_risk import (
    ExtractionStatus,
    URLTriageAction,
)
from meridian_assessment.models.osint_plan import (
    CandidateURL,
    SearchExecutionStatus,
    URLTriageStatus,
)
from meridian_assessment.services.end_to_end_pipeline import MeridianEndToEndPipeline
from meridian_assessment.services.extraction import (
    FetchResponse,
    OSINTExtractionService,
    ResilientPublicFetchAdapter,
    ScraperAPIFetchAdapter,
    StubFetchAdapter,
    build_fetch_adapter_from_env,
    validate_source_integrity,
)
from meridian_assessment.services.extraction.evidence_repository import EvidenceRepository
from meridian_assessment.services.osint import (
    OSINTInvestigationPlanner,
    OSINTSearchCollector,
    SearchConfigurationError,
    SearchProviderError,
    SearchQuotaExhaustedError,
    SearchRateLimitError,
    SearchResultItem,
    SearchTimeoutError,
    SerperSearchBackend,
    StubSearchBackend,
    build_search_backend_from_env,
)
from meridian_assessment.services.readiness_and_exports import SystemReadinessInspector


class _FakeHTTPResponse:
    """Context-manager mock for urllib.request.urlopen responses."""

    def __init__(
        self,
        body: bytes | str,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        url: str = "https://example.com",
    ) -> None:
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}
        self._url = url

    def read(self, max_bytes: int | None = None) -> bytes:
        return self._body if max_bytes is None else self._body[:max_bytes]

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


# ─── 1. Serper Configured / Unconfigured & Request/Response Parsing ───────────


def test_serper_configured_vs_unconfigured_and_secret_safety() -> None:
    """Verify SerperSearchBackend configuration state, readiness reporting, and secret safety."""
    unconf = build_search_backend_from_env(
        env={"MERIDIAN_SEARCH_PROVIDER": "serper", "MERIDIAN_SERPER_API_KEY": ""},
        load_dotenv=False,
    )
    assert isinstance(unconf, SerperSearchBackend)
    ok, msg = unconf.is_configured()
    assert ok is False
    assert "MERIDIAN_SERPER_API_KEY" in msg

    with pytest.raises(SearchConfigurationError):
        unconf.search("fiserv subprocessor ai")

    secret_key = "SERPER_SECRET_TOKEN_XYZ_987654"
    conf = build_search_backend_from_env(
        env={"MERIDIAN_SEARCH_PROVIDER": "serper", "MERIDIAN_SERPER_API_KEY": secret_key},
        load_dotenv=False,
    )
    assert isinstance(conf, SerperSearchBackend)
    ok_conf, _ = conf.is_configured()
    assert ok_conf is True

    readiness = SystemReadinessInspector.check_system_readiness(
        env={
            "MERIDIAN_SEARCH_PROVIDER": "serper",
            "MERIDIAN_SERPER_API_KEY": secret_key,
            "MERIDIAN_SCRAPERAPI_KEY": "SCRAPER_SECRET_TOKEN_12345",
        },
        load_dotenv=False,
        http_get_fn=lambda u, t: {"models": []},
    )
    assert readiness.search_provider == "serper"
    assert readiness.search_configured is True
    assert readiness.scraperapi_configured is True
    dumped = readiness.model_dump_json()
    assert secret_key not in dumped
    assert "SCRAPER_SECRET_TOKEN_12345" not in dumped


def test_serper_post_request_format_response_parsing_and_query_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SerperSearchBackend sends POST to https://google.serper.dev/search with X-API-KEY and caches duplicate queries."""
    captured_requests: list[dict[str, Any]] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 10.0) -> _FakeHTTPResponse:
        headers_lower = {k.lower(): v for k, v in req.header_items()}
        body_json = json.loads((req.data or b"{}").decode("utf-8"))
        captured_requests.append({
            "url": req.full_url,
            "method": req.get_method(),
            "headers": headers_lower,
            "body": body_json,
            "timeout": timeout,
        })
        payload = {
            "searchParameters": {"q": body_json.get("q"), "type": "search"},
            "organic": [
                {
                    "title": "Fiserv Trust Center - AI & Subprocessors",
                    "link": "https://trust.fiserv.com/ai-governance?utm_source=test",
                    "snippet": "Fiserv uses Azure OpenAI GPT-4o for DNA banking copilot.",
                    "position": 1,
                },
                {
                    "title": "Fiserv Privacy Notice",
                    "link": "https://www.fiserv.com/privacy",
                    "snippet": "Customer data protection and privacy commitments.",
                    "position": 2,
                },
            ],
        }
        return _FakeHTTPResponse(json.dumps(payload), status=200)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    backend = SerperSearchBackend(api_key="test-serper-key-abc", enable_cache=True)
    resp1 = backend.search("site:fiserv.com   AI subprocessor", max_results=5, timeout_seconds=8.0)

    assert len(captured_requests) == 1
    req_info = captured_requests[0]
    assert req_info["url"] == "https://google.serper.dev/search"
    assert req_info["method"] == "POST"
    assert req_info["headers"]["x-api-key"] == "test-serper-key-abc"
    assert req_info["headers"]["content-type"] == "application/json"
    assert req_info["body"] == {"q": "site:fiserv.com AI subprocessor", "num": 5}

    assert resp1.provider == "serper"
    assert len(resp1.results) == 2
    assert resp1.results[0].url == "https://trust.fiserv.com/ai-governance?utm_source=test"
    assert resp1.results[0].title == "Fiserv Trust Center - AI & Subprocessors"
    assert resp1.results[0].rank == 1

    # Duplicate query with extra whitespace hits cache without making a second HTTP POST
    resp2 = backend.search("site:fiserv.com AI subprocessor", max_results=5)
    assert len(captured_requests) == 1
    assert backend.telemetry["requests_made"] == 1
    assert backend.telemetry["cache_hits"] == 1
    assert len(resp2.results) == 2


def test_serper_error_handling_and_collector_halts_on_auth_or_quota_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Serper 401/403, quota exhaustion, 429, 500, timeout, malformed JSON, and collector halt guard."""
    # 1. Quota exhaustion (HTTP 400 with "Not enough credits")
    def fake_quota_err(req: urllib.request.Request, timeout: float = 10.0) -> Any:
        err_fp = io.BytesIO(b'{"message": "Not enough credits"}')
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, err_fp)

    monkeypatch.setattr(urllib.request, "urlopen", fake_quota_err)
    backend = SerperSearchBackend(api_key="key-123", enable_cache=False)
    with pytest.raises(SearchQuotaExhaustedError):
        backend.search("fiserv ai")

    # 2. Collector stops issuing further HTTP requests after first quota exhaustion or auth error
    call_count = 0

    def fake_counting_quota_err(req: urllib.request.Request, timeout: float = 10.0) -> Any:
        nonlocal call_count
        call_count += 1
        err_fp = io.BytesIO(b'{"message": "Not enough credits"}')
        raise urllib.error.HTTPError(req.full_url, 402, "Payment Required", {}, err_fp)

    monkeypatch.setattr(urllib.request, "urlopen", fake_counting_quota_err)
    vendors = MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))
    plan = OSINTInvestigationPlanner().plan_investigation(vendors[0], criticality_input="High")
    assert len(plan.planned_queries) >= 5

    collector = OSINTSearchCollector(
        backend=SerperSearchBackend(api_key="key-123", enable_cache=False),
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    collector.execute_planned_queries(plan, persist=False)
    # Only 1 network call should have been made before halting the remaining queries!
    assert call_count == 1
    assert all(q.execution_status == SearchExecutionStatus.FAILED for q in plan.planned_queries)

    # 3. HTTP 429 rate limit -> SearchRateLimitError
    def fake_429(req: urllib.request.Request, timeout: float = 10.0) -> Any:
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"rate limit"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_429)
    with pytest.raises(SearchRateLimitError):
        backend.search("fiserv rate limit")

    # 4. Timeout -> SearchTimeoutError
    def fake_timeout(req: urllib.request.Request, timeout: float = 10.0) -> Any:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_timeout)
    with pytest.raises(SearchTimeoutError):
        backend.search("fiserv timeout")

    # 5. Malformed JSON -> SearchProviderError
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=10.0: _FakeHTTPResponse(b"{not-valid-json", status=200),
    )
    with pytest.raises(SearchProviderError):
        backend.search("fiserv bad json")


# ─── 2. ScraperAPI & Direct-Fetch-First Fallback ──────────────────────────────


def test_scraperapi_adapter_request_format_and_secret_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ScraperAPIFetchAdapter calls https://api.scraperapi.com without render=true by default and never leaks api_key."""
    captured_urls: list[str] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 20.0) -> _FakeHTTPResponse:
        captured_urls.append(req.full_url)
        return _FakeHTTPResponse(
            "<html><head><title>Vendor Trust Page</title></head><body><p>Our service uses Anthropic Claude 3.5 Sonnet with zero data retention.</p></body></html>",
            status=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    secret = "SCRAPER_SECRET_KEY_777888"
    adapter = ScraperAPIFetchAdapter(api_key=secret, max_retries=0)
    resp = adapter.fetch("https://trust.vendor.example.com/security")

    assert len(captured_urls) == 1
    parsed_req = urllib.parse.urlsplit(captured_urls[0])
    assert f"{parsed_req.scheme}://{parsed_req.netloc}{parsed_req.path}" == "https://api.scraperapi.com"
    q_params = dict(urllib.parse.parse_qsl(parsed_req.query))
    assert q_params["api_key"] == secret
    assert q_params["url"] == "https://trust.vendor.example.com/security"
    assert "render" not in q_params

    assert resp.error_status is None
    assert resp.retrieval_method == "SCRAPERAPI"
    assert resp.requested_url == "https://trust.vendor.example.com/security"
    assert resp.final_url == "UNKNOWN"
    assert secret not in (resp.url or "")
    assert secret not in (resp.final_url or "")

    # Verify error redaction if exception message contains the API key
    def fake_err_with_secret(req: urllib.request.Request, timeout: float = 20.0) -> Any:
        raise urllib.error.URLError(f"Connection failed for {req.full_url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_err_with_secret)
    err_resp = adapter.fetch("https://trust.vendor.example.com/security")
    assert err_resp.error_status == ExtractionStatus.FETCH_FAILED
    assert secret not in (err_resp.error_message or "")
    assert "[REDACTED]" in (err_resp.error_message or "")


def test_resilient_fetch_adapter_direct_first_then_scraperapi_fallback() -> None:
    """Verify ResilientPublicFetchAdapter uses direct HTTP when usable and falls back to ScraperAPI only on failure/block."""
    direct_html = (
        "<html><head><title>Direct Docs</title></head>"
        "<body><h1>AI Features</h1><p>This platform embeds OpenAI GPT-4o for automated ticket triage.</p></body></html>"
    )
    cloudflare_block_html = (
        "<html><head><title>Just a moment...</title></head>"
        "<body>Enable JavaScript and cookies to continue</body></html>"
    )
    scraper_recovered_html = (
        "<html><head><title>Recovered Vendor Subprocessors</title></head>"
        "<body><h1>Subprocessors</h1><p>Amazon Bedrock is used for generative AI summarization in production.</p></body></html>"
    )

    direct_stub = StubFetchAdapter(
        fixtures={
            "https://docs.vendor.com/direct-ok": {
                "body": direct_html,
                "status_code": 200,
                "content_type": "text/html",
                "retrieval_method": "DIRECT_HTTP",
            },
            "https://trust.vendor.com/blocked-page": {
                "body": cloudflare_block_html,
                "status_code": 200,
                "content_type": "text/html",
                "retrieval_method": "DIRECT_HTTP",
            },
        }
    )

    class _MockScraperAPI(ScraperAPIFetchAdapter):
        def __init__(self) -> None:
            super().__init__(api_key="mock-scraper-key")
            self.called_urls: list[str] = []

        def fetch(self, url: str, *, timeout_seconds: float = 20.0, max_bytes: int = 2_000_000, render_js: bool | None = None) -> FetchResponse:
            self.called_urls.append(url)
            return FetchResponse(
                url=url,
                requested_url=url,
                final_url="UNKNOWN",
                retrieval_method="SCRAPERAPI",
                status_code=200,
                content_type="text/html",
                content_bytes=scraper_recovered_html.encode("utf-8"),
            )

    mock_scraper = _MockScraperAPI()
    resilient = ResilientPublicFetchAdapter(
        direct_adapter=direct_stub,
        scraperapi_adapter=mock_scraper,
        enable_cache=True,
    )

    # Case 1: Direct fetch succeeds -> ScraperAPI is NOT called
    r1 = resilient.fetch("https://docs.vendor.com/direct-ok")
    assert r1.retrieval_method == "DIRECT_HTTP"
    assert len(mock_scraper.called_urls) == 0
    assert resilient.telemetry["direct_successes"] == 1
    assert resilient.telemetry["scraperapi_attempts"] == 0

    # Case 2: Direct fetch returns Cloudflare challenge -> ScraperAPI fallback is invoked and succeeds
    r2 = resilient.fetch("https://trust.vendor.com/blocked-page")
    assert r2.retrieval_method == "SCRAPERAPI"
    assert mock_scraper.called_urls == ["https://trust.vendor.com/blocked-page"]
    assert resilient.telemetry["scraperapi_attempts"] == 1
    assert resilient.telemetry["scraperapi_successes"] == 1

    # Case 3: Repeated fetch of same URL hits cache
    r3 = resilient.fetch("https://trust.vendor.com/blocked-page")
    assert r3.retrieval_method == "SCRAPERAPI"
    assert len(mock_scraper.called_urls) == 1
    assert resilient.telemetry["cache_hits"] == 1


# ─── 3. Source-Integrity Validation (httpbin.org/json vs saas.group) ──────────


def test_httpbin_vs_saas_group_source_identity_mismatch_rejected_from_evidence(
    tmp_path: Path,
) -> None:
    """Verify requesting https://httpbin.org/json/ and receiving saas.group HTML triggers SOURCE_IDENTITY_MISMATCH and produces zero evidence records."""
    saas_group_html = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>saas.group - A great new home for your SaaS business</title>
        <link rel="canonical" href="https://saas.group/" />
        <meta property="og:url" content="https://saas.group/" />
    </head>
    <body>
        <h1>We acquire promising SaaS companies</h1>
        <p>Our portfolio uses OpenAI GPT-4 and Anthropic Claude across multiple business units.</p>
    </body>
    </html>
    """

    stub_fetch = StubFetchAdapter(
        fixtures={
            "https://httpbin.org/json": FetchResponse(
                url="https://httpbin.org/json/",
                requested_url="https://httpbin.org/json/",
                final_url="UNKNOWN",
                retrieval_method="SCRAPERAPI",
                status_code=200,
                content_type="text/html",
                content_bytes=saas_group_html.encode("utf-8"),
            )
        }
    )

    extractor = OSINTExtractionService(fetch_adapter=stub_fetch)
    doc = extractor.extract_from_url(
        vendor_id="V-TEST",
        url="https://httpbin.org/json/",
        source_class="S2",
        triage_action="FETCH",
    )

    assert doc.extraction_status == ExtractionStatus.SOURCE_IDENTITY_MISMATCH
    assert doc.source_integrity_status == "SOURCE_IDENTITY_MISMATCH"
    assert doc.requested_url == "https://httpbin.org/json/"
    assert doc.final_response_url == "UNKNOWN"
    assert doc.declared_canonical_url == "https://saas.group/"
    assert doc.declared_og_url == "https://saas.group/"
    assert doc.retrieval_method == "SCRAPERAPI"
    # Canonical URL must NOT be overwritten by untrusted HTML metadata from saas.group
    assert "saas.group" not in doc.canonical_url
    assert any("saas.group" in lim for lim in doc.extraction_limitations)

    # Mismatched document must NEVER be converted into EvidenceRecord items
    ev_records = extractor.build_evidence_records_from_document(
        doc,
        legal_entity="Httpbin Test Vendor",
        domain="httpbin.org",
    )
    assert ev_records == []

    # Verify full end-to-end pipeline preserves diagnostic record while keeping evidence_records empty
    vendors = MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))
    test_vendor = vendors[0].model_copy(update={"domain": "httpbin.org", "vendor_name": "Httpbin Corp"})
    stub_search = StubSearchBackend(
        default_results=[
            SearchResultItem(
                url="https://httpbin.org/json/",
                title="Httpbin JSON API Documentation",
                snippet="Httpbin AI documentation endpoint.",
                rank=1,
            )
        ]
    )
    repo = EvidenceRepository(
        db_path=tmp_path / "mismatch.sqlite3",
        artifact_dir=tmp_path / "mismatch_artifacts",
    )
    pipeline = MeridianEndToEndPipeline(
        search_backend=stub_search,
        fetch_adapter=stub_fetch,
        evidence_repository=repo,
        enable_local_llm_from_env=False,
    )
    report = pipeline.run_vendor_assessment(
        test_vendor,
        execute_searches=True,
        execute_extraction=True,
        persist_to_repository=True,
    )

    assert len(report.extracted_documents) >= 1
    assert all(
        d.extraction_status == ExtractionStatus.SOURCE_IDENTITY_MISMATCH
        for d in report.extracted_documents
    )
    assert report.evidence_records == []
    assert report.risk_findings == []
    assert any("Source-identity mismatch" in trig for trig in report.human_review_triggers)


def test_same_host_canonical_path_difference_warns_without_discarding_valid_evidence() -> None:
    """Verify a same-host canonical path difference produces a warning but keeps valid vendor documentation."""
    vendor_html = """<!DOCTYPE html>
    <html>
    <head>
        <title>Fiserv AI Governance & Subprocessors</title>
        <link rel="canonical" href="https://trust.fiserv.com/en-us/ai-governance-policy" />
        <meta property="og:url" content="https://trust.fiserv.com/en-us/ai-governance-policy" />
    </head>
    <body>
        <h1>AI Subprocessor Disclosure</h1>
        <p>Fiserv DNA banking platform integrates Azure OpenAI Service for customer service assistance.</p>
    </body>
    </html>
    """
    stub_fetch = StubFetchAdapter(
        fixtures={
            "https://trust.fiserv.com/ai-policy": FetchResponse(
                url="https://trust.fiserv.com/ai-policy",
                requested_url="https://trust.fiserv.com/ai-policy",
                final_url="https://trust.fiserv.com/ai-policy",
                retrieval_method="DIRECT_HTTP",
                status_code=200,
                content_type="text/html",
                content_bytes=vendor_html.encode("utf-8"),
            )
        }
    )
    extractor = OSINTExtractionService(fetch_adapter=stub_fetch)
    doc = extractor.extract_from_url(
        vendor_id="V-002",
        url="https://trust.fiserv.com/ai-policy",
        source_class="S1",
        triage_action="FETCH",
    )

    assert doc.extraction_status == ExtractionStatus.EXTRACTED
    assert doc.source_integrity_status == "WARNING_SAME_HOST_PATH_DIFF"
    assert len(doc.source_integrity_warnings) >= 1
    # Must preserve actual requested/response URL rather than overwriting with untrusted canonical metadata
    assert doc.canonical_url == "https://trust.fiserv.com/ai-policy"

    ev_records = extractor.build_evidence_records_from_document(
        doc,
        legal_entity="Fiserv",
        domain="fiserv.com",
    )
    assert len(ev_records) >= 1
    assert ev_records[0].source_integrity_status == "WARNING_SAME_HOST_PATH_DIFF"


# ─── 4. Sole Source O2 Override, Per-Stage AI Toggles, Large Doc Cleaning & Intra-Level Progress ───


def test_sole_source_checkbox_triggers_o2_critical_override() -> None:
    """Verify is_sole_source=True with O=3 triggers Override O2 and elevates tier to Critical."""
    from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine

    vendors = MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))
    # Create a vendor with O=3 (high operational criticality) and no "no alternative" text,
    # then toggle is_sole_source=True vs False.
    base_vendor = vendors[0].model_copy(
        update={
            "vendor_name": "Acme Payments Engine",
            "service_product_provided": "Real-time payment clearing gateway",
            "business_process_supported": "Core real-time payment settlement and ledger posting",
            "operational_dependency": "Critical",
            "notes": "Standard enterprise contract.",
            "is_sole_source": False,
        }
    )
    engine = MeridianCriticalityEngine()
    res_not_sole = engine.evaluate(base_vendor)
    o2_not_sole = next(ov for ov in res_not_sole.overrides_evaluated if ov.override_id == "O2")

    sole_vendor = base_vendor.model_copy(update={"is_sole_source": True})
    res_sole = engine.evaluate(sole_vendor)
    o2_sole = next(ov for ov in res_sole.overrides_evaluated if ov.override_id == "O2")

    assert res_sole.O.score == 3
    assert o2_not_sole.triggered is False
    assert o2_sole.triggered is True
    assert "is_sole_source=True" in (o2_sole.evidence or "")
    assert res_sole.final_criticality == "Critical"


def test_large_document_cleaning_and_per_stage_progress_callbacks(tmp_path: Path) -> None:
    """Verify >20 block documents are deduplicated/bounded and pipeline emits (level, title, detail, completed, total)."""
    from meridian_assessment.models.ai_risk import ExtractedDocument

    # Build a synthetic document with 120 noisy boilerplate paragraphs and 4 high-signal AI paragraphs
    blocks: list[str] = []
    for i in range(60):
        blocks.append(f"Navigation link item {i}")  # short noise (<45 chars without AI keywords)
        blocks.append("Copyright 2026 Vendor Corp. All rights reserved worldwide. Cookie preferences and terms.")  # repeated boilerplate
    blocks.append(
        "AutomWorx embeds OpenAI GPT-4o and Anthropic Claude 3.5 Sonnet for automated exception routing in production."
    )
    blocks.append(
        "Customer production data is processed under a zero data retention subprocessor agreement with Microsoft Azure OpenAI."
    )
    blocks.append(
        "Model training on customer prompts is disabled by default and SOC 2 Type II controls govern AI inference pipelines."
    )
    blocks.append(
        "Automated underwriting workflow decisions require human-in-the-loop review before final credit denial."
    )

    doc = ExtractedDocument(
        document_id="DOC_LARGE_001",
        vendor_id="V-001",
        url="https://trust.automworx.com/ai-architecture",
        canonical_url="https://trust.automworx.com/ai-architecture",
        source_class="S1",
        retrieved_at="2026-10-13T00:00:00Z",
        extraction_status=ExtractionStatus.EXTRACTED,
        cleaned_text="\n\n".join(blocks),
    )

    extractor = OSINTExtractionService(fetch_adapter=StubFetchAdapter())
    records = extractor.build_evidence_records_from_document(
        doc,
        legal_entity="AutomWorx",
        domain="automworx.com",
        max_records_per_document=25,
    )
    # Instead of 124 raw records, cleaning + deduplication + bounding keeps only clean relevant records (<= 25)
    assert 4 <= len(records) <= 25
    combined_excerpts = " ".join(r.verbatim_excerpt for r in records)
    assert "GPT-4o" in combined_excerpts
    assert "zero data retention" in combined_excerpts

    # Verify per-stage toggles and 5-arg progress callback on MeridianEndToEndPipeline
    vendors = MeridianCSVVendorLoader().load(Path("data/input/meridian_vendors.csv"))
    progress_events: list[tuple[int, str, str, int, int]] = []

    def on_progress(lvl: int, title: str, detail: str, completed: int, total: int) -> None:
        progress_events.append((lvl, title, detail, completed, total))

    pipeline = MeridianEndToEndPipeline(
        search_backend=StubSearchBackend(),
        fetch_adapter=StubFetchAdapter(),
        evidence_repository=EvidenceRepository(
            db_path=tmp_path / "prog.sqlite3",
            artifact_dir=tmp_path / "prog_artifacts",
        ),
        enable_local_llm_from_env=False,
        enable_triage_llm=False,
        enable_rag_embeddings=False,
        enable_classification_llm=False,
        enable_summary_llm=False,
    )
    assert pipeline.triage_service.llm_provider.is_configured()[0] is False
    assert type(pipeline.rag_service.embedding_provider).__name__ == "UnconfiguredEmbeddingProvider"
    assert pipeline.classification_service.llm_adapter.is_configured()[0] is False
    assert pipeline.risk_analyzer.summary_llm_adapter.is_configured()[0] is False

    pipeline.run_vendor_assessment(
        vendors[0],
        execute_searches=False,
        execute_extraction=False,
        persist_to_repository=False,
        progress_callback=on_progress,
    )
    assert len(progress_events) >= 9
    assert progress_events[0][0] == 1
    assert all(ev[4] >= 1 and 0 <= ev[3] <= ev[4] for ev in progress_events)


def test_ssl_read_timeout_handled_cleanly_without_crashing_pipeline():
    """Regression test: Verify ssl.SSLError 'The operation did not complete (read)' is gracefully caught."""
    import ssl
    from unittest.mock import MagicMock, patch
    from meridian_assessment.services.extraction.extractor import UrllibPublicFetchAdapter

    adapter = UrllibPublicFetchAdapter()
    with patch("urllib.request.build_opener") as mock_opener:
        mock_instance = MagicMock()
        mock_instance.open.side_effect = ssl.SSLError("The operation did not complete (read) (_ssl.c:2559)")
        mock_opener.return_value = mock_instance

        resp = adapter.fetch("https://www.reddit.com/r/Futurology/test")
        assert resp.error_status == ExtractionStatus.TIMEOUT
        assert "SSL transport error" in resp.error_message
        assert "_ssl.c:2559" in resp.error_message

    # Also verify extraction service handles it cleanly when fetch raises directly
    faulty_adapter = MagicMock()
    faulty_adapter.fetch.side_effect = ssl.SSLError("The operation did not complete (read) (_ssl.c:2559)")
    svc = OSINTExtractionService(fetch_adapter=faulty_adapter)
    doc = svc.extract_from_url(
        vendor_id="V-005",
        url="https://www.reddit.com/r/Futurology/test",
    )
    assert doc.extraction_status in (ExtractionStatus.TIMEOUT, ExtractionStatus.FETCH_FAILED)
    assert doc.source_integrity_status == "FETCH_ERROR"


