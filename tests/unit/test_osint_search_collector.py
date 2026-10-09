"""Offline unit tests for Phase 1 Bounded Search Execution and Candidate-URL Collection.

All tests use offline StubSearchBackend, mocked HTTP handlers, or local fixtures.
Zero live network requests are made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from meridian_assessment.models.factor_result import HumanReviewRecord
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.models.osint_plan import (
    AssessmentStopStatus,
    CoverageState,
    GateStatus,
    OSINTInvestigationPlan,
    PlannedQuery,
    SearchExecutionStatus,
    URLTriageStatus,
)
from meridian_assessment.services.osint import (
    OSINTCoverageTracker,
    OSINTInvestigationPlanner,
    OSINTSearchCollector,
    SearchProviderError,
    SearchRateLimitError,
    SearchResultItem,
    SearchTimeoutError,
    StubSearchBackend,
    UnconfiguredSearchBackend,
    build_search_backend_from_env,
    canonicalize_url,
)


@pytest.fixture
def sample_vendor() -> MeridianVendor:
    return MeridianVendor(
        vendor_id="V-002",
        vendor_name="Fiserv",
        domain="fiserv.com",
        service_product_provided="Core banking platform",
        business_process_supported="Core banking operations",
        data_classification_accessed="Customer master data and PII",
        operational_dependency="Critical",
        data_volume_annual="240 million transactions annually",
    )


@pytest.fixture
def low_plan(sample_vendor: MeridianVendor) -> OSINTInvestigationPlan:
    planner = OSINTInvestigationPlanner()
    return planner.plan_investigation(sample_vendor, criticality_input="Low")


# ─── 1. Executing PLANNED Queries & Recording Candidate URLs ──────────────────


def test_execute_planned_query_records_candidate_urls_and_metadata(low_plan: OSINTInvestigationPlan) -> None:
    first_query = low_plan.planned_queries[0]
    stub = StubSearchBackend(
        responses={
            first_query.rendered_query: [
                {
                    "url": "https://www.fiserv.com/en/trust-center/subprocessors.html",
                    "title": "Fiserv Trust Center — Subprocessor Register",
                    "snippet": "Official list of third-party subprocessors and cloud infrastructure providers.",
                    "rank": 1,
                    "retrieved_at": "2026-10-10T10:00:00+00:00",
                },
                {
                    "url": "https://www.fiserv.com/en/legal/privacy-notice.html",
                    "title": "Fiserv Privacy Notice",
                    "snippet": "Details on customer data handling and retention.",
                    "rank": 2,
                    "retrieved_at": "2026-10-10T10:00:00+00:00",
                },
            ]
        },
        provider_name="mock_google_cse",
    )
    collector = OSINTSearchCollector(backend=stub, retry_backoff_seconds=0.0)
    collector.execute_planned_queries(low_plan, query_ids=[first_query.query_id])

    assert first_query.execution_status == SearchExecutionStatus.EXECUTED
    assert first_query.provider == "mock_google_cse"
    assert first_query.executed_at is not None
    assert first_query.result_count == 2
    assert first_query.attempts == 1
    assert first_query.error_message is None

    assert len(low_plan.candidate_urls) == 2
    c1 = low_plan.candidate_urls[0]
    assert c1.url == "https://www.fiserv.com/en/trust-center/subprocessors.html"
    assert c1.title == "Fiserv Trust Center — Subprocessor Register"
    assert c1.snippet == "Official list of third-party subprocessors and cloud infrastructure providers."
    assert c1.rank == 1
    assert c1.retrieved_at == "2026-10-10T10:00:00+00:00"
    assert c1.provider == "mock_google_cse"
    assert c1.vendor_id == "V-002"
    assert first_query.query_id in c1.query_ids
    assert set(first_query.source_classes).issubset(set(c1.source_classes))
    assert c1.triage_status == URLTriageStatus.PENDING


# ─── 2. Executing Query with Zero Results ─────────────────────────────────────


def test_execute_query_with_zero_results_marks_search_attempted_only(low_plan: OSINTInvestigationPlan) -> None:
    stub = StubSearchBackend(default_results=[], provider_name="mock_empty")
    collector = OSINTSearchCollector(backend=stub, retry_backoff_seconds=0.0)
    collector.execute_planned_queries(low_plan)

    assert all(q.execution_status == SearchExecutionStatus.EXECUTED for q in low_plan.planned_queries)
    assert all(q.result_count == 0 for q in low_plan.planned_queries)
    assert len(low_plan.candidate_urls) == 0

    # Sources with executed queries and 0 results become SEARCH_ATTEMPTED, never REVIEWED_NO_EVIDENCE
    for s_id in ("S1", "S2", "S3"):
        rec = low_plan.coverage_records[s_id]
        assert rec.status == CoverageState.SEARCH_ATTEMPTED
        assert rec.candidate_urls_count == 0
        assert rec.reviewed_urls_count == 0
        assert rec.produced_new_evidence is False


# ─── 3. Handling Malformed or Incomplete Result Items ─────────────────────────


def test_malformed_result_items_are_filtered_safely(low_plan: OSINTInvestigationPlan) -> None:
    target_q = low_plan.planned_queries[0]
    stub = StubSearchBackend(
        responses={
            target_q.rendered_query: [
                None,
                "not-a-dict",
                {},
                {"title": "Missing URL field"},
                {"url": ""},
                {"url": "javascript:void(0)", "title": "Invalid scheme"},
                {"url": "mailto:security@fiserv.com", "title": "Mailto link"},
                {"url": "https://valid.fiserv.com/docs/api", "title": "Valid API Doc", "snippet": "REST API"},
            ]
        },
        provider_name="mock_malformed",
    )
    collector = OSINTSearchCollector(backend=stub, retry_backoff_seconds=0.0)
    collector.execute_planned_queries(low_plan, query_ids=[target_q.query_id])

    assert target_q.execution_status == SearchExecutionStatus.EXECUTED
    assert target_q.result_count == 1
    assert len(low_plan.candidate_urls) == 1
    assert low_plan.candidate_urls[0].url == "https://valid.fiserv.com/docs/api"


def test_non_list_provider_payload_marks_query_failed(low_plan: OSINTInvestigationPlan) -> None:
    target_q = low_plan.planned_queries[0]
    stub = StubSearchBackend(
        responses={target_q.rendered_query: "corrupted-payload"},
        provider_name="mock_bad_payload",
    )
    collector = OSINTSearchCollector(backend=stub, max_retries=2, retry_backoff_seconds=0.0)
    collector.execute_planned_queries(low_plan, query_ids=[target_q.query_id])

    assert target_q.execution_status == SearchExecutionStatus.FAILED
    assert target_q.attempts == 1
    assert target_q.result_count == 0
    assert "Malformed provider payload" in (target_q.error_message or "")


# ─── 4. Provider Error, Timeout, Rate-Limit & Bounded Retries ─────────────────


def test_timeout_and_rate_limit_perform_bounded_retries_then_recover_or_fail(low_plan: OSINTInvestigationPlan) -> None:
    q1, q2, q3 = low_plan.planned_queries[0], low_plan.planned_queries[1], low_plan.planned_queries[2]
    call_counts: dict[str, int] = {q1.rendered_query: 0, q2.rendered_query: 0, q3.rendered_query: 0}

    def side_effect(query: str, max_results: int, timeout: float) -> Any:
        call_counts[query] = call_counts.get(query, 0) + 1
        if query == q1.rendered_query:
            # Succeeds on 2nd attempt after transient rate limit
            if call_counts[query] == 1:
                raise SearchRateLimitError("HTTP 429 rate limit")
            return [SearchResultItem(url="https://fiserv.com/trust/dpa", title="Fiserv DPA", rank=1)]
        if query == q2.rendered_query:
            # Always times out -> exhausts max_retries
            raise SearchTimeoutError("Request timed out after 5.0s")
        # Non-retryable provider error -> fails immediately on attempt 1
        raise SearchProviderError("HTTP 400 Bad Request: invalid query syntax", retryable=False)

    stub = StubSearchBackend(responses=side_effect, provider_name="mock_resilient")
    collector = OSINTSearchCollector(
        backend=stub,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    collector.execute_planned_queries(
        low_plan,
        query_ids=[q1.query_id, q2.query_id, q3.query_id],
    )

    # q1 recovered on attempt 2
    assert q1.execution_status == SearchExecutionStatus.EXECUTED
    assert q1.attempts == 2
    assert q1.result_count == 1
    assert q1.error_message is None

    # q2 exhausted 1 + 2 = 3 attempts
    assert q2.execution_status == SearchExecutionStatus.FAILED
    assert q2.attempts == 3
    assert q2.result_count == 0
    assert "timed out" in (q2.error_message or "").lower()

    # q3 failed immediately without retry
    assert q3.execution_status == SearchExecutionStatus.FAILED
    assert q3.attempts == 1
    assert q3.result_count == 0
    assert "HTTP 400" in (q3.error_message or "")


# ─── 5. Missing Provider Configuration or Credentials ─────────────────────────


def test_missing_provider_credentials_marks_queries_unavailable_without_faking_search_attempted(
    low_plan: OSINTInvestigationPlan,
) -> None:
    backend = build_search_backend_from_env(env={})
    assert isinstance(backend, UnconfiguredSearchBackend)
    ok, reason = backend.is_configured()
    assert ok is False
    assert "MERIDIAN_SEARCH_PROVIDER" in reason

    collector = OSINTSearchCollector(backend=backend, retry_backoff_seconds=0.0)
    collector.execute_planned_queries(low_plan)

    assert all(q.execution_status == SearchExecutionStatus.UNAVAILABLE for q in low_plan.planned_queries)
    assert all(q.error_message for q in low_plan.planned_queries)
    assert len(low_plan.candidate_urls) == 0

    # Source coverage must remain NOT_STARTED (not falsely marked SEARCH_ATTEMPTED or REVIEWED_NO_EVIDENCE)
    for rec in low_plan.coverage_records.values():
        assert rec.status == CoverageState.NOT_STARTED
        assert rec.candidate_urls_count == 0


# ─── 6. Conservative URL Canonicalization & Multi-Query Deduplication ─────────


def test_canonicalize_url_preserves_meaningful_paths_and_params_while_stripping_fragments_and_utm() -> None:
    raw = "HTTPS://Docs.Fiserv.com:443/API/v2/Models/?utm_source=google&model_id=gpt-4o&page=2&fbclid=xyz#section-3"
    canon = canonicalize_url(raw)
    assert canon == "https://docs.fiserv.com/API/v2/Models?model_id=gpt-4o&page=2"

    # Root path normalized
    assert canonicalize_url("http://fiserv.com:80#top") == "http://fiserv.com/"
    # Non-standard port preserved
    assert canonicalize_url("https://portal.fiserv.com:8443/trust/") == "https://portal.fiserv.com:8443/trust"
    # Invalid URLs rejected
    assert canonicalize_url("not a url") is None
    assert canonicalize_url("ftp://files.fiserv.com/doc.pdf") is None


def test_deduplicate_urls_across_multiple_queries_and_merge_provenance(low_plan: OSINTInvestigationPlan) -> None:
    # Pick one S1 query and one S2/S3 query
    q_s1 = next(q for q in low_plan.planned_queries if "S1" in q.source_classes)
    q_other = next(q for q in low_plan.planned_queries if q.query_id != q_s1.query_id)
    # Ensure q_other has S2 so we test multi-source-class merging
    if "S2" not in q_other.source_classes:
        q_other.source_classes.append("S2")

    stub = StubSearchBackend(
        responses={
            q_s1.rendered_query: [
                {
                    "url": "https://www.fiserv.com/trust/ai-governance/?utm_source=search#overview",
                    "title": "",
                    "snippet": "Initial snippet from S1 query.",
                    "rank": 4,
                }
            ],
            q_other.rendered_query: [
                {
                    "url": "https://WWW.FISERV.COM:443/trust/ai-governance",
                    "title": "Fiserv AI Governance & Subprocessors",
                    "snippet": "Detailed snippet.",
                    "rank": 1,
                }
            ],
        },
        provider_name="mock_dedup",
    )
    collector = OSINTSearchCollector(backend=stub, retry_backoff_seconds=0.0)
    collector.execute_planned_queries(low_plan, query_ids=[q_s1.query_id, q_other.query_id])

    assert len(low_plan.candidate_urls) == 1
    merged = low_plan.candidate_urls[0]
    assert merged.url == "https://www.fiserv.com/trust/ai-governance"
    assert q_s1.query_id in merged.query_ids
    assert q_other.query_id in merged.query_ids
    assert q_s1.rendered_query in merged.executed_queries
    assert q_other.rendered_query in merged.executed_queries
    assert "S1" in merged.source_classes
    assert "S2" in merged.source_classes
    # Filled missing title from second query and kept best (lowest) rank
    assert merged.title == "Fiserv AI Governance & Subprocessors"
    assert merged.rank == 1


# ─── 7. Enforcing Result-Count Limits & Max Queries Per Run ──────────────────


def test_enforces_max_results_per_query_and_max_queries_per_run(low_plan: OSINTInvestigationPlan) -> None:
    many_items = [
        {"url": f"https://fiserv.com/doc/{i}", "title": f"Doc {i}", "rank": i}
        for i in range(1, 15)
    ]
    stub = StubSearchBackend(default_results=many_items, provider_name="mock_limits")
    collector = OSINTSearchCollector(
        backend=stub,
        max_results_per_query=3,
        max_queries_per_run=2,
        retry_backoff_seconds=0.0,
    )
    collector.execute_planned_queries(low_plan)

    executed = [q for q in low_plan.planned_queries if q.execution_status == SearchExecutionStatus.EXECUTED]
    planned_left = [q for q in low_plan.planned_queries if q.execution_status == SearchExecutionStatus.PLANNED]

    assert len(executed) == 2
    assert len(planned_left) == len(low_plan.planned_queries) - 2
    assert all(q.result_count == 3 for q in executed)
    assert len(low_plan.candidate_urls) == 3


# ─── 8. Skipping Already-Executed Queries Unless Retry Requested ──────────────


def test_skips_already_executed_and_failed_queries_unless_retry_requested(low_plan: OSINTInvestigationPlan) -> None:
    q1, q2 = low_plan.planned_queries[0], low_plan.planned_queries[1]
    should_fail = True

    def handler(query: str, max_results: int, timeout: float) -> Any:
        if query == q2.rendered_query and should_fail:
            raise SearchProviderError("Upstream error", retryable=False)
        return [{"url": f"https://fiserv.com/res/{ abs(hash(query)) % 1000 }", "title": "Result"}]

    stub = StubSearchBackend(responses=handler, provider_name="mock_skip")
    collector = OSINTSearchCollector(backend=stub, max_retries=0, retry_backoff_seconds=0.0)

    # First run on q1 and q2: q1 succeeds, q2 fails
    collector.execute_planned_queries(low_plan, query_ids=[q1.query_id, q2.query_id])
    assert q1.execution_status == SearchExecutionStatus.EXECUTED
    assert q2.execution_status == SearchExecutionStatus.FAILED
    assert len(stub.call_history) == 2

    # Second run without retry_failed: neither q1 nor q2 is called again
    collector.execute_planned_queries(low_plan, query_ids=[q1.query_id, q2.query_id], retry_failed=False)
    assert len(stub.call_history) == 2

    # Third run with retry_failed=True: only q2 is retried (q1 remains skipped)
    should_fail = False
    collector.execute_planned_queries(low_plan, query_ids=[q1.query_id, q2.query_id], retry_failed=True)
    assert len(stub.call_history) == 3
    assert stub.call_history[-1]["query"] == q2.rendered_query
    assert q2.execution_status == SearchExecutionStatus.EXECUTED


# ─── 9 & 10. Source Coverage Updates & Strict Evidence/Gate Boundaries ────────


def test_candidate_urls_update_coverage_without_fabricating_evidence_or_resolving_gates(
    low_plan: OSINTInvestigationPlan,
) -> None:
    # Pre-mark S3 as REVIEWED_EVIDENCE_RECORDED by an analyst to verify collection never overwrites it
    tracker = OSINTCoverageTracker(low_plan.coverage_records)
    tracker.record_evidence_found(
        "S3",
        evidence_references=["https://fiserv.com/legal/terms"],
        candidate_urls=1,
        reviewed_urls=1,
        produced_new_evidence=True,
    )

    s1_queries = [q for q in low_plan.planned_queries if "S1" in q.source_classes]
    s2_queries = [q for q in low_plan.planned_queries if "S2" in q.source_classes and "S1" not in q.source_classes]

    responses: dict[str, Any] = {}
    for q in s1_queries:
        responses[q.rendered_query] = [
            {"url": "https://fiserv.com/trust/ai-policy", "title": "Fiserv AI Policy", "rank": 1}
        ]
    for q in s2_queries:
        responses[q.rendered_query] = []

    stub = StubSearchBackend(responses=responses, default_results=[], provider_name="mock_boundaries")
    collector = OSINTSearchCollector(backend=stub, retry_backoff_seconds=0.0)
    collector.execute_planned_queries(low_plan)

    # S1 found 1 distinct URL -> RESULTS_FOUND, but reviewed_urls_count == 0 and produced_new_evidence == False
    s1_rec = low_plan.coverage_records["S1"]
    assert s1_rec.status == CoverageState.RESULTS_FOUND
    assert s1_rec.candidate_urls_count == 1
    assert s1_rec.reviewed_urls_count == 0
    assert s1_rec.produced_new_evidence is False
    assert s1_rec.evidence_references == []

    # S2 had executed queries with 0 results -> SEARCH_ATTEMPTED
    s2_rec = low_plan.coverage_records["S2"]
    assert s2_rec.status == CoverageState.SEARCH_ATTEMPTED
    assert s2_rec.candidate_urls_count == 0
    assert s2_rec.reviewed_urls_count == 0
    assert s2_rec.produced_new_evidence is False

    # S3 preserved its analyst-reviewed state
    s3_rec = low_plan.coverage_records["S3"]
    assert s3_rec.status == CoverageState.REVIEWED_EVIDENCE_RECORDED
    assert s3_rec.reviewed_urls_count == 1
    assert s3_rec.produced_new_evidence is True
    assert s3_rec.evidence_references == ["https://fiserv.com/legal/terms"]

    # G1-G4 gates and stop_status must NOT be resolved by unreviewed search results
    assert all(g.status == GateStatus.NOT_INVESTIGATED for g in low_plan.gate_checklist)
    assert low_plan.stop_status == AssessmentStopStatus.IN_PROGRESS


# ─── 11. Persistence & Replanning Preservation ────────────────────────────────


def test_persistence_and_o5_replanning_preserve_executed_queries_and_candidate_urls(
    sample_vendor: MeridianVendor,
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "osint_state.json"
    planner = OSINTInvestigationPlanner()
    plan = planner.plan_investigation(sample_vendor, criticality_input="Low")

    target_q = plan.planned_queries[0]
    stub = StubSearchBackend(
        responses={
            target_q.rendered_query: [
                {
                    "url": "https://fiserv.com/trust/subprocessors",
                    "title": "Fiserv Subprocessors",
                    "snippet": "AWS Bedrock and Azure OpenAI subprocessor listing.",
                    "rank": 1,
                }
            ]
        },
        provider_name="mock_persist",
    )
    collector = OSINTSearchCollector(backend=stub, retry_backoff_seconds=0.0)
    collector.execute_planned_queries(plan, query_ids=[target_q.query_id], persist=True, state_path=state_file)

    # Reload from JSON state file
    loaded = OSINTInvestigationPlanner.load_plan_state(sample_vendor.vendor_id, state_path=state_file)
    assert loaded is not None
    loaded_q = next(q for q in loaded.planned_queries if q.query_id == target_q.query_id)
    assert loaded_q.execution_status == SearchExecutionStatus.EXECUTED
    assert loaded_q.provider == "mock_persist"
    assert loaded_q.result_count == 1
    assert len(loaded.candidate_urls) == 1
    assert loaded.candidate_urls[0].url == "https://fiserv.com/trust/subprocessors"
    assert loaded.last_collection_summary.get("queries_executed") == 1

    # Replan after O5 analyst override to Critical
    override = HumanReviewRecord(
        user_decision="CHANGE_CRITICALITY",
        original_criticality="Low",
        final_criticality="Critical",
        rationale="Escalated to Critical for full S1-S12 review.",
    )
    replanned = planner.plan_investigation(
        sample_vendor,
        criticality_input="Low",
        human_review=override,
        existing_plan=loaded,
    )
    assert replanned.selected_depth_profile == "Critical"
    assert len(replanned.candidate_urls) == 1
    assert replanned.candidate_urls[0].url == "https://fiserv.com/trust/subprocessors"
    preserved_q = next(
        q for q in replanned.planned_queries if q.rendered_query.lower() == target_q.rendered_query.lower()
    )
    assert preserved_q.execution_status == SearchExecutionStatus.EXECUTED
    assert preserved_q.result_count == 1


def test_rest_json_backends_parse_mocked_payloads_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    import meridian_assessment.services.osint.search_collector as sc_mod

    def fake_http_get(url: str, *, headers: Any = None, timeout_seconds: float = 10.0) -> dict[str, Any]:
        if "googleapis.com" in url:
            return {
                "items": [
                    {
                        "link": "https://fiserv.com/trust/dpa",
                        "title": "Fiserv DPA",
                        "snippet": "Google CSE result",
                    }
                ]
            }
        if "search.brave.com" in url:
            return {
                "web": {
                    "results": [
                        {
                            "url": "https://fiserv.com/docs/api",
                            "title": "Fiserv API",
                            "description": "Brave result",
                        }
                    ]
                }
            }
        if "bing.microsoft.com" in url:
            return {
                "webPages": {
                    "value": [
                        {
                            "url": "https://fiserv.com/security/whitepaper",
                            "name": "Fiserv Security Whitepaper",
                            "snippet": "Bing result",
                        }
                    ]
                }
            }
        return {
            "organic_results": [
                {
                    "link": "https://fiserv.com/ai/model-card",
                    "title": "Fiserv Model Card",
                    "snippet": "SerpAPI result",
                    "position": 1,
                }
            ]
        }

    monkeypatch.setattr(sc_mod, "_execute_json_http_get", fake_http_get)

    g_backend = build_search_backend_from_env(
        "google_cse",
        env={"MERIDIAN_GOOGLE_API_KEY": "test-key", "MERIDIAN_GOOGLE_CSE_ID": "test-cx"},
    )
    g_resp = g_backend.search('"Fiserv" DPA', max_results=5)
    assert g_resp.provider == "google_cse"
    assert len(g_resp.results) == 1
    assert g_resp.results[0].url == "https://fiserv.com/trust/dpa"

    brave_backend = build_search_backend_from_env(
        "brave",
        env={"MERIDIAN_BRAVE_API_KEY": "test-brave"},
    )
    b_resp = brave_backend.search('"Fiserv" API', max_results=5)
    assert b_resp.provider == "brave"
    assert len(b_resp.results) == 1
    assert b_resp.results[0].snippet == "Brave result"

    bing_backend = build_search_backend_from_env(
        "bing",
        env={"MERIDIAN_BING_API_KEY": "test-bing"},
    )
    m_resp = bing_backend.search('"Fiserv" whitepaper', max_results=5)
    assert m_resp.provider == "bing"
    assert len(m_resp.results) == 1
    assert m_resp.results[0].title == "Fiserv Security Whitepaper"

    serp_backend = build_search_backend_from_env(
        "serpapi",
        env={"MERIDIAN_SERPAPI_KEY": "test-serp"},
    )
    s_resp = serp_backend.search('"Fiserv" model card', max_results=5)
    assert s_resp.provider == "serpapi"
    assert len(s_resp.results) == 1
    assert s_resp.results[0].rank == 1

