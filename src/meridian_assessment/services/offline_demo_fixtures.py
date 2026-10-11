"""Offline synthetic test evidence fixtures for end-to-end 9-level walkthroughs.

IMPORTANT:
All candidate URLs and HTML/PDF payloads in this module are explicitly labelled
SYNTHETIC TEST FIXTURES for offline POC verification and stakeholder demos.
They are NEVER presented as live OSINT collection or real-world assertions
about the sample vendors.
"""

from __future__ import annotations

from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.services.extraction.extractor import (
    FetchResponse,
    StubFetchAdapter,
)
from meridian_assessment.services.osint.search_collector import (
    SearchBackendResponse,
    SearchResultItem,
    StubSearchBackend,
)


def build_synthetic_demo_adapters_for_vendor(
    vendor: MeridianVendor,
) -> tuple[StubSearchBackend, StubFetchAdapter]:
    """Return a clearly labelled (StubSearchBackend, StubFetchAdapter) pair for offline testing."""
    domain = (vendor.domain or f"{vendor.vendor_id.lower()}.example.com").strip().lower()
    service_name = (
        vendor.service_product_provided
        or vendor.business_process_supported
        or "Enterprise Platform"
    )
    vid = vendor.vendor_id.upper()

    if vid == "V-001":
        # AutomWorx: Deterministic cron/rule automation only (Rule E6 -> NONE_AUTOMATION_ONLY -> NO AI verdict)
        url_docs = f"https://docs.{domain}/admin-guide/batch-scheduler-architecture"
        url_trust = f"https://trust.{domain}/subprocessors"
        items = [
            SearchResultItem(
                url=url_docs,
                title=f"[SYNTHETIC TEST FIXTURE] {vendor.vendor_name} Batch Scheduler Admin Guide",
                snippet=f"Technical documentation for {service_name} deterministic batch scheduling and job orchestration.",
                rank=1,
            ),
            SearchResultItem(
                url=url_trust,
                title=f"[SYNTHETIC TEST FIXTURE] {vendor.vendor_name} Trust Center & Subprocessors",
                snippet="Authoritative list of infrastructure subprocessors. No AI/ML subprocessors used.",
                rank=2,
            ),
        ]
        fixtures = {
            url_docs: FetchResponse(
                url=url_docs,
                status_code=200,
                content_type="text/html",
                content_bytes=f"""
                <html><head><title>{vendor.vendor_name} Admin Guide</title></head>
                <body>
                  <h1>{service_name} Architecture</h1>
                  <p>{service_name} executes deterministic batch scheduling and job orchestration
                  using static rules engine workflows and cron schedules.</p>
                </body></html>
                """.encode("utf-8"),
            ),
            url_trust: FetchResponse(
                url=url_trust,
                status_code=200,
                content_type="text/html",
                content_bytes=f"""
                <html><head><title>{vendor.vendor_name} Subprocessor Register</title></head>
                <body>
                  <h1>Subprocessor Register</h1>
                  <p>Our {service_name} platform uses deterministic job orchestration only.</p>
                </body></html>
                """.encode("utf-8"),
            ),
        }
    elif vid in {"V-002", "V-005", "V-006"}:
        # Critical payment/core banking vendors: synthetic test fixture demonstrating S1 subprocessor + S2 API docs
        url_trust = f"https://trust.{domain}/subprocessors-and-ai-policy"
        url_docs = f"https://docs.{domain}/api/ai-copilot-data-flows"
        items = [
            SearchResultItem(
                url=url_trust,
                title=f"[SYNTHETIC TEST FIXTURE] {vendor.vendor_name} Subprocessor Register & DPA",
                snippet=f"Subprocessor disclosure for {service_name} including Microsoft Azure OpenAI GPT-4o.",
                rank=1,
            ),
            SearchResultItem(
                url=url_docs,
                title=f"[SYNTHETIC TEST FIXTURE] {vendor.vendor_name} Technical API Documentation",
                snippet=f"Architecture and inference pipeline documentation for {service_name} LLM completions.",
                rank=2,
            ),
        ]
        fixtures = {
            url_trust: FetchResponse(
                url=url_trust,
                status_code=200,
                content_type="text/html",
                content_bytes=f"""
                <html><head><title>{vendor.vendor_name} Trust Center — Subprocessors</title></head>
                <body>
                  <h1>Subprocessor Register & Data Processing Addendum</h1>
                  <p>For {service_name}, {vendor.vendor_name} engages Microsoft Azure OpenAI (GPT-4o)
                  hosted on Microsoft Azure as an AI subprocessor. Customer data and transaction records
                  submitted to the inference pipeline may be used to train and improve models unless opted out,
                  and prompt logs are retained for 30 days.</p>
                </body></html>
                """.encode("utf-8"),
            ),
            url_docs: FetchResponse(
                url=url_docs,
                status_code=200,
                content_type="text/html",
                content_bytes=f"""
                <html><head><title>{vendor.vendor_name} API Reference & Admin Guide</title></head>
                <body>
                  <h1>{service_name} — AI Inference Pipeline & Autonomous Agent</h1>
                  <p>The {service_name} production API endpoint invokes GPT-4o LLM completions and an
                  autonomous agentic workflow on customer data, account balances, and payment instructions.</p>
                </body></html>
                """.encode("utf-8"),
            ),
        }
    else:
        # V-003 (FSSI), V-004 (Terrapin): synthetic fixture showing contractual no-training & opt-in AI
        url_trust = f"https://trust.{domain}/subprocessors"
        url_docs = f"https://docs.{domain}/release-notes/2025"
        items = [
            SearchResultItem(
                url=url_trust,
                title=f"[SYNTHETIC TEST FIXTURE] {vendor.vendor_name} Subprocessor List & DPA",
                snippet=f"Subprocessor register listing AWS Bedrock (Claude 3) for {service_name}.",
                rank=1,
            ),
            SearchResultItem(
                url=url_docs,
                title=f"[SYNTHETIC TEST FIXTURE] {vendor.vendor_name} Product Documentation",
                snippet=f"Admin guide for {service_name} optional opt-in document summarization.",
                rank=2,
            ),
        ]
        fixtures = {
            url_trust: FetchResponse(
                url=url_trust,
                status_code=200,
                content_type="text/html",
                content_bytes=f"""
                <html><head><title>{vendor.vendor_name} Subprocessors</title></head>
                <body>
                  <h1>Third-Party Subprocessors</h1>
                  <p>For {service_name}, {vendor.vendor_name} utilizes AWS Bedrock (Claude 3) hosted on AWS.
                  Under our Data Processing Addendum (DPA), customer data is never used for model training
                  and zero data retention (ZDR) is enforced by default.</p>
                </body></html>
                """.encode("utf-8"),
            ),
            url_docs: FetchResponse(
                url=url_docs,
                status_code=200,
                content_type="text/html",
                content_bytes=f"""
                <html><head><title>{vendor.vendor_name} Documentation</title></head>
                <body>
                  <h1>{service_name} Administrator Guide</h1>
                  <p>In {service_name}, the Claude 3 LLM summarization API endpoint is disabled by default
                  and requires explicit admin opt-in activation.</p>
                </body></html>
                """.encode("utf-8"),
            ),
        }

    return (
        StubSearchBackend(default_results=items, provider_name="synthetic_offline_fixture"),
        StubFetchAdapter(fixtures=fixtures),
    )
