"""Level 3B — Candidate URL Triage Service for the Meridian OSINT Pipeline.

Assesses each CandidateURL to determine:
  - Whether the URL is associated with the target vendor or an authoritative external registry.
  - Whether it appears relevant to the vendor's product or the specific service Meridian purchases.
  - Which investigative gates (G1–G4) and evidence questions it may help answer.
  - Which canonical source class (S1–S12) best matches the URL.
  - Whether the URL should be fetched (`FETCH`), deferred (`DEFER`), rejected (`REJECT`), or
    routed to manual review (`MANUAL_REVIEW`).

Provides a clean `URLTriageLLMProvider` interface for future local-LLM triage while using
transparent deterministic heuristics labelled `DETERMINISTIC_HEURISTIC` when no LLM is configured.
Passing triage never marks a candidate URL as verified evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import re
from typing import Optional, Sequence
import urllib.parse

from meridian_assessment.models.ai_risk import (
    TriageDecisionMethod,
    URLTriageAction,
    URLTriageDecision,
)
from meridian_assessment.models.osint_plan import (
    CandidateURL,
    OSINTInvestigationPlan,
    URLTriageStatus,
)
from meridian_assessment.services.osint.query_planner import OSINTQueryPlanner


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Recognized third-party discovery hosts that legitimately host vendor OSINT outside `{domain}`
_KNOWN_EXTERNAL_DISCOVERY_HOSTS: dict[str, tuple[str, str]] = {
    "github.com": ("S2", "Public code repository / SDK host"),
    "gitlab.com": ("S2", "Public code repository host"),
    "pypi.org": ("S2", "Python package registry"),
    "npmjs.com": ("S2", "Node.js package registry"),
    " central.sonatype.com": ("S2", "Maven package registry"),
    "mvnrepository.com": ("S2", "Maven repository index"),
    "hub.docker.com": ("S2", "Container image registry"),
    "huggingface.co": ("S4", "Model & dataset card hub"),
    "arxiv.org": ("S10", "Research paper preprint repository"),
    "patents.google.com": ("S10", "Patent registry"),
    "sec.gov": ("S5", "SEC EDGAR regulatory filings"),
    "linkedin.com": ("S6", "Technical job postings / hiring signals"),
    "indeed.com": ("S6", "Technical job postings"),
    "boards.greenhouse.io": ("S6", "Greenhouse ATS job board"),
    "jobs.lever.co": ("S6", "Lever ATS job board"),
    "aws.amazon.com": ("S8", "AWS Marketplace / partner listing"),
    "azuremarketplace.microsoft.com": ("S8", "Azure Marketplace listing"),
    "cloud.google.com": ("S8", "Google Cloud Marketplace / partner listing"),
    "appexchange.salesforce.com": ("S8", "Salesforce AppExchange listing"),
    "stackoverflow.com": ("S9", "Developer forum technical Q&A"),
    "reddit.com": ("S9", "Community technical discussion"),
    "youtube.com": ("S4", "Technical conference talk / product demo video"),
    "web.archive.org": ("S1", "Wayback Machine historical archive"),
}


class URLTriageLLMProvider(ABC):
    """Interface for optional local-LLM URL triage assistance."""

    @abstractmethod
    def is_configured(self) -> tuple[bool, str]:
        """Return `(True, info)` if the local LLM triage provider is configured and ready."""

    @abstractmethod
    def evaluate_candidate(
        self,
        candidate: CandidateURL,
        *,
        vendor_name: str,
        domain: str,
        service_scope: Optional[str] = None,
    ) -> Optional[URLTriageDecision]:
        """Return a structured `URLTriageDecision` or `None` if unavailable."""


class UnconfiguredURLTriageLLMProvider(URLTriageLLMProvider):
    """Default unconfigured LLM triage provider (falls back to deterministic heuristics)."""

    def is_configured(self) -> tuple[bool, str]:
        return False, "Local LLM triage provider is not configured; using deterministic URL triage."

    def evaluate_candidate(
        self,
        candidate: CandidateURL,
        *,
        vendor_name: str,
        domain: str,
        service_scope: Optional[str] = None,
    ) -> Optional[URLTriageDecision]:
        return None


class OSINTURLTriageService:
    """Evaluates CandidateURL records and assigns triage actions and rationale."""

    def __init__(self, llm_provider: Optional[URLTriageLLMProvider] = None) -> None:
        self.llm_provider: URLTriageLLMProvider = llm_provider or UnconfiguredURLTriageLLMProvider()

    def triage_plan_candidates(
        self,
        plan: OSINTInvestigationPlan,
    ) -> list[URLTriageDecision]:
        """Triage all CandidateURL items on an OSINTInvestigationPlan and update their triage_status."""
        decisions: list[URLTriageDecision] = []
        for cand in plan.candidate_urls:
            decision = self.triage_candidate(
                cand,
                vendor_name=plan.vendor_name,
                domain=plan.domain,
                service_scope=plan.service_product_provided,
                business_process=plan.business_process_supported,
            )
            decisions.append(decision)
        return decisions

    def triage_candidate(
        self,
        candidate: CandidateURL,
        *,
        vendor_name: str,
        domain: str,
        service_scope: Optional[str] = None,
        business_process: Optional[str] = None,
    ) -> URLTriageDecision:
        """Evaluate a single CandidateURL using optional LLM provider or deterministic heuristics."""
        llm_ok, _ = self.llm_provider.is_configured()
        if llm_ok:
            llm_decision = self.llm_provider.evaluate_candidate(
                candidate,
                vendor_name=vendor_name,
                domain=domain,
                service_scope=service_scope,
            )
            if llm_decision is not None:
                self._apply_decision_to_candidate(candidate, llm_decision)
                return llm_decision

        decision = self._deterministic_triage(
            candidate,
            vendor_name=vendor_name,
            domain=domain,
            service_scope=service_scope,
            business_process=business_process,
        )
        self._apply_decision_to_candidate(candidate, decision)
        return decision

    @staticmethod
    def _apply_decision_to_candidate(candidate: CandidateURL, decision: URLTriageDecision) -> None:
        if decision.action == URLTriageAction.FETCH:
            candidate.triage_status = URLTriageStatus.RELEVANT
        elif decision.action == URLTriageAction.REJECT:
            candidate.triage_status = URLTriageStatus.IRRELEVANT
        else:
            # DEFER or MANUAL_REVIEW stays PENDING review/fetch
            candidate.triage_status = URLTriageStatus.PENDING
        candidate.notes = f"[{decision.action.value} via {decision.decision_method.value}] {decision.rationale}"

    def _deterministic_triage(
        self,
        candidate: CandidateURL,
        *,
        vendor_name: str,
        domain: str,
        service_scope: Optional[str] = None,
        business_process: Optional[str] = None,
    ) -> URLTriageDecision:
        url_str = candidate.url or ""
        parsed = urllib.parse.urlsplit(url_str)
        host = (parsed.hostname or "").lower()
        path_low = (parsed.path or "/").lower()
        combined_text = " ".join(
            filter(None, [url_str, candidate.title or "", candidate.snippet or ""])
        ).lower()

        clean_v = OSINTQueryPlanner.clean_vendor_name(vendor_name).lower()
        clean_d = OSINTQueryPlanner.clean_domain(domain).lower()

        # 1. Check for gated login / portal / auth walls -> MANUAL_REVIEW
        if any(
            seg in path_low
            for seg in ("/login", "/signin", "/sign-in", "/auth/", "/sso/", "/account/login")
        ) or any(
            phrase in combined_text
            for phrase in ("sign in to view", "login required", "customer portal login", "request access to trust center")
        ):
            src_cls = self._infer_source_class(candidate, host, path_low, combined_text)
            return URLTriageDecision(
                candidate_id=candidate.candidate_id,
                vendor_id=candidate.vendor_id,
                url=url_str,
                action=URLTriageAction.MANUAL_REVIEW,
                triage_status=URLTriageStatus.PENDING.value,
                vendor_match=True,
                product_scope_relevance="UNRESOLVED",
                candidate_source_class=src_cls,
                target_gates=self._infer_target_gates(src_cls, combined_text),
                evidence_questions_addressed=self._infer_questions(src_cls, combined_text),
                rationale="URL appears to require authentication or portal access; routed to manual analyst review.",
                decision_method=TriageDecisionMethod.DETERMINISTIC_HEURISTIC,
                confidence="HIGH",
                triaged_at=_utc_now_iso(),
            )

        # 2. Check vendor association
        is_vendor_domain = bool(clean_d and (host == clean_d or host.endswith("." + clean_d)))
        vendor_tokens = [tok for tok in re.split(r"\W+", clean_v) if len(tok) >= 3]
        mentions_vendor = bool(
            (clean_v and clean_v in combined_text)
            or (vendor_tokens and any(tok in combined_text for tok in vendor_tokens))
        )
        external_host_info = None
        for ext_host, info in _KNOWN_EXTERNAL_DISCOVERY_HOSTS.items():
            eh = ext_host.strip()
            if host == eh or host.endswith("." + eh):
                external_host_info = info
                break

        vendor_match = is_vendor_domain or mentions_vendor

        if not vendor_match:
            return URLTriageDecision(
                candidate_id=candidate.candidate_id,
                vendor_id=candidate.vendor_id,
                url=url_str,
                action=URLTriageAction.REJECT,
                triage_status=URLTriageStatus.IRRELEVANT.value,
                vendor_match=False,
                product_scope_relevance="UNRELATED",
                candidate_source_class=candidate.source_class or "S12",
                target_gates=[],
                evidence_questions_addressed=[],
                rationale=(
                    f"URL host '{host}' does not match vendor domain '{clean_d or 'N/A'}' "
                    f"and title/snippet does not reference vendor '{vendor_name}'."
                ),
                decision_method=TriageDecisionMethod.DETERMINISTIC_HEURISTIC,
                confidence="HIGH",
                triaged_at=_utc_now_iso(),
            )

        # 3. Infer source class and scope relevance
        src_cls = self._infer_source_class(candidate, host, path_low, combined_text, external_host_info)
        scope_rel = self._assess_scope_relevance(
            combined_text,
            service_scope=service_scope,
            business_process=business_process,
        )
        gates = self._infer_target_gates(src_cls, combined_text)
        questions = self._infer_questions(src_cls, combined_text)

        # 4. Decide FETCH vs DEFER
        has_ai_or_governance_signal = any(
            kw in combined_text
            for kw in (
                "ai",
                "artificial intelligence",
                "machine learning",
                "llm",
                "genai",
                "generative",
                "copilot",
                "openai",
                "anthropic",
                "bedrock",
                "vertex",
                "subprocessor",
                "dpa",
                "data processing",
                "privacy",
                "trust",
                "security",
                "whitepaper",
                "retention",
                "training",
                "model",
                "api",
                "sdk",
                "release notes",
                "documentation",
            )
        )

        if src_cls == "S12" and not has_ai_or_governance_signal:
            return URLTriageDecision(
                candidate_id=candidate.candidate_id,
                vendor_id=candidate.vendor_id,
                url=url_str,
                action=URLTriageAction.DEFER,
                triage_status=URLTriageStatus.PENDING.value,
                vendor_match=True,
                product_scope_relevance=scope_rel,
                candidate_source_class=src_cls,
                target_gates=gates,
                evidence_questions_addressed=questions,
                rationale="Low-priority marketing page (S12) without technical AI or governance indicators; deferred.",
                decision_method=TriageDecisionMethod.DETERMINISTIC_HEURISTIC,
                confidence="MEDIUM",
                triaged_at=_utc_now_iso(),
            )

        return URLTriageDecision(
            candidate_id=candidate.candidate_id,
            vendor_id=candidate.vendor_id,
            url=url_str,
            action=URLTriageAction.FETCH,
            triage_status=URLTriageStatus.RELEVANT.value,
            vendor_match=True,
            product_scope_relevance=scope_rel,
            candidate_source_class=src_cls,
            target_gates=gates,
            evidence_questions_addressed=questions,
            rationale=(
                f"Matched vendor '{vendor_name}' ({'official domain' if is_vendor_domain else 'external OSINT source'}) "
                f"for source class {src_cls} with scope '{scope_rel}'; selected for extraction."
            ),
            decision_method=TriageDecisionMethod.DETERMINISTIC_HEURISTIC,
            confidence="HIGH" if is_vendor_domain else "MEDIUM",
            triaged_at=_utc_now_iso(),
        )

    @staticmethod
    def _infer_source_class(
        candidate: CandidateURL,
        host: str,
        path_low: str,
        combined_text: str,
        external_host_info: Optional[tuple[str, str]] = None,
    ) -> str:
        if any(k in path_low or k in combined_text for k in ("subprocessor", "trust-center", "/trust", "/dpa", "data-processing-addendum", "ai-policy")):
            return "S1"
        if any(k in path_low for k in ("/docs", "/api", "/developer", "/sdk", "release-notes", "changelog")):
            return "S2"
        if any(k in path_low or k in combined_text for k in ("/privacy", "/terms", "/legal", "security-whitepaper", "whitepaper")):
            return "S3"
        if any(k in path_low for k in ("/blog", "/engineering", "model-card")):
            return "S4"
        if any(k in path_low or k in combined_text for k in ("10-k", "annual-report", "soc-2", "iso-42001", "iso-27001")):
            return "S5"
        if any(k in path_low for k in ("/careers", "/jobs")):
            return "S6"
        if any(k in path_low for k in ("/investors", "investor-relations", "earnings")):
            return "S7"
        if external_host_info is not None:
            return external_host_info[0]
        if candidate.source_classes:
            return candidate.source_classes[0]
        if candidate.source_class:
            return candidate.source_class
        return "S2"

    @staticmethod
    def _assess_scope_relevance(
        combined_text: str,
        *,
        service_scope: Optional[str] = None,
        business_process: Optional[str] = None,
    ) -> str:
        scope_terms: list[str] = []
        for raw in (service_scope, business_process):
            if raw:
                for tok in re.split(r"\W+", raw.lower()):
                    if len(tok) >= 4 and tok not in ("with", "from", "that", "this", "data", "services", "service"):
                        scope_terms.append(tok)
        if scope_terms and any(t in combined_text for t in scope_terms):
            return "MERIDIAN_SERVICE"
        if any(k in combined_text for k in ("platform", "api", "documentation", "release", "product", "guide")):
            return "VENDOR_PRODUCT"
        return "VENDOR_FIRM_WIDE"

    @staticmethod
    def _infer_target_gates(source_class: str, combined_text: str) -> list[str]:
        gates: list[str] = []
        if any(k in combined_text for k in ("ai", "machine learning", "llm", "model", "copilot", "inference", "api")):
            gates.append("G1")
            gates.append("G2")
        if source_class in ("S1", "S3") or any(
            k in combined_text for k in ("privacy", "dpa", "retention", "training", "customer data", "pii")
        ):
            gates.append("G3")
        if source_class in ("S1", "S2", "S8") or any(
            k in combined_text for k in ("subprocessor", "openai", "anthropic", "bedrock", "vertex", "azure", "aws")
        ):
            gates.append("G4")
        return sorted(list(set(gates))) or ["G1"]

    @staticmethod
    def _infer_questions(source_class: str, combined_text: str) -> list[str]:
        questions: list[str] = []
        if source_class == "S1":
            questions.append("Which AI subprocessors and data-processing commitments are contractually disclosed?")
        if source_class == "S2":
            questions.append("Does technical product/API documentation confirm operational AI features in scope?")
        if source_class == "S3":
            questions.append("Do terms or privacy notices permit customer-data model training or define retention windows?")
        if not questions:
            questions.append(f"What evidence does {source_class} provide regarding operational AI use and risk?")
        return questions
