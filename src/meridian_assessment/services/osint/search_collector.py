"""Bounded search execution and candidate-URL collection for the Meridian OSINT Planner (Phase 1).

Executes `PLANNED` OSINT queries against an explicitly configured search backend,
records per-query execution metadata, conservatively canonicalizes and deduplicates
candidate URLs while preserving full multi-query and multi-source-class provenance,
and enforces strict boundaries between candidate URL collection and reviewed evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request

from pydantic import BaseModel, Field
import yaml

from meridian_assessment.models.osint_plan import (
    CandidateURL,
    CoverageState,
    OSINTInvestigationPlan,
    PlannedQuery,
    SearchExecutionStatus,
    URLTriageStatus,
)
from meridian_assessment.services.osint.investigation_planner import (
    DEFAULT_OSINT_STATE_PATH,
    OSINTInvestigationPlanner,
)

_DEFAULT_SEARCH_CONFIG_PATH = Path("config/osint_search.yaml")

_DEFAULT_TRACKING_PARAMS = frozenset({
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
})


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ─── Exceptions ───────────────────────────────────────────────────────────────


class SearchExecutionError(Exception):
    """Base exception for search execution failures."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class SearchConfigurationError(SearchExecutionError):
    """Raised when no search provider or required API credential is configured."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class SearchTimeoutError(SearchExecutionError):
    """Raised when a search provider request times out."""

    def __init__(self, message: str = "Search request timed out.") -> None:
        super().__init__(message, retryable=True)


class SearchRateLimitError(SearchExecutionError):
    """Raised when a search provider returns HTTP 429 / rate limit exceeded."""

    def __init__(self, message: str = "Search provider rate limit exceeded (HTTP 429).") -> None:
        super().__init__(message, retryable=True)


class SearchProviderError(SearchExecutionError):
    """Raised when a search provider returns an upstream error or malformed response."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)


# ─── Data Structures ──────────────────────────────────────────────────────────


class SearchResultItem(BaseModel):
    """Normalized single result item returned by a search backend."""

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    rank: Optional[int] = None
    retrieved_at: Optional[str] = None


class SearchBackendResponse(BaseModel):
    """Structured response from a single query execution against a search backend."""

    provider: str
    executed_query: str
    executed_at: str = Field(default_factory=_utc_now_iso)
    results: list[Any] = Field(default_factory=list)


# ─── Conservative URL Canonicalization ────────────────────────────────────────


def canonicalize_url(
    raw_url: Optional[str],
    *,
    strip_tracking_params: Optional[set[str] | frozenset[str]] = None,
) -> Optional[str]:
    """Canonicalize a candidate URL conservatively without discarding meaningful paths or query parameters.

    Rules:
      - Requires explicit `http` or `https` scheme and a valid network host.
      - Lowercases scheme and hostname; strips default port (`:80` for http, `:443` for https).
      - Strips URL fragments (`#...`).
      - Preserves path case and structure (normalizes empty path `""` to `"/"`, and strips trailing
        slash on non-root paths only when safe, e.g. `/trust/` -> `/trust`).
      - Preserves all query parameters and their values in order, removing only known analytics
        tracking parameters (`utm_*`, `gclid`, `fbclid`, etc.).
    """
    if not raw_url or not isinstance(raw_url, str):
        return None

    cleaned = raw_url.strip()
    if not cleaned:
        return None

    try:
        parsed = urllib.parse.urlsplit(cleaned)
    except Exception:
        return None

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return None

    hostname = (parsed.hostname or "").lower()
    if not hostname or "." not in hostname and hostname != "localhost":
        return None

    try:
        port = parsed.port
    except ValueError:
        return None

    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    tracking_to_strip = (
        _DEFAULT_TRACKING_PARAMS
        if strip_tracking_params is None
        else frozenset(p.lower() for p in strip_tracking_params)
    )

    if parsed.query:
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        filtered_pairs = [
            (k, v) for (k, v) in query_pairs if k.lower() not in tracking_to_strip
        ]
        query_str = urllib.parse.urlencode(filtered_pairs, doseq=True)
    else:
        query_str = ""

    return urllib.parse.urlunsplit((scheme, netloc, path, query_str, ""))


# ─── Search Backend Abstraction & Implementations ─────────────────────────────


class SearchBackend(ABC):
    """Abstract interface for bounded OSINT search backends."""

    provider_name: str = "abstract"

    @abstractmethod
    def is_configured(self) -> tuple[bool, str]:
        """Return `(True, status_message)` if ready to execute queries, else `(False, reason)`."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        timeout_seconds: float = 10.0,
    ) -> SearchBackendResponse:
        """Execute a single search query and return raw/normalized candidate results."""


class UnconfiguredSearchBackend(SearchBackend):
    """Fallback backend used when no search provider or credentials are configured."""

    def __init__(
        self,
        reason: str = (
            "No search provider credentials configured. Set MERIDIAN_SEARCH_PROVIDER "
            "and the corresponding API credentials (e.g. MERIDIAN_GOOGLE_API_KEY + "
            "MERIDIAN_GOOGLE_CSE_ID, MERIDIAN_BRAVE_API_KEY, MERIDIAN_BING_API_KEY, "
            "or MERIDIAN_SERPAPI_KEY)."
        ),
        provider_name: str = "unconfigured",
    ) -> None:
        self.provider_name = provider_name
        self.reason = reason

    def is_configured(self) -> tuple[bool, str]:
        return False, self.reason

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        timeout_seconds: float = 10.0,
    ) -> SearchBackendResponse:
        raise SearchConfigurationError(self.reason)


class StubSearchBackend(SearchBackend):
    """Offline/mock search backend for deterministic testing without network calls."""

    def __init__(
        self,
        responses: Optional[dict[str, Any] | Callable[[str, int, float], Any]] = None,
        *,
        default_results: Optional[Sequence[Any]] = None,
        provider_name: str = "stub_search",
        configured: bool = True,
        unconfigured_reason: str = "Stub search provider is marked unconfigured.",
    ) -> None:
        self.provider_name = provider_name
        self._responses = responses
        self._default_results = list(default_results) if default_results is not None else []
        self._configured = configured
        self._unconfigured_reason = unconfigured_reason
        self.call_history: list[dict[str, Any]] = []

    def is_configured(self) -> tuple[bool, str]:
        if not self._configured:
            return False, self._unconfigured_reason
        return True, f"Configured offline stub backend ({self.provider_name})."

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        timeout_seconds: float = 10.0,
    ) -> SearchBackendResponse:
        self.call_history.append({
            "query": query,
            "max_results": max_results,
            "timeout_seconds": timeout_seconds,
            "timestamp": _utc_now_iso(),
        })
        if not self._configured:
            raise SearchConfigurationError(self._unconfigured_reason)

        if callable(self._responses):
            outcome = self._responses(query, max_results, timeout_seconds)
        elif isinstance(self._responses, dict):
            outcome = self._responses.get(query, self._default_results)
        else:
            outcome = list(self._default_results)

        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, SearchBackendResponse):
            return outcome
        if not isinstance(outcome, list):
            raise SearchProviderError(f"Malformed provider payload of type {type(outcome).__name__}.")

        return SearchBackendResponse(
            provider=self.provider_name,
            executed_query=query,
            executed_at=_utc_now_iso(),
            results=outcome,
        )


def _execute_json_http_get(
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Perform a bounded HTTP GET returning parsed JSON or raising typed SearchExecutionError."""
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise SearchRateLimitError(f"HTTP 429 rate limit from search provider: {exc.reason}") from exc
        if exc.code in (401, 403):
            raise SearchConfigurationError(
                f"Search provider authentication/authorization failed (HTTP {exc.code}: {exc.reason})."
            ) from exc
        if 500 <= exc.code < 600:
            raise SearchProviderError(
                f"Search provider server error (HTTP {exc.code}: {exc.reason}).",
                retryable=True,
            ) from exc
        raise SearchProviderError(
            f"Search provider request failed (HTTP {exc.code}: {exc.reason}).",
            retryable=False,
        ) from exc
    except TimeoutError as exc:
        raise SearchTimeoutError(f"Search request timed out after {timeout_seconds}s.") from exc
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), TimeoutError) or "timed out" in str(exc).lower():
            raise SearchTimeoutError(f"Search request timed out after {timeout_seconds}s.") from exc
        raise SearchProviderError(f"Network error contacting search provider: {exc.reason}", retryable=True) from exc

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise SearchProviderError(f"Malformed JSON response from search provider: {exc}", retryable=False) from exc

    if not isinstance(data, dict):
        raise SearchProviderError("Search provider returned non-object JSON payload.", retryable=False)
    return data


class GoogleCSESearchBackend(SearchBackend):
    """Google Custom Search JSON API backend."""

    provider_name = "google_cse"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        endpoint: str = "https://www.googleapis.com/customsearch/v1",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.cx = (cx or "").strip()
        self.endpoint = endpoint

    def is_configured(self) -> tuple[bool, str]:
        if not self.api_key or not self.cx:
            return (
                False,
                "Google Custom Search requires MERIDIAN_GOOGLE_API_KEY and MERIDIAN_GOOGLE_CSE_ID.",
            )
        return True, "Google Custom Search JSON API configured."

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        timeout_seconds: float = 10.0,
    ) -> SearchBackendResponse:
        ok, reason = self.is_configured()
        if not ok:
            raise SearchConfigurationError(reason)

        num = max(1, min(int(max_results), 10))
        params = urllib.parse.urlencode({"key": self.api_key, "cx": self.cx, "q": query, "num": num})
        payload = _execute_json_http_get(
            f"{self.endpoint}?{params}",
            headers={"Accept": "application/json"},
            timeout_seconds=timeout_seconds,
        )
        now_iso = _utc_now_iso()
        raw_items = payload.get("items", [])
        if raw_items is None:
            raw_items = []
        if not isinstance(raw_items, list):
            raise SearchProviderError("Malformed 'items' field in Google CSE response.")

        results: list[SearchResultItem] = []
        for idx, item in enumerate(raw_items[:max_results], start=1):
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            if not link or not isinstance(link, str):
                continue
            results.append(
                SearchResultItem(
                    url=link,
                    title=item.get("title"),
                    snippet=item.get("snippet"),
                    rank=idx,
                    retrieved_at=now_iso,
                )
            )
        return SearchBackendResponse(
            provider=self.provider_name,
            executed_query=query,
            executed_at=now_iso,
            results=results,
        )


class BraveSearchBackend(SearchBackend):
    """Brave Web Search REST API backend."""

    provider_name = "brave"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.search.brave.com/res/v1/web/search",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.endpoint = endpoint

    def is_configured(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Brave Search requires MERIDIAN_BRAVE_API_KEY."
        return True, "Brave Web Search API configured."

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        timeout_seconds: float = 10.0,
    ) -> SearchBackendResponse:
        ok, reason = self.is_configured()
        if not ok:
            raise SearchConfigurationError(reason)

        count = max(1, min(int(max_results), 20))
        params = urllib.parse.urlencode({"q": query, "count": count})
        payload = _execute_json_http_get(
            f"{self.endpoint}?{params}",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            timeout_seconds=timeout_seconds,
        )
        now_iso = _utc_now_iso()
        web_block = payload.get("web", {})
        if not isinstance(web_block, dict):
            raise SearchProviderError("Malformed 'web' field in Brave Search response.")
        raw_items = web_block.get("results", [])
        if not isinstance(raw_items, list):
            raise SearchProviderError("Malformed 'web.results' field in Brave Search response.")

        results: list[SearchResultItem] = []
        for idx, item in enumerate(raw_items[:max_results], start=1):
            if not isinstance(item, dict):
                continue
            link = item.get("url")
            if not link or not isinstance(link, str):
                continue
            results.append(
                SearchResultItem(
                    url=link,
                    title=item.get("title"),
                    snippet=item.get("description") or item.get("snippet"),
                    rank=idx,
                    retrieved_at=now_iso,
                )
            )
        return SearchBackendResponse(
            provider=self.provider_name,
            executed_query=query,
            executed_at=now_iso,
            results=results,
        )


class BingSearchBackend(SearchBackend):
    """Bing Web Search v7 REST API backend."""

    provider_name = "bing"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.bing.microsoft.com/v7.0/search",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.endpoint = endpoint

    def is_configured(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Bing Web Search requires MERIDIAN_BING_API_KEY."
        return True, "Bing Web Search API configured."

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        timeout_seconds: float = 10.0,
    ) -> SearchBackendResponse:
        ok, reason = self.is_configured()
        if not ok:
            raise SearchConfigurationError(reason)

        count = max(1, min(int(max_results), 50))
        params = urllib.parse.urlencode({"q": query, "count": count})
        payload = _execute_json_http_get(
            f"{self.endpoint}?{params}",
            headers={
                "Accept": "application/json",
                "Ocp-Apim-Subscription-Key": self.api_key,
            },
            timeout_seconds=timeout_seconds,
        )
        now_iso = _utc_now_iso()
        web_pages = payload.get("webPages", {})
        if not isinstance(web_pages, dict):
            raise SearchProviderError("Malformed 'webPages' field in Bing Search response.")
        raw_items = web_pages.get("value", [])
        if not isinstance(raw_items, list):
            raise SearchProviderError("Malformed 'webPages.value' field in Bing Search response.")

        results: list[SearchResultItem] = []
        for idx, item in enumerate(raw_items[:max_results], start=1):
            if not isinstance(item, dict):
                continue
            link = item.get("url")
            if not link or not isinstance(link, str):
                continue
            results.append(
                SearchResultItem(
                    url=link,
                    title=item.get("name") or item.get("title"),
                    snippet=item.get("snippet"),
                    rank=idx,
                    retrieved_at=now_iso,
                )
            )
        return SearchBackendResponse(
            provider=self.provider_name,
            executed_query=query,
            executed_at=now_iso,
            results=results,
        )


class SerpAPISearchBackend(SearchBackend):
    """SerpAPI Google Search JSON backend."""

    provider_name = "serpapi"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://serpapi.com/search.json",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.endpoint = endpoint

    def is_configured(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "SerpAPI requires MERIDIAN_SERPAPI_KEY."
        return True, "SerpAPI Search configured."

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        timeout_seconds: float = 10.0,
    ) -> SearchBackendResponse:
        ok, reason = self.is_configured()
        if not ok:
            raise SearchConfigurationError(reason)

        num = max(1, min(int(max_results), 20))
        params = urllib.parse.urlencode({
            "engine": "google",
            "q": query,
            "num": num,
            "api_key": self.api_key,
        })
        payload = _execute_json_http_get(
            f"{self.endpoint}?{params}",
            headers={"Accept": "application/json"},
            timeout_seconds=timeout_seconds,
        )
        now_iso = _utc_now_iso()
        raw_items = payload.get("organic_results", [])
        if not isinstance(raw_items, list):
            raise SearchProviderError("Malformed 'organic_results' field in SerpAPI response.")

        results: list[SearchResultItem] = []
        for idx, item in enumerate(raw_items[:max_results], start=1):
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            if not link or not isinstance(link, str):
                continue
            rank_val = item.get("position")
            rank_int = int(rank_val) if isinstance(rank_val, int) else idx
            results.append(
                SearchResultItem(
                    url=link,
                    title=item.get("title"),
                    snippet=item.get("snippet"),
                    rank=rank_int,
                    retrieved_at=now_iso,
                )
            )
        return SearchBackendResponse(
            provider=self.provider_name,
            executed_query=query,
            executed_at=now_iso,
            results=results,
        )


def build_search_backend_from_env(
    provider: Optional[str] = None,
    *,
    env: Optional[dict[str, str]] = None,
    config_path: Path = _DEFAULT_SEARCH_CONFIG_PATH,
) -> SearchBackend:
    """Resolve and instantiate a SearchBackend from environment variables and config."""
    environ = os.environ if env is None else env
    selected = (provider or environ.get("MERIDIAN_SEARCH_PROVIDER", "")).strip().lower()

    # Auto-detect if MERIDIAN_SEARCH_PROVIDER was not explicitly set
    if not selected:
        if environ.get("MERIDIAN_GOOGLE_API_KEY") and environ.get("MERIDIAN_GOOGLE_CSE_ID"):
            selected = "google_cse"
        elif environ.get("MERIDIAN_BRAVE_API_KEY"):
            selected = "brave"
        elif environ.get("MERIDIAN_BING_API_KEY"):
            selected = "bing"
        elif environ.get("MERIDIAN_SERPAPI_KEY"):
            selected = "serpapi"

    if selected in ("google_cse", "google"):
        return GoogleCSESearchBackend(
            api_key=environ.get("MERIDIAN_GOOGLE_API_KEY"),
            cx=environ.get("MERIDIAN_GOOGLE_CSE_ID"),
        )
    if selected == "brave":
        return BraveSearchBackend(api_key=environ.get("MERIDIAN_BRAVE_API_KEY"))
    if selected == "bing":
        return BingSearchBackend(api_key=environ.get("MERIDIAN_BING_API_KEY"))
    if selected == "serpapi":
        return SerpAPISearchBackend(api_key=environ.get("MERIDIAN_SERPAPI_KEY"))

    if selected and selected not in ("none", "unconfigured"):
        return UnconfiguredSearchBackend(
            reason=f"Unsupported or unconfigured search provider '{selected}'.",
            provider_name=selected,
        )

    return UnconfiguredSearchBackend()


# ─── Bounded Search Collector ─────────────────────────────────────────────────


class OSINTSearchCollector:
    """Executes planned OSINT search queries, collects & deduplicates candidate URLs, and updates coverage state."""

    def __init__(
        self,
        backend: Optional[SearchBackend] = None,
        *,
        config_path: Path = _DEFAULT_SEARCH_CONFIG_PATH,
        timeout_seconds: Optional[float] = None,
        max_results_per_query: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff_seconds: Optional[float] = None,
        max_queries_per_run: Optional[int] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config_path = config_path
        self._cfg = self._load_config(config_path)
        exec_cfg = self._cfg.get("execution", {})

        self.backend: SearchBackend = backend or build_search_backend_from_env(config_path=config_path)
        self.timeout_seconds: float = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(exec_cfg.get("timeout_seconds", 10.0))
        )
        self.max_results_per_query: int = (
            int(max_results_per_query)
            if max_results_per_query is not None
            else int(exec_cfg.get("max_results_per_query", 10))
        )
        self.max_retries: int = (
            int(max_retries)
            if max_retries is not None
            else int(exec_cfg.get("max_retries", 2))
        )
        self.retry_backoff_seconds: float = (
            float(retry_backoff_seconds)
            if retry_backoff_seconds is not None
            else float(exec_cfg.get("retry_backoff_seconds", 0.5))
        )
        self.max_queries_per_run: int = (
            int(max_queries_per_run)
            if max_queries_per_run is not None
            else int(exec_cfg.get("max_queries_per_run", 50))
        )
        self._sleep_fn = sleep_fn

        canon_cfg = self._cfg.get("canonicalization", {})
        raw_strip = canon_cfg.get("strip_tracking_params")
        self._strip_tracking_params: frozenset[str] = (
            frozenset(str(p).lower() for p in raw_strip)
            if isinstance(raw_strip, list)
            else _DEFAULT_TRACKING_PARAMS
        )

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}

    def _execute_single_query_with_retry(
        self,
        query: PlannedQuery,
    ) -> tuple[SearchExecutionStatus, Optional[SearchBackendResponse], int, Optional[str]]:
        """Execute a single PlannedQuery with bounded retries for transient failures."""
        configured, config_msg = self.backend.is_configured()
        if not configured:
            return (
                SearchExecutionStatus.UNAVAILABLE,
                None,
                1,
                config_msg,
            )

        max_attempts = max(1, 1 + self.max_retries)
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.backend.search(
                    query.rendered_query,
                    max_results=self.max_results_per_query,
                    timeout_seconds=self.timeout_seconds,
                )
                if not isinstance(response, SearchBackendResponse):
                    raise SearchProviderError(
                        f"Search backend returned unexpected response type: {type(response).__name__}."
                    )
                if not isinstance(response.results, list):
                    raise SearchProviderError("Search backend response 'results' must be a list.")
                return SearchExecutionStatus.EXECUTED, response, attempt, None
            except SearchConfigurationError as exc:
                return SearchExecutionStatus.UNAVAILABLE, None, attempt, str(exc)
            except SearchExecutionError as exc:
                last_error = str(exc)
                if exc.retryable and attempt < max_attempts:
                    if self.retry_backoff_seconds > 0:
                        self._sleep_fn(self.retry_backoff_seconds * attempt)
                    continue
                return SearchExecutionStatus.FAILED, None, attempt, last_error
            except Exception as exc:
                last_error = f"Unexpected search provider error: {exc}"
                return SearchExecutionStatus.FAILED, None, attempt, last_error

        return SearchExecutionStatus.FAILED, None, max_attempts, last_error or "Unknown search failure."

    def _normalize_result_item(
        self,
        raw_item: Any,
        default_rank: int,
        default_retrieved_at: str,
    ) -> Optional[ tuple[str, SearchResultItem] ]:
        """Validate a single raw result item and canonicalize its URL.

        Returns `(canonical_url, SearchResultItem)` or `None` if malformed/unusable.
        """
        if isinstance(raw_item, SearchResultItem):
            raw_url = raw_item.url
            title = raw_item.title
            snippet = raw_item.snippet
            rank = raw_item.rank if raw_item.rank is not None else default_rank
            retrieved_at = raw_item.retrieved_at or default_retrieved_at
        elif isinstance(raw_item, dict):
            raw_url = raw_item.get("url") or raw_item.get("link")
            if not raw_url or not isinstance(raw_url, str):
                return None
            title = raw_item.get("title") if isinstance(raw_item.get("title"), str) else None
            snippet = raw_item.get("snippet") if isinstance(raw_item.get("snippet"), str) else None
            raw_rank = raw_item.get("rank")
            rank = int(raw_rank) if isinstance(raw_rank, int) and raw_rank > 0 else default_rank
            raw_ts = raw_item.get("retrieved_at")
            retrieved_at = raw_ts if isinstance(raw_ts, str) and raw_ts.strip() else default_retrieved_at
        else:
            return None

        canonical = canonicalize_url(raw_url, strip_tracking_params=self._strip_tracking_params)
        if not canonical:
            return None

        return canonical, SearchResultItem(
            url=raw_url.strip(),
            title=title.strip() if isinstance(title, str) and title.strip() else None,
            snippet=snippet.strip() if isinstance(snippet, str) and snippet.strip() else None,
            rank=rank,
            retrieved_at=retrieved_at,
        )

    def execute_planned_queries(
        self,
        plan: OSINTInvestigationPlan,
        *,
        query_ids: Optional[Sequence[str]] = None,
        retry_failed: bool = False,
        retry_unavailable: bool = False,
        persist: bool = False,
        state_path: Path = DEFAULT_OSINT_STATE_PATH,
    ) -> OSINTInvestigationPlan:
        """Execute eligible queries in `plan`, collect/deduplicate candidate URLs, and update source coverage.

        By default, only queries with `execution_status == SearchExecutionStatus.PLANNED` are executed.
        Already-`EXECUTED` queries are never re-executed unless explicitly targeted in `query_ids` with retry flags.
        """
        eligible_statuses = {SearchExecutionStatus.PLANNED}
        if retry_failed:
            eligible_statuses.add(SearchExecutionStatus.FAILED)
        if retry_unavailable:
            eligible_statuses.add(SearchExecutionStatus.UNAVAILABLE)

        target_id_set = set(query_ids) if query_ids is not None else None

        # Index existing candidate URLs by canonical URL for conservative deduplication & provenance merging
        candidates_by_url: dict[str, CandidateURL] = {}
        for existing_cand in plan.candidate_urls:
            canon = canonicalize_url(existing_cand.url, strip_tracking_params=self._strip_tracking_params) or existing_cand.url
            candidates_by_url[canon] = existing_cand

        executed_count = 0
        failed_count = 0
        unavailable_count = 0
        skipped_count = 0
        new_urls_added = 0
        queries_processed = 0

        for q in plan.planned_queries:
            if target_id_set is not None and q.query_id not in target_id_set:
                continue

            if q.execution_status not in eligible_statuses:
                skipped_count += 1
                continue

            if queries_processed >= self.max_queries_per_run:
                skipped_count += 1
                continue

            queries_processed += 1
            now_iso = _utc_now_iso()
            status, response, attempts, error_msg = self._execute_single_query_with_retry(q)

            q.execution_status = status
            q.provider = response.provider if response is not None else self.backend.provider_name
            q.executed_at = response.executed_at if response is not None else now_iso
            q.attempts = attempts
            q.error_message = error_msg

            if status == SearchExecutionStatus.UNAVAILABLE:
                q.result_count = 0
                unavailable_count += 1
                continue

            if status == SearchExecutionStatus.FAILED or response is None:
                q.result_count = 0
                failed_count += 1
                continue

            # Process and bound results for EXECUTED query
            raw_results = response.results[: self.max_results_per_query]
            valid_results_count = 0

            for idx, raw_item in enumerate(raw_results, start=1):
                norm_pair = self._normalize_result_item(
                    raw_item,
                    default_rank=idx,
                    default_retrieved_at=q.executed_at or now_iso,
                )
                if norm_pair is None:
                    continue

                canonical_url, norm_item = norm_pair
                valid_results_count += 1

                primary_source = q.source_classes[0] if q.source_classes else ""
                primary_resource = (
                    q.discovery_resources[0]
                    if q.discovery_resources
                    else (q.provider or self.backend.provider_name)
                )

                if canonical_url not in candidates_by_url:
                    cand_id = f"CURL_{plan.vendor_id}_{len(candidates_by_url) + 1:03d}"
                    new_cand = CandidateURL(
                        candidate_id=cand_id,
                        url=canonical_url,
                        raw_url=norm_item.url,
                        vendor_id=plan.vendor_id,
                        source_class=primary_source,
                        source_classes=list(q.source_classes),
                        discovery_resource=primary_resource,
                        discovery_resources=list(q.discovery_resources) if q.discovery_resources else [primary_resource],
                        query_id=q.query_id,
                        query_ids=[q.query_id],
                        executed_queries=[q.rendered_query],
                        triage_status=URLTriageStatus.PENDING,
                        title=norm_item.title,
                        snippet=norm_item.snippet,
                        rank=norm_item.rank,
                        retrieved_at=norm_item.retrieved_at,
                        provider=q.provider,
                    )
                    candidates_by_url[canonical_url] = new_cand
                    plan.candidate_urls.append(new_cand)
                    new_urls_added += 1
                else:
                    existing = candidates_by_url[canonical_url]
                    for sc in q.source_classes:
                        if sc not in existing.source_classes:
                            existing.source_classes.append(sc)
                    if not existing.source_class and existing.source_classes:
                        existing.source_class = existing.source_classes[0]

                    for dr in q.discovery_resources:
                        if dr not in existing.discovery_resources:
                            existing.discovery_resources.append(dr)
                    if not existing.discovery_resource and existing.discovery_resources:
                        existing.discovery_resource = existing.discovery_resources[0]

                    if q.query_id not in existing.query_ids:
                        existing.query_ids.append(q.query_id)
                    if not existing.query_id and existing.query_ids:
                        existing.query_id = existing.query_ids[0]

                    if q.rendered_query not in existing.executed_queries:
                        existing.executed_queries.append(q.rendered_query)

                    if not existing.title and norm_item.title:
                        existing.title = norm_item.title
                    if not existing.snippet and norm_item.snippet:
                        existing.snippet = norm_item.snippet
                    if norm_item.rank is not None and (existing.rank is None or norm_item.rank < existing.rank):
                        existing.rank = norm_item.rank

            q.result_count = valid_results_count
            executed_count += 1

        # Synchronize SourceCoverageRecord candidate counts & pre-review coverage states
        self._sync_source_coverage_from_collection(plan)

        planned_remaining = sum(
            1 for q in plan.planned_queries if q.execution_status == SearchExecutionStatus.PLANNED
        )
        plan.last_collection_summary = {
            "provider": self.backend.provider_name,
            "collected_at": _utc_now_iso(),
            "queries_executed": executed_count,
            "queries_failed": failed_count,
            "queries_unavailable": unavailable_count,
            "queries_skipped": skipped_count,
            "queries_planned_remaining": planned_remaining,
            "new_candidate_urls": new_urls_added,
            "total_candidate_urls": len(plan.candidate_urls),
        }

        if persist:
            OSINTInvestigationPlanner.save_plan_state(plan, state_path=state_path)

        return plan

    @staticmethod
    def _sync_source_coverage_from_collection(plan: OSINTInvestigationPlan) -> None:
        """Update SourceCoverageRecord candidate counts and search states without fabricating reviewed evidence.

        Enforces Section 4 boundaries:
          - Never sets `REVIEWED_EVIDENCE_RECORDED` or `REVIEWED_NO_EVIDENCE` from search results alone.
          - Never overwrites analyst-reviewed states (`REVIEWED_EVIDENCE_RECORDED`, `REVIEWED_NO_EVIDENCE`,
            `RELEVANT_SOURCE_IDENTIFIED`, `UNAVAILABLE`, `REQUIRES_MANUAL_REVIEW`).
          - Transitions `NOT_STARTED` / `SEARCH_ATTEMPTED` to `RESULTS_FOUND` when candidate URLs > 0.
          - Transitions `NOT_STARTED` to `SEARCH_ATTEMPTED` when >= 1 query for that source class executed
            and returned 0 candidate URLs.
          - Leaves `NOT_STARTED` unchanged if queries for that source class were only `PLANNED`, `FAILED`,
            or `UNAVAILABLE`.
        """
        preserved_analyst_states = {
            CoverageState.REVIEWED_EVIDENCE_RECORDED,
            CoverageState.REVIEWED_NO_EVIDENCE,
            CoverageState.RELEVANT_SOURCE_IDENTIFIED,
            CoverageState.UNAVAILABLE,
            CoverageState.REQUIRES_MANUAL_REVIEW,
        }

        for s_id, rec in plan.coverage_records.items():
            # Count distinct candidate URLs associated with this source class
            source_url_count = sum(
                1
                for cand in plan.candidate_urls
                if s_id in cand.source_classes or cand.source_class == s_id
            )
            had_executed_query = any(
                q.execution_status == SearchExecutionStatus.EXECUTED and s_id in q.source_classes
                for q in plan.planned_queries
            )

            if rec.status in preserved_analyst_states:
                rec.candidate_urls_count = max(rec.candidate_urls_count, source_url_count)
                continue

            rec.candidate_urls_count = source_url_count
            if source_url_count > 0:
                rec.status = CoverageState.RESULTS_FOUND
                if not rec.notes:
                    rec.notes = f"{source_url_count} candidate URL(s) collected; awaiting analyst triage and review."
            elif had_executed_query:
                rec.status = CoverageState.SEARCH_ATTEMPTED
                if not rec.notes:
                    rec.notes = "Planned search query executed with 0 candidate URLs; awaiting review or manual lookup."
