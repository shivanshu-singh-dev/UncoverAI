"""Centralized Dual-Model Local LLM Router for UncoverAI (6 GB VRAM Budget).

Routes tasks across the two local Ollama generative models while keeping Level 5
embeddings completely separate:
  - Model A (`llama3.2:latest`, ~2.0 GB):
      * `LLMTaskType.URL_TRIAGE` (Level 3B candidate URL & search-result triage)
  - Model B (`mistral:latest`, ~4.4 GB):
      * `LLMTaskType.EVIDENCE_CLASSIFICATION` (Level 6 semantic evidence classification)
      * `LLMTaskType.RISK_SUMMARIZATION` (Level 8 evidence-grounded risk finding summaries)
  - Separate Embedding Path (`sentence-transformers/all-MiniLM-L6-v2`):
      * `LLMTaskType.RAG_EMBEDDING` (Level 5 semantic filtering & chunk retrieval — never routed to Ollama)

VRAM Management (6 GB GPU):
  - Tracks the currently loaded generative model (`active_generative_model`).
  - Automatically unloads the previous model from VRAM (`keep_alive: 0`) before switching
    from `llama3.2:latest` to `mistral:latest` (or vice versa), preventing both models
    (~6.4 GB combined) from occupying VRAM simultaneously.
  - Supports explicit stage boundary unloading (`unload_active_model()`) after Level 3B completes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Mapping, Optional
import urllib.error
import urllib.request

from pydantic import BaseModel, Field
import yaml

from meridian_assessment.config.env_loader import DEFAULT_DOTENV_PATH, load_dotenv_if_present

_DEFAULT_AI_RISK_CONFIG_PATH = Path("config/ai_risk_scoring.yaml")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LLMTaskType(StrEnum):
    """Supported model tasks across the 9-level Meridian pipeline."""

    URL_TRIAGE = "URL_TRIAGE"                          # Level 3B -> Model A (llama3.2:latest)
    RAG_EMBEDDING = "RAG_EMBEDDING"                    # Level 5  -> Separate embedding model (all-MiniLM-L6-v2)
    EVIDENCE_CLASSIFICATION = "EVIDENCE_CLASSIFICATION"  # Level 6  -> Model B (mistral:latest)
    RISK_SUMMARIZATION = "RISK_SUMMARIZATION"          # Level 8  -> Model B (mistral:latest)


class LLMExecutionStatus(StrEnum):
    """Explicit execution status for a local LLM operation."""

    SUCCEEDED = "SUCCEEDED"
    SKIPPED = "SKIPPED"
    UNAVAILABLE = "UNAVAILABLE"
    MODEL_NOT_INSTALLED = "MODEL_NOT_INSTALLED"
    TIMEOUT = "TIMEOUT"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    PROMPT_INJECTION_BLOCKED = "PROMPT_INJECTION_BLOCKED"
    FAILED = "FAILED"


class ModelRoutingConfig(BaseModel):
    """Centralized configuration for dual-model Ollama routing and separate RAG embeddings."""

    enabled: bool = False
    provider: str = "ollama"
    endpoint: str = "http://localhost:11434"
    triage_model: str = "llama3.2:latest"
    analysis_model: str = "mistral:latest"
    summary_model: str = "mistral:latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_rag_embeddings: bool = False
    timeout_seconds: float = 25.0
    keep_alive: str = "2m"
    unload_on_model_switch: bool = True

    @classmethod
    def from_env_and_config(
        cls,
        *,
        env: Optional[Mapping[str, str]] = None,
        dotenv_path: Path = DEFAULT_DOTENV_PATH,
        load_dotenv: bool = True,
        config_path: Path = _DEFAULT_AI_RISK_CONFIG_PATH,
    ) -> "ModelRoutingConfig":
        """Build `ModelRoutingConfig` by merging YAML defaults, `.env`, and environment variables."""
        if env is None:
            if load_dotenv:
                load_dotenv_if_present(dotenv_path)
            environ: Mapping[str, str] = os.environ
        else:
            if load_dotenv and dotenv_path.is_file():
                merged: dict[str, str] = {}
                load_dotenv_if_present(dotenv_path, target_env=merged)
                merged.update(env)
                environ = merged
            else:
                environ = env

        yaml_llm: dict[str, Any] = {}
        yaml_rag: dict[str, Any] = {}
        if config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                    yaml_llm = raw.get("local_llm", {}) or {}
                    yaml_rag = raw.get("rag_chunking", {}) or {}
            except Exception:
                pass

        def _parse_bool(val: Any, default: bool) -> bool:
            if isinstance(val, bool):
                return val
            if val is None:
                return default
            s = str(val).strip().lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off", ""):
                return False
            return default

        enabled = _parse_bool(
            environ.get("MERIDIAN_OLLAMA_ENABLED"),
            bool(yaml_llm.get("enabled", False)),
        )
        endpoint = (
            environ.get("MERIDIAN_OLLAMA_ENDPOINT")
            or str(yaml_llm.get("endpoint") or "http://localhost:11434")
        ).strip().rstrip("/")

        triage_model = (
            environ.get("MERIDIAN_OLLAMA_TRIAGE_MODEL")
            or str(yaml_llm.get("triage_model") or "llama3.2:latest")
        ).strip()

        # Support MERIDIAN_OLLAMA_ANALYSIS_MODEL with fallback to legacy MERIDIAN_OLLAMA_MODEL
        analysis_model = (
            environ.get("MERIDIAN_OLLAMA_ANALYSIS_MODEL")
            or environ.get("MERIDIAN_OLLAMA_MODEL")
            or str(yaml_llm.get("analysis_model") or yaml_llm.get("model_name") or "mistral:latest")
        ).strip()

        summary_model = (
            environ.get("MERIDIAN_OLLAMA_SUMMARY_MODEL")
            or environ.get("MERIDIAN_OLLAMA_ANALYSIS_MODEL")
            or environ.get("MERIDIAN_OLLAMA_MODEL")
            or str(yaml_llm.get("summary_model") or analysis_model or "mistral:latest")
        ).strip()

        embedding_model = (
            environ.get("MERIDIAN_EMBEDDING_MODEL")
            or str(yaml_rag.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2")
        ).strip()

        enable_rag_embeddings = _parse_bool(
            environ.get("MERIDIAN_ENABLE_RAG_EMBEDDINGS"),
            bool(yaml_rag.get("auto_load_model", False)),
        )

        raw_timeout = environ.get("MERIDIAN_OLLAMA_TIMEOUT_SECONDS") or yaml_llm.get("timeout_seconds", 25.0)
        try:
            timeout_seconds = max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            timeout_seconds = 25.0

        keep_alive = (
            environ.get("MERIDIAN_OLLAMA_KEEP_ALIVE")
            or str(yaml_llm.get("keep_alive") or "2m")
        ).strip()

        unload_on_switch = _parse_bool(
            environ.get("MERIDIAN_OLLAMA_UNLOAD_ON_SWITCH"),
            bool(yaml_llm.get("unload_on_model_switch", True)),
        )

        return cls(
            enabled=enabled,
            provider=str(yaml_llm.get("provider") or "ollama"),
            endpoint=endpoint,
            triage_model=triage_model,
            analysis_model=analysis_model,
            summary_model=summary_model,
            embedding_model=embedding_model,
            enable_rag_embeddings=enable_rag_embeddings,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
            unload_on_model_switch=unload_on_switch,
        )


class LLMCallTelemetry(BaseModel):
    """Per-call audit and telemetry record for local LLM invocations."""

    call_id: str
    task: LLMTaskType
    model_name: str
    endpoint: str
    status: LLMExecutionStatus
    latency_ms: float = 0.0
    vram_unloaded_prior_model: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: str = Field(default_factory=_utc_now_iso)


class LLMGenerationResponse(BaseModel):
    """Structured result from `OllamaModelRouter.generate_json`."""

    call_id: str
    task: LLMTaskType
    model_name: str
    status: LLMExecutionStatus
    parsed_json: Optional[dict[str, Any]] = None
    raw_text: str = ""
    error_message: Optional[str] = None
    vram_unloaded_prior_model: Optional[str] = None
    latency_ms: float = 0.0


def _default_ollama_http_post(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Execute a synchronous JSON POST request to the local Ollama HTTP server."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw_bytes = resp.read()
        return json.loads(raw_bytes.decode("utf-8", errors="replace"))


class OllamaModelRouter:
    """Centralized task-based model router and VRAM lifecycle manager for local Ollama models."""

    def __init__(
        self,
        config: Optional[ModelRoutingConfig] = None,
        *,
        http_post_fn: Optional[Callable[[str, dict[str, Any], float], dict[str, Any]]] = None,
    ) -> None:
        self.config: ModelRoutingConfig = config or ModelRoutingConfig.from_env_and_config()
        self._http_post_fn = http_post_fn or _default_ollama_http_post
        self.active_generative_model: Optional[str] = None
        self.call_history: list[LLMCallTelemetry] = []
        self.vram_unload_events: list[dict[str, Any]] = []
        self._call_seq: int = 0

    def resolve_model_for_task(self, task: LLMTaskType) -> str:
        """Return the configured model name for the specified pipeline task."""
        if task == LLMTaskType.URL_TRIAGE:
            return self.config.triage_model
        if task == LLMTaskType.EVIDENCE_CLASSIFICATION:
            return self.config.analysis_model
        if task == LLMTaskType.RISK_SUMMARIZATION:
            return self.config.summary_model
        if task == LLMTaskType.RAG_EMBEDDING:
            return self.config.embedding_model
        raise ValueError(f"Unsupported LLMTaskType: {task}")

    def is_task_configured(self, task: LLMTaskType) -> tuple[bool, str]:
        """Return `(True, summary)` if the task's model is enabled and configured."""
        if task == LLMTaskType.RAG_EMBEDDING:
            if not self.config.enable_rag_embeddings:
                return (
                    False,
                    f"RAG embedding model '{self.config.embedding_model}' is disabled (MERIDIAN_ENABLE_RAG_EMBEDDINGS=false).",
                )
            return True, f"RAG embedding model '{self.config.embedding_model}' enabled."

        model_name = self.resolve_model_for_task(task)
        if not self.config.enabled:
            return (
                False,
                f"Local Ollama routing disabled (MERIDIAN_OLLAMA_ENABLED=false; assigned model for {task.value}: '{model_name}').",
            )
        if not self.config.endpoint or not model_name:
            return (
                False,
                f"Local Ollama endpoint or model name missing for task {task.value}.",
            )
        return (
            True,
            f"Ollama model '{model_name}' configured at {self.config.endpoint} for {task.value}.",
        )

    def unload_model(self, model_name: str, *, reason: str = "Explicit VRAM unload") -> bool:
        """Explicitly unload `model_name` from Ollama VRAM by sending `keep_alive: 0`.

        Ensures `llama3.2:latest` (~2 GB) and `mistral:latest` (~4.4 GB) are never kept
        resident in the 6 GB VRAM simultaneously.
        """
        if not self.config.enabled or not self.config.endpoint or not model_name:
            if self.active_generative_model == model_name:
                self.active_generative_model = None
            return False

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        unloaded_ok = False
        error_str: Optional[str] = None
        try:
            self._http_post_fn(
                f"{self.config.endpoint}/api/generate",
                payload,
                min(5.0, self.config.timeout_seconds),
            )
            unloaded_ok = True
        except Exception as exc:
            error_str = str(exc)

        if self.active_generative_model == model_name:
            self.active_generative_model = None

        self.vram_unload_events.append({
            "model_unloaded": model_name,
            "reason": reason,
            "keep_alive": 0,
            "succeeded": unloaded_ok,
            "error": error_str,
            "timestamp": _utc_now_iso(),
        })
        return unloaded_ok

    def unload_active_model(self, *, reason: str = "Stage completion VRAM release") -> Optional[str]:
        """Unload whichever generative model is currently active in VRAM, if any."""
        if self.active_generative_model:
            prev = self.active_generative_model
            self.unload_model(prev, reason=reason)
            return prev
        return None

    def generate_json(
        self,
        task: LLMTaskType,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        keep_alive_override: Optional[str | int] = None,
    ) -> LLMGenerationResponse:
        """Execute a structured JSON generation request for `task` using its routed Ollama model.

        Never allows `LLMTaskType.RAG_EMBEDDING` to be routed to Ollama generative models.
        Automatically unloads the prior generative model when switching between Model A and Model B.
        """
        if task == LLMTaskType.RAG_EMBEDDING:
            raise ValueError(
                "LLMTaskType.RAG_EMBEDDING must use the separate EmbeddingProvider "
                f"('{self.config.embedding_model}') and must never be routed through Ollama generative models."
            )

        self._call_seq += 1
        call_id = f"LLM_{self._call_seq:04d}_{task.value}"
        model_name = self.resolve_model_for_task(task)

        ok, msg = self.is_task_configured(task)
        if not ok:
            rec = LLMCallTelemetry(
                call_id=call_id,
                task=task,
                model_name=model_name,
                endpoint=self.config.endpoint,
                status=LLMExecutionStatus.SKIPPED,
                error_message=msg,
            )
            self.call_history.append(rec)
            return LLMGenerationResponse(
                call_id=call_id,
                task=task,
                model_name=model_name,
                status=LLMExecutionStatus.SKIPPED,
                error_message=msg,
            )

        # VRAM-safe model switch check (e.g. llama3.2:latest -> mistral:latest)
        unloaded_prior: Optional[str] = None
        if (
            self.active_generative_model is not None
            and self.active_generative_model != model_name
            and self.config.unload_on_model_switch
        ):
            prior_model = self.active_generative_model
            self.unload_model(
                prior_model,
                reason=f"VRAM-safe switch from '{prior_model}' to '{model_name}' for task {task.value}",
            )
            unloaded_prior = prior_model

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "context": [],
            "keep_alive": self.config.keep_alive if keep_alive_override is None else keep_alive_override,
            "options": {
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        t0 = time.perf_counter()
        try:
            raw_resp = self._http_post_fn(
                f"{self.config.endpoint}/api/generate",
                payload,
                self.config.timeout_seconds,
            )
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        except urllib.error.HTTPError as http_err:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            err_body = ""
            try:
                err_body = http_err.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if http_err.code == 404 or "not found" in err_body.lower():
                status = LLMExecutionStatus.MODEL_NOT_INSTALLED
                err_msg = f"Ollama model '{model_name}' not installed (HTTP {http_err.code}): {err_body or http_err.reason}"
            else:
                status = LLMExecutionStatus.FAILED
                err_msg = f"Ollama HTTP {http_err.code} error for '{model_name}': {err_body or http_err.reason}"
            self.call_history.append(
                LLMCallTelemetry(
                    call_id=call_id,
                    task=task,
                    model_name=model_name,
                    endpoint=self.config.endpoint,
                    status=status,
                    latency_ms=latency_ms,
                    vram_unloaded_prior_model=unloaded_prior,
                    error_message=err_msg,
                )
            )
            return LLMGenerationResponse(
                call_id=call_id,
                task=task,
                model_name=model_name,
                status=status,
                error_message=err_msg,
                vram_unloaded_prior_model=unloaded_prior,
                latency_ms=latency_ms,
            )
        except (TimeoutError, socket.timeout) as timeout_err:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            err_msg = f"Ollama request timed out after {self.config.timeout_seconds}s for '{model_name}': {timeout_err}"
            self.call_history.append(
                LLMCallTelemetry(
                    call_id=call_id,
                    task=task,
                    model_name=model_name,
                    endpoint=self.config.endpoint,
                    status=LLMExecutionStatus.TIMEOUT,
                    latency_ms=latency_ms,
                    vram_unloaded_prior_model=unloaded_prior,
                    error_message=err_msg,
                )
            )
            return LLMGenerationResponse(
                call_id=call_id,
                task=task,
                model_name=model_name,
                status=LLMExecutionStatus.TIMEOUT,
                error_message=err_msg,
                vram_unloaded_prior_model=unloaded_prior,
                latency_ms=latency_ms,
            )
        except urllib.error.URLError as url_err:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if isinstance(getattr(url_err, "reason", None), (TimeoutError, socket.timeout)):
                status = LLMExecutionStatus.TIMEOUT
                err_msg = f"Ollama request timed out after {self.config.timeout_seconds}s for '{model_name}': {url_err.reason}"
            else:
                status = LLMExecutionStatus.UNAVAILABLE
                err_msg = f"Ollama service unreachable at {self.config.endpoint} for '{model_name}': {url_err.reason}"
            self.call_history.append(
                LLMCallTelemetry(
                    call_id=call_id,
                    task=task,
                    model_name=model_name,
                    endpoint=self.config.endpoint,
                    status=status,
                    latency_ms=latency_ms,
                    vram_unloaded_prior_model=unloaded_prior,
                    error_message=err_msg,
                )
            )
            return LLMGenerationResponse(
                call_id=call_id,
                task=task,
                model_name=model_name,
                status=status,
                error_message=err_msg,
                vram_unloaded_prior_model=unloaded_prior,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            msg_low = str(exc).lower()
            if "timed out" in msg_low or "timeout" in msg_low:
                status = LLMExecutionStatus.TIMEOUT
            elif "connection refused" in msg_low or "unreachable" in msg_low:
                status = LLMExecutionStatus.UNAVAILABLE
            elif "not found" in msg_low or "not installed" in msg_low:
                status = LLMExecutionStatus.MODEL_NOT_INSTALLED
            else:
                status = LLMExecutionStatus.FAILED
            err_msg = f"Ollama invocation error for '{model_name}': {exc}"
            self.call_history.append(
                LLMCallTelemetry(
                    call_id=call_id,
                    task=task,
                    model_name=model_name,
                    endpoint=self.config.endpoint,
                    status=status,
                    latency_ms=latency_ms,
                    vram_unloaded_prior_model=unloaded_prior,
                    error_message=err_msg,
                )
            )
            return LLMGenerationResponse(
                call_id=call_id,
                task=task,
                model_name=model_name,
                status=status,
                error_message=err_msg,
                vram_unloaded_prior_model=unloaded_prior,
                latency_ms=latency_ms,
            )

        # Update active model state if keep_alive != 0
        effective_keep_alive = payload.get("keep_alive")
        if effective_keep_alive in (0, "0", "0s"):
            self.active_generative_model = None
        else:
            self.active_generative_model = model_name

        if not isinstance(raw_resp, dict):
            err_msg = f"Malformed non-dict response from Ollama for '{model_name}'."
            self.call_history.append(
                LLMCallTelemetry(
                    call_id=call_id,
                    task=task,
                    model_name=model_name,
                    endpoint=self.config.endpoint,
                    status=LLMExecutionStatus.INVALID_OUTPUT,
                    latency_ms=latency_ms,
                    vram_unloaded_prior_model=unloaded_prior,
                    error_message=err_msg,
                )
            )
            return LLMGenerationResponse(
                call_id=call_id,
                task=task,
                model_name=model_name,
                status=LLMExecutionStatus.INVALID_OUTPUT,
                error_message=err_msg,
                vram_unloaded_prior_model=unloaded_prior,
                latency_ms=latency_ms,
            )

        if "error" in raw_resp and raw_resp["error"]:
            err_text = str(raw_resp["error"])
            status = (
                LLMExecutionStatus.MODEL_NOT_INSTALLED
                if "not found" in err_text.lower()
                else LLMExecutionStatus.FAILED
            )
            self.call_history.append(
                LLMCallTelemetry(
                    call_id=call_id,
                    task=task,
                    model_name=model_name,
                    endpoint=self.config.endpoint,
                    status=status,
                    latency_ms=latency_ms,
                    vram_unloaded_prior_model=unloaded_prior,
                    error_message=err_text,
                )
            )
            return LLMGenerationResponse(
                call_id=call_id,
                task=task,
                model_name=model_name,
                status=status,
                error_message=err_text,
                vram_unloaded_prior_model=unloaded_prior,
                latency_ms=latency_ms,
            )

        raw_text = str(raw_resp.get("response") or "").strip()
        if not raw_text:
            err_msg = f"Empty response body returned by Ollama model '{model_name}'."
            self.call_history.append(
                LLMCallTelemetry(
                    call_id=call_id,
                    task=task,
                    model_name=model_name,
                    endpoint=self.config.endpoint,
                    status=LLMExecutionStatus.INVALID_OUTPUT,
                    latency_ms=latency_ms,
                    vram_unloaded_prior_model=unloaded_prior,
                    error_message=err_msg,
                )
            )
            return LLMGenerationResponse(
                call_id=call_id,
                task=task,
                model_name=model_name,
                status=LLMExecutionStatus.INVALID_OUTPUT,
                raw_text="",
                error_message=err_msg,
                vram_unloaded_prior_model=unloaded_prior,
                latency_ms=latency_ms,
            )

        try:
            parsed_json = json.loads(raw_text)
            if not isinstance(parsed_json, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed_json).__name__}")
        except Exception as parse_err:
            err_msg = f"Invalid JSON returned by Ollama model '{model_name}': {parse_err}"
            self.call_history.append(
                LLMCallTelemetry(
                    call_id=call_id,
                    task=task,
                    model_name=model_name,
                    endpoint=self.config.endpoint,
                    status=LLMExecutionStatus.INVALID_OUTPUT,
                    latency_ms=latency_ms,
                    vram_unloaded_prior_model=unloaded_prior,
                    error_message=err_msg,
                )
            )
            return LLMGenerationResponse(
                call_id=call_id,
                task=task,
                model_name=model_name,
                status=LLMExecutionStatus.INVALID_OUTPUT,
                raw_text=raw_text,
                error_message=err_msg,
                vram_unloaded_prior_model=unloaded_prior,
                latency_ms=latency_ms,
            )

        self.call_history.append(
            LLMCallTelemetry(
                call_id=call_id,
                task=task,
                model_name=model_name,
                endpoint=self.config.endpoint,
                status=LLMExecutionStatus.SUCCEEDED,
                latency_ms=latency_ms,
                vram_unloaded_prior_model=unloaded_prior,
            )
        )
        return LLMGenerationResponse(
            call_id=call_id,
            task=task,
            model_name=model_name,
            status=LLMExecutionStatus.SUCCEEDED,
            parsed_json=parsed_json,
            raw_text=raw_text,
            vram_unloaded_prior_model=unloaded_prior,
            latency_ms=latency_ms,
        )

    def mark_call_invalid_output(self, call_id: str, reason: str) -> None:
        """Downgrade a call's telemetry status from SUCCEEDED to INVALID_OUTPUT when schema/grounding validation fails."""
        for rec in reversed(self.call_history):
            if rec.call_id == call_id:
                rec.status = LLMExecutionStatus.INVALID_OUTPUT
                rec.error_message = reason
                break

    def record_skipped_or_blocked_call(
        self,
        task: LLMTaskType,
        *,
        status: LLMExecutionStatus,
        reason: str,
    ) -> None:
        """Record a skipped or prompt-injection-blocked operation in telemetry."""
        self._call_seq += 1
        call_id = f"LLM_{self._call_seq:04d}_{task.value}"
        model_name = self.resolve_model_for_task(task)
        self.call_history.append(
            LLMCallTelemetry(
                call_id=call_id,
                task=task,
                model_name=model_name,
                endpoint=self.config.endpoint,
                status=status,
                error_message=reason,
            )
        )

    def get_task_telemetry_summary(self, task: LLMTaskType) -> dict[str, Any]:
        """Summarize execution status for a specific task (`URL_TRIAGE`, `EVIDENCE_CLASSIFICATION`, `RISK_SUMMARIZATION`)."""
        model_name = self.resolve_model_for_task(task)
        configured, config_msg = self.is_task_configured(task)
        records = [r for r in self.call_history if r.task == task]

        counts: dict[str, int] = {}
        for r in records:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1

        succeeded = counts.get(LLMExecutionStatus.SUCCEEDED.value, 0)
        failed_or_invalid = (
            counts.get(LLMExecutionStatus.FAILED.value, 0)
            + counts.get(LLMExecutionStatus.INVALID_OUTPUT.value, 0)
            + counts.get(LLMExecutionStatus.TIMEOUT.value, 0)
        )
        unavailable_cnt = (
            counts.get(LLMExecutionStatus.UNAVAILABLE.value, 0)
            + counts.get(LLMExecutionStatus.MODEL_NOT_INSTALLED.value, 0)
        )

        if not configured:
            overall_status = "SKIPPED_DISABLED"
        elif not records:
            overall_status = "SKIPPED_NO_INPUT"
        elif succeeded > 0 and failed_or_invalid == 0 and unavailable_cnt == 0:
            overall_status = "SUCCEEDED"
        elif succeeded > 0:
            overall_status = "PARTIAL_FALLBACK"
        elif unavailable_cnt > 0:
            overall_status = "UNAVAILABLE_FALLBACK"
        elif failed_or_invalid > 0:
            overall_status = "FAILED_FALLBACK"
        else:
            overall_status = "SKIPPED"

        errors = [r.error_message for r in records if r.error_message]
        return {
            "task": task.value,
            "selected_model": model_name,
            "configured": configured,
            "configuration_message": config_msg,
            "overall_status": overall_status,
            "calls_total": len(records),
            "calls_succeeded": succeeded,
            "status_breakdown": counts,
            "errors": errors[:5],
        }

    def get_full_telemetry_summary(self) -> dict[str, Any]:
        """Return complete dual-model routing and VRAM switch telemetry for reporting and UI display."""
        return {
            "ollama_enabled": self.config.enabled,
            "ollama_endpoint": self.config.endpoint,
            "model_assignments": {
                LLMTaskType.URL_TRIAGE.value: self.config.triage_model,
                LLMTaskType.EVIDENCE_CLASSIFICATION.value: self.config.analysis_model,
                LLMTaskType.RISK_SUMMARIZATION.value: self.config.summary_model,
                LLMTaskType.RAG_EMBEDDING.value: self.config.embedding_model,
            },
            "vram_management": {
                "unload_on_model_switch": self.config.unload_on_model_switch,
                "keep_alive": self.config.keep_alive,
                "active_generative_model": self.active_generative_model,
                "vram_unload_events": list(self.vram_unload_events),
            },
            "tasks": {
                LLMTaskType.URL_TRIAGE.value: self.get_task_telemetry_summary(LLMTaskType.URL_TRIAGE),
                LLMTaskType.EVIDENCE_CLASSIFICATION.value: self.get_task_telemetry_summary(
                    LLMTaskType.EVIDENCE_CLASSIFICATION
                ),
                LLMTaskType.RISK_SUMMARIZATION.value: self.get_task_telemetry_summary(
                    LLMTaskType.RISK_SUMMARIZATION
                ),
            },
            "call_history": [r.model_dump(mode="json") for r in self.call_history],
        }
