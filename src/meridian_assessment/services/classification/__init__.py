"""Level 6 Signal Classification, Local-LLM Adapter, and Prompt-Injection Defence package."""

from meridian_assessment.services.classification.signal_classifier import (
    LocalLLMClassificationAdapter,
    OllamaClassificationAdapter,
    SignalClassificationService,
    StubLLMClassificationAdapter,
    UnconfiguredOllamaAdapter,
    detect_prompt_injection,
)

__all__ = [
    "LocalLLMClassificationAdapter",
    "OllamaClassificationAdapter",
    "SignalClassificationService",
    "StubLLMClassificationAdapter",
    "UnconfiguredOllamaAdapter",
    "detect_prompt_injection",
]
