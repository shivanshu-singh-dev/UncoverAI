"""Text normalization utilities shared by factor classifiers."""
import re
from typing import Optional


ABBREVIATIONS = {
    "pii": "personally identifiable information",
    "ssn": "social security number",
    "phi": "protected health information",
    "ein": "employer identification number",
    "tin": "tax identification number",
    "ach": "automated clearing house",
    "rtp": "real time payment",
    "api": "application programming interface",
    "fmu": "financial market utility",
    "dco": "designated clearing organization",
}


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, strip, collapse whitespace, expand abbreviations."""
    if not text:
        return ""
    t = text.lower().strip()
    # Collapse multiple spaces/newlines
    t = re.sub(r'[\s\n\r\t]+', ' ', t)
    # Remove punctuation except hyphens (keep compound words)
    t = re.sub(r'[^\w\s-]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    # Expand abbreviations
    words = t.split()
    expanded = [ABBREVIATIONS.get(w, w) for w in words]
    return ' '.join(expanded)


def contains_any(text: str, phrases: list[str]) -> list[str]:
    """Return all phrases found in text. Case-insensitive substring match."""
    norm = text.lower()
    return [p for p in phrases if p.lower() in norm]


def has_negation_before(text: str, concept: str, window: int = 60) -> bool:
    """Check if a negation pattern appears within `window` chars before concept in text."""
    low = text.lower()
    idx = low.find(concept.lower())
    if idx == -1:
        return False
    context = low[max(0, idx - window):idx]
    negators = [
        "does not", "do not", "not", "no", "without", "excluding",
        "doesn't", "don't", "never", "neither",
    ]
    return any(neg in context for neg in negators)
