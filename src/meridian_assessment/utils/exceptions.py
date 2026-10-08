"""Custom exception hierarchy for Meridian Vendor Assessment."""


class MeridianAssessmentError(Exception):
    """Base exception for all domain errors in Meridian Assessment."""
    pass


class ConfigurationError(MeridianAssessmentError):
    """Raised when configuration file is missing, unreadable, or invalid."""
    pass


class IngestionError(MeridianAssessmentError):
    """Raised when vendor ingestion fails due to format, I/O, or parsing issues."""
    pass


class VendorValidationError(MeridianAssessmentError):
    """Raised when vendor data fails validation rules (missing fields, duplicate IDs, invalid values)."""
    pass


class CriticalityEvaluationError(MeridianAssessmentError):
    """Raised when evaluation of vendor criticality encounters unrecoverable errors."""
    pass
