"""Abstract base interface for vendor ingestion loaders."""

from abc import ABC, abstractmethod
from typing import Any
from meridian_assessment.models.vendor import Vendor


class BaseVendorLoader(ABC):
    """Abstract interface for loading and validating vendors from various sources (CSV, JSON, DB, API)."""

    @abstractmethod
    def load(self, source: Any) -> list[Vendor]:
        """Load, validate, and return a collection of Vendor entities.

        Args:
            source: Source identifier (file path, stream, connection string, or payload).

        Returns:
            List of validated Vendor instances.
        """
        pass
