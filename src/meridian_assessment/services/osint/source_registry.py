"""Official S1–S12 source registry and discovery resource catalog loader."""

from pathlib import Path
from typing import Optional
import yaml

from meridian_assessment.models.osint_plan import (
    DiscoveryResource,
    SourceClassDefinition,
)

_DEFAULT_SOURCES_PATH = Path("config/osint_source_registry.yaml")
_DEFAULT_RESOURCES_PATH = Path("config/osint_resources.yaml")


class OSINTSourceRegistry:
    """Manages the official S1–S12 source registry and discovery resource catalog."""

    def __init__(
        self,
        sources_path: Path = _DEFAULT_SOURCES_PATH,
        resources_path: Path = _DEFAULT_RESOURCES_PATH,
    ) -> None:
        self.sources_path = sources_path
        self.resources_path = resources_path
        self._sources: dict[str, SourceClassDefinition] = {}
        self._resources: dict[str, DiscoveryResource] = {}
        self._load()

    def _load(self) -> None:
        if self.sources_path.exists():
            with open(self.sources_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for s_id, s_data in data.get("sources", {}).items():
                    self._sources[s_id] = SourceClassDefinition(
                        source_id=s_data.get("source_id", s_id),
                        name=s_data.get("name", ""),
                        description=s_data.get("description", ""),
                        default_evidence_grade=s_data.get("default_evidence_grade", "C"),
                        reliability_rating=s_data.get("reliability_rating", 3),
                        is_corroborative=s_data.get("is_corroborative", False),
                        source_nature=s_data.get("source_nature", "HYBRID"),
                        discovery_mode=s_data.get("discovery_mode", ""),
                        evidence_targets=s_data.get("evidence_targets", []),
                        discovery_resources=s_data.get("discovery_resources", []),
                        query_template_ids=s_data.get("query_template_ids", []),
                    )

        if self.resources_path.exists():
            with open(self.resources_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for r_id, r_data in data.get("resources", {}).items():
                    self._resources[r_id] = DiscoveryResource(
                        id=r_data.get("id", r_id),
                        name=r_data.get("name", ""),
                        category=r_data.get("category", ""),
                        associated_sources=r_data.get("associated_sources", []),
                        purpose=r_data.get("purpose", ""),
                        access_mode=r_data.get("access_mode", "UNIMPLEMENTED"),
                        limitations=r_data.get("limitations", ""),
                    )

    def get_source(self, source_id: str) -> Optional[SourceClassDefinition]:
        """Lookup an official S1–S12 source class by ID."""
        return self._sources.get(source_id.upper())

    def get_all_sources(self) -> dict[str, SourceClassDefinition]:
        """Return all registered source classes."""
        return self._sources

    def get_resource(self, resource_id: str) -> Optional[DiscoveryResource]:
        """Lookup a discovery resource by ID."""
        return self._resources.get(resource_id)

    def get_all_resources(self) -> dict[str, DiscoveryResource]:
        """Return all discovery resources."""
        return self._resources

    def get_resources_for_source(self, source_id: str) -> list[DiscoveryResource]:
        """Return all registered discovery resources associated with a source class."""
        s_id = source_id.upper()
        matching = []
        for res in self._resources.values():
            if s_id in res.associated_sources:
                matching.append(res)
        return matching
