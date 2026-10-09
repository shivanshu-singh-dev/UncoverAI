"""OSINT search query generator and deduplication engine."""

import re
from pathlib import Path
from typing import Optional
import yaml

from meridian_assessment.models.osint_plan import (
    PlannedQuery,
    SearchExecutionStatus,
)

_DEFAULT_TEMPLATES_PATH = Path("config/osint_query_templates.yaml")


class OSINTQueryPlanner:
    """Substitutes vendor context into approved query templates and deduplicates results."""

    def __init__(self, templates_path: Path = _DEFAULT_TEMPLATES_PATH) -> None:
        self.templates_path = templates_path
        self._templates: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.templates_path.exists():
            with open(self.templates_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self._templates = data.get("templates", {})

    @staticmethod
    def clean_vendor_name(vendor_name: str) -> str:
        """Produce a clean search term for the vendor while stripping corporate suffixes."""
        if not vendor_name:
            return ""
        # Strip internal parentheses e.g. "Financial Statement Services, Inc. (FSSI)" -> "FSSI" if present, or main part
        # Check for acronym in parens
        paren_match = re.search(r'\(([^)]+)\)', vendor_name)
        if paren_match:
            acronym = paren_match.group(1).strip()
            # If acronym is short (2-8 chars), it's often a great concise search term
            if 2 <= len(acronym) <= 8 and not any(ch in acronym for ch in [",", "."]):
                return acronym

        name = re.sub(r'[\'\"”″]', '', vendor_name)
        # Strip common trailing corporate entity suffixes
        suffixes = [
            r',?\s+inc\.?$',
            r',?\s+llc\.?$',
            r',?\s+l\.l\.c\.?$',
            r',?\s+corp\.?$',
            r',?\s+corporation$',
            r',?\s+co\.?$',
            r',?\s+company$',
            r',?\s+ltd\.?$',
            r',?\s+limited$',
            r',?\s+technologies$',
        ]
        cleaned = name
        for s in suffixes:
            cleaned = re.sub(s, '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def clean_domain(domain: str) -> str:
        """Strip http/https and trailing paths from domain."""
        if not domain:
            return ""
        d = domain.lower().strip()
        d = re.sub(r'^https?://', '', d)
        d = re.sub(r'/.*$', '', d)
        d = re.sub(r'^www\.', '', d)
        return d.strip()

    def generate_queries_for_vendor(
        self,
        vendor_id: str,
        vendor_name: str,
        domain: str,
        active_source_classes: list[str],
        product_name: Optional[str] = None,
        date_range: Optional[str] = None,
    ) -> list[PlannedQuery]:
        """Generate deduplicated PlannedQuery objects matching active source classes."""
        clean_v = self.clean_vendor_name(vendor_name) or vendor_name
        clean_d = self.clean_domain(domain)
        clean_p = product_name or clean_v

        # Collect raw rendered queries
        rendered_map: dict[str, dict] = {}

        # Set of active source classes
        active_sources_set = set(s.upper() for s in active_source_classes)

        for t_id, t_data in self._templates.items():
            tpl_source = t_data.get("source_class", "").upper()

            # Only include template if its source class is in active profile
            if tpl_source not in active_sources_set:
                continue

            raw_template: str = t_data.get("template", "")

            # If template requires domain and no domain provided, skip
            if "{DOMAIN}" in raw_template and not clean_d:
                continue

            # Substitutions
            rendered = raw_template.replace("{VENDOR}", clean_v)
            rendered = rendered.replace("{DOMAIN}", clean_d)
            rendered = rendered.replace("{PRODUCT}", clean_p)
            rendered = rendered.replace("{DATE_RANGE}", date_range or "")
            rendered = re.sub(r'\s+', ' ', rendered).strip()

            norm_key = rendered.lower()

            if norm_key not in rendered_map:
                rendered_map[norm_key] = {
                    "primary_id": f"PQ_{vendor_id}_{t_id}",
                    "vendor_id": vendor_id,
                    "rendered_query": rendered,
                    "template_used": raw_template,
                    "source_classes": set([tpl_source]),
                    "discovery_resources": set([t_data.get("discovery_resource", "google_search")]),
                    "investigative_objectives": [t_data.get("investigative_objective", "")],
                    "expected_evidence_targets": set(t_data.get("expected_evidence_targets", [])),
                    "date_constraints": date_range,
                }
            else:
                # Merge into existing query record
                rendered_map[norm_key]["source_classes"].add(tpl_source)
                rendered_map[norm_key]["discovery_resources"].add(t_data.get("discovery_resource", "google_search"))
                obj = t_data.get("investigative_objective", "")
                if obj and obj not in rendered_map[norm_key]["investigative_objectives"]:
                    rendered_map[norm_key]["investigative_objectives"].append(obj)
                for tgt in t_data.get("expected_evidence_targets", []):
                    rendered_map[norm_key]["expected_evidence_targets"].add(tgt)

        # Convert to PlannedQuery objects
        planned_queries: list[PlannedQuery] = []
        for item in rendered_map.values():
            planned_queries.append(
                PlannedQuery(
                    query_id=item["primary_id"],
                    vendor_id=item["vendor_id"],
                    source_classes=sorted(list(item["source_classes"])),
                    discovery_resources=sorted(list(item["discovery_resources"])),
                    investigative_objectives=item["investigative_objectives"],
                    template_used=item["template_used"],
                    rendered_query=item["rendered_query"],
                    expected_evidence_targets=sorted(list(item["expected_evidence_targets"])),
                    date_constraints=item["date_constraints"],
                    execution_status=SearchExecutionStatus.PLANNED,
                )
            )

        return planned_queries
