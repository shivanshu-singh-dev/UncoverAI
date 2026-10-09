"""Main orchestration engine for the Meridian OSINT Depth Engine and Investigation Planner."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
import yaml

from meridian_assessment.models.criticality import CriticalityAssessment, CriticalityTier
from meridian_assessment.models.factor_result import (
    CriticalityResult,
    HumanReviewRecord,
)
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.models.osint_plan import (
    AssessmentStopStatus,
    CriticalityApprovalStatus,
    GateChecklistItem,
    GateStatus,
    OSINTInvestigationPlan,
    SourceCoverageRecord,
    TimeboxBudget,
)
from meridian_assessment.models.vendor import Vendor
from meridian_assessment.services.osint.query_planner import OSINTQueryPlanner
from meridian_assessment.services.osint.source_registry import OSINTSourceRegistry

_DEFAULT_PROFILES_PATH = Path("config/osint_depth_profiles.yaml")


class OSINTInvestigationPlanner:
    """Orchestrates vendor depth selection, source resolution, and query generation from one unified configuration."""

    def __init__(
        self,
        profiles_path: Path = _DEFAULT_PROFILES_PATH,
        source_registry: Optional[OSINTSourceRegistry] = None,
        query_planner: Optional[OSINTQueryPlanner] = None,
    ) -> None:
        self.profiles_path = profiles_path
        self.source_registry = source_registry or OSINTSourceRegistry()
        self.query_planner = query_planner or OSINTQueryPlanner()
        self._profiles: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.profiles_path.exists():
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self._profiles = data.get("profiles", {})

    def resolve_criticality(
        self,
        criticality_input: Optional[Union[CriticalityResult, CriticalityAssessment, str]],
        human_review: Optional[HumanReviewRecord] = None,
    ) -> tuple[str, CriticalityApprovalStatus]:
        """Resolve effective criticality level and audit approval governance status."""
        # Case 1: Direct HumanReviewRecord supplied (O5 governance decision)
        if human_review:
            return (
                self._normalize_level_str(human_review.final_criticality),
                CriticalityApprovalStatus.APPROVED_BY_ANALYST,
            )

        # Case 2: CriticalityResult from Meridian engine
        if isinstance(criticality_input, CriticalityResult):
            if criticality_input.human_review:
                return (
                    self._normalize_level_str(criticality_input.human_review.final_criticality),
                    CriticalityApprovalStatus.APPROVED_BY_ANALYST,
                )
            if criticality_input.proposed_criticality:
                return (
                    self._normalize_level_str(criticality_input.proposed_criticality),
                    CriticalityApprovalStatus.PROPOSED_UNREVIEWED,
                )
            if criticality_input.provisional_criticality:
                return (
                    self._normalize_level_str(criticality_input.provisional_criticality),
                    CriticalityApprovalStatus.PROVISIONAL_CALCULATED,
                )

        # Case 3: Legacy CriticalityAssessment
        if isinstance(criticality_input, CriticalityAssessment):
            tier_map = {
                CriticalityTier.TIER_1: "Critical",
                CriticalityTier.TIER_2: "Medium",
                CriticalityTier.TIER_3: "Low",
            }
            return (
                tier_map.get(criticality_input.criticality_tier, "Medium"),
                CriticalityApprovalStatus.LEGACY_TIER,
            )

        # Case 4: Plain string
        if isinstance(criticality_input, str):
            norm = self._normalize_level_str(criticality_input)
            return norm, CriticalityApprovalStatus.PROVISIONAL_CALCULATED

        return "Low", CriticalityApprovalStatus.PROVISIONAL_CALCULATED

    @staticmethod
    def _normalize_level_str(val: str) -> str:
        s = str(val).strip().lower()
        if "crit" in s or "tier_1" in s or "tier 1" in s:
            return "Critical"
        if "high" in s:
            return "High"
        if "med" in s or "tier_2" in s or "tier 2" in s:
            return "Medium"
        if "low" in s or "tier_3" in s or "tier 3" in s:
            return "Low"
        return "Low"

    def plan_investigation(
        self,
        vendor: Union[MeridianVendor, Vendor, CriticalityResult, CriticalityAssessment, dict[str, Any]],
        criticality_input: Optional[Union[CriticalityResult, CriticalityAssessment, str]] = None,
        human_review: Optional[HumanReviewRecord] = None,
    ) -> OSINTInvestigationPlan:
        """Construct a complete, deterministic OSINT Investigation Plan for a vendor."""
        # Extract vendor fields safely
        if isinstance(vendor, (MeridianVendor, Vendor)):
            vendor_id = vendor.vendor_id
            vendor_name = vendor.vendor_name
            domain = vendor.domain or ""
            v_desc = getattr(vendor, "vendor_description", None)
            svc_prod = getattr(vendor, "service_product_provided", None)
            biz_proc = getattr(vendor, "business_process_supported", None)
        elif isinstance(vendor, (CriticalityResult, CriticalityAssessment)):
            vendor_id = vendor.vendor_id
            vendor_name = vendor.vendor_name
            domain = getattr(vendor, "domain", "") or ""
            v_desc = getattr(vendor, "vendor_description", None)
            svc_prod = getattr(vendor, "service_product_provided", None)
            biz_proc = getattr(vendor, "business_process_supported", None)
            if criticality_input is None:
                criticality_input = vendor
        elif isinstance(vendor, dict):
            vendor_id = vendor.get("vendor_id", "V-UNKNOWN")
            vendor_name = vendor.get("vendor_name", "Unknown Vendor")
            domain = vendor.get("domain", "") or ""
            v_desc = vendor.get("vendor_description")
            svc_prod = vendor.get("service_product_provided")
            biz_proc = vendor.get("business_process_supported")
        else:
            raise ValueError(f"Unsupported vendor type: {type(vendor)}")

        # Only look up missing fields from meridian_vendors.csv when a CriticalityResult/Assessment was passed directly
        if isinstance(vendor, (CriticalityResult, CriticalityAssessment)) and (not domain or not svc_prod):
            if Path("data/input/meridian_vendors.csv").exists():
                try:
                    from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
                    _loader = MeridianCSVVendorLoader()
                    for _v in _loader.load(Path("data/input/meridian_vendors.csv")):
                        if _v.vendor_id == vendor_id:
                            domain = domain or _v.domain or ""
                            v_desc = v_desc or _v.vendor_description
                            svc_prod = svc_prod or _v.service_product_provided
                            biz_proc = biz_proc or _v.business_process_supported
                            break
                except Exception:
                    pass

        # 1. Resolve approved / effective criticality
        crit_level, approval_status = self.resolve_criticality(criticality_input, human_review)

        # 2. Select Depth Profile from authoritative configuration
        profile_name = crit_level
        profile_cfg = self._profiles.get(profile_name, self._profiles.get("Low", {}))

        selection_reason = (
            f"Assigned '{profile_name}' OSINT depth based on {approval_status.value.lower().replace('_', ' ')} "
            f"vendor criticality level of '{crit_level}'."
        )

        required_sources: list[str] = list(profile_cfg.get("required_sources", []))
        corroborative_sources: list[str] = list(profile_cfg.get("corroborative_sources", []))
        conditional_sources: list[dict[str, str]] = list(profile_cfg.get("conditional_sources", []))
        conditional_map: dict[str, str] = {
            c["source_id"]: c.get("condition", "")
            for c in conditional_sources
            if "source_id" in c
        }

        # Active sources for query generation
        active_source_classes = list(required_sources)
        for cond_sid in conditional_map:
            if cond_sid not in active_source_classes:
                active_source_classes.append(cond_sid)
        for s in corroborative_sources:
            if s not in active_source_classes:
                active_source_classes.append(s)

        # 3. Build Source Coverage Records
        coverage_records: dict[str, SourceCoverageRecord] = {}
        for s_id in active_source_classes:
            s_def = self.source_registry.get_source(s_id)
            s_name = s_def.name if s_def else s_id
            is_req = s_id in required_sources
            is_cond = s_id in conditional_map
            cond_text = conditional_map.get(s_id, "")
            is_corrob = s_id in corroborative_sources
            coverage_records[s_id] = SourceCoverageRecord(
                source_id=s_id,
                source_name=s_name,
                is_required=is_req,
                is_conditional=is_cond,
                condition=cond_text,
                is_corroborative=is_corrob,
            )

        # 4. Gather Discovery Resources
        associated_resources: set[str] = set()
        for s_id in active_source_classes:
            s_def = self.source_registry.get_source(s_id)
            if s_def:
                for res_id in s_def.discovery_resources:
                    associated_resources.add(res_id)

        # 5. Generate Deduplicated Queries
        planned_queries = self.query_planner.generate_queries_for_vendor(
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            domain=domain,
            active_source_classes=active_source_classes,
            product_name=svc_prod,
        )

        # 6. Configure POC Timebox Budget deterministically from depth profile
        tb_cfg = profile_cfg.get("timebox", {})
        planned_mins = int(tb_cfg.get("planned_minutes", 10))
        timebox = TimeboxBudget(
            planning_target=tb_cfg.get("planning_target", f"{planned_mins} minutes"),
            planned_minutes=planned_mins,
            elapsed_wall_clock_minutes=0.0,
            actual_analyst_effort_minutes=0.0,
            min_hours=float(tb_cfg.get("min_hours", round(planned_mins / 60.0, 2))),
            max_hours=float(tb_cfg.get("max_hours", round(planned_mins / 60.0, 2))),
            is_provisional=bool(tb_cfg.get("is_provisional", True)),
        )

        # 7. Configure Stopping Condition
        sc_cfg = profile_cfg.get("stopping_condition", {})
        stop_rule_id = sc_cfg.get("rule_id", "STOP_DEFAULT")
        stop_rule_desc = sc_cfg.get("description", "Follow standard rulebook stopping condition.")

        # 8. Setup G1–G4 Checklist
        gate_checklist = [
            GateChecklistItem(
                gate_id="G1",
                title="AI Operational Use",
                question="Is artificial intelligence / machine learning actually used by the vendor?",
                status=GateStatus.NOT_INVESTIGATED,
                investigative_objectives=[
                    "Locate direct admissions of AI feature availability",
                    "Identify model API endpoints or client SDK integration",
                    "Distinguish marketing claims (S12) from verified technical documentation (S2)",
                ],
                qualifying_criteria="Requires Grade A or corroborated Grade B evidence confirming functional AI execution.",
            ),
            GateChecklistItem(
                gate_id="G2",
                title="Service Scope Relevance",
                question="Is AI used inside the specific service/product Meridian purchases?",
                status=GateStatus.NOT_INVESTIGATED,
                investigative_objectives=[
                    f"Determine if AI features apply specifically to: '{svc_prod or 'Purchased service'}'",
                    "Check if AI is optional or mandatory in contracted workflow",
                ],
                qualifying_criteria="Evidence directly linking operational AI to Meridian's contracted scope.",
            ),
            GateChecklistItem(
                gate_id="G3",
                title="Customer Data Exposure",
                question="Can Meridian financial / customer data touch the AI processing path?",
                status=GateStatus.NOT_INVESTIGATED,
                investigative_objectives=[
                    "Inspect Data Processing Agreements and trust centers (S1)",
                    "Verify prompt and output retention windows",
                    "Check customer-data model fine-tuning and training exclusion clauses (S3)",
                ],
                qualifying_criteria="Contractual or technical disclosure establishing data ingestion boundaries.",
            ),
            GateChecklistItem(
                gate_id="G4",
                title="Supply Chain & Hyperscaler Identity",
                question="Which specific model provider, foundation model, and cloud hosting infrastructure is involved?",
                status=GateStatus.NOT_INVESTIGATED,
                investigative_objectives=[
                    "Identify third-party model providers (OpenAI, Anthropic, Bedrock, Vertex AI)",
                    "Locate subprocessor register disclosures for AI hyperscalers (S1)",
                    "Determine geographical data processing boundaries",
                ],
                qualifying_criteria="Explicit subprocessor entry or verified technical architectural specification.",
            ),
        ]

        # 9. Check for unresolved planning issues
        unresolved_issues: list[str] = []
        if not domain:
            unresolved_issues.append("Missing public domain — domain-restricted queries ('site:') cannot be rendered.")
        if approval_status == CriticalityApprovalStatus.PROPOSED_UNREVIEWED:
            unresolved_issues.append("Criticality used for depth planning is proposed and unreviewed by an analyst.")

        return OSINTInvestigationPlan(
            plan_id=f"PLAN_{vendor_id}_{crit_level.upper()}",
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            domain=domain,
            vendor_description=v_desc,
            service_product_provided=svc_prod,
            business_process_supported=biz_proc,
            input_criticality=crit_level,
            approval_status=approval_status,
            selected_depth_profile=profile_name,
            selection_reason=selection_reason,
            required_source_classes=required_sources,
            corroborative_source_classes=corroborative_sources,
            conditional_source_classes=conditional_sources,
            coverage_records=coverage_records,
            associated_discovery_resources=sorted(list(associated_resources)),
            planned_queries=planned_queries,
            timebox=timebox,
            stopping_rule_id=stop_rule_id,
            stopping_rule_description=stop_rule_desc,
            stop_status=AssessmentStopStatus.IN_PROGRESS,
            stop_progress_notes=f"Investigation planned. Awaiting OSINT collection across {len(required_sources)} required source classes.",
            reviewer_requirements=profile_cfg.get("review_requirements", {}),
            gate_checklist=gate_checklist,
            unresolved_planning_issues=unresolved_issues,
            generated_at=datetime.now().isoformat(),
        )
