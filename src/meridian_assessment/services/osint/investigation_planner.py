"""Main orchestration engine for the Meridian OSINT Depth Engine and Investigation Planner."""

import json
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
    CoverageState,
    CriticalityApprovalStatus,
    GateChecklistItem,
    GateStatus,
    OSINTInvestigationPlan,
    SourceCoverageRecord,
    TimeboxBudget,
    TimerState,
)
from meridian_assessment.models.vendor import Vendor
from meridian_assessment.services.osint.query_planner import OSINTQueryPlanner
from meridian_assessment.services.osint.source_registry import OSINTSourceRegistry
from meridian_assessment.services.osint.stop_conditions import StopConditionEvaluator

_DEFAULT_PROFILES_PATH = Path("config/osint_depth_profiles.yaml")
DEFAULT_OSINT_STATE_PATH = Path("data/output/osint_investigation_state.json")


class OSINTInvestigationPlanner:
    """Orchestrates vendor depth selection, source resolution, query generation, and state persistence from one unified configuration."""

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

    def resolve_effective_criticality(
        self,
        criticality_input: Optional[Union[CriticalityResult, CriticalityAssessment, str]],
        human_review: Optional[HumanReviewRecord] = None,
    ) -> tuple[str, CriticalityApprovalStatus]:
        """Alias for resolve_criticality used by UI and downstream callers."""
        return self.resolve_criticality(criticality_input, human_review)

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
        existing_plan: Optional[OSINTInvestigationPlan] = None,
    ) -> OSINTInvestigationPlan:
        """Construct a complete, deterministic OSINT Investigation Plan for a vendor.

        If `existing_plan` is provided (e.g., when an O5 human review decision changes the effective criticality),
        updates the depth profile, source requirements, queries, and planned timebox limit while preserving
        previously recorded source coverage evidence, G1–G4 gate statuses, and timer effort history.
        """
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

        # 3. Build Source Coverage Records (preserving existing evidence/progress if replanning)
        coverage_records: dict[str, SourceCoverageRecord] = {}
        for s_id in active_source_classes:
            s_def = self.source_registry.get_source(s_id)
            s_name = s_def.name if s_def else s_id
            is_req = s_id in required_sources
            is_cond = s_id in conditional_map
            cond_text = conditional_map.get(s_id, "")
            is_corrob = s_id in corroborative_sources

            prev_rec = existing_plan.coverage_records.get(s_id) if existing_plan else None
            if prev_rec is not None:
                coverage_records[s_id] = SourceCoverageRecord(
                    source_id=s_id,
                    source_name=s_name,
                    is_required=is_req,
                    is_conditional=is_cond,
                    condition=cond_text,
                    is_corroborative=is_corrob,
                    status=prev_rec.status,
                    candidate_urls_count=prev_rec.candidate_urls_count,
                    reviewed_urls_count=prev_rec.reviewed_urls_count,
                    produced_new_evidence=prev_rec.produced_new_evidence,
                    evidence_references=list(prev_rec.evidence_references),
                    notes=prev_rec.notes,
                )
            else:
                coverage_records[s_id] = SourceCoverageRecord(
                    source_id=s_id,
                    source_name=s_name,
                    is_required=is_req,
                    is_conditional=is_cond,
                    condition=cond_text,
                    is_corroborative=is_corrob,
                )

        # Also retain any previously recorded source that has non-default evidence/progress even if no longer in active_source_classes
        if existing_plan:
            for prev_sid, prev_rec in existing_plan.coverage_records.items():
                if prev_sid not in coverage_records and (
                    prev_rec.status != CoverageState.NOT_STARTED
                    or prev_rec.evidence_references
                    or prev_rec.notes
                ):
                    coverage_records[prev_sid] = SourceCoverageRecord(
                        source_id=prev_sid,
                        source_name=prev_rec.source_name,
                        is_required=False,
                        is_conditional=False,
                        condition="",
                        is_corroborative=True,
                        status=prev_rec.status,
                        candidate_urls_count=prev_rec.candidate_urls_count,
                        reviewed_urls_count=prev_rec.reviewed_urls_count,
                        produced_new_evidence=prev_rec.produced_new_evidence,
                        evidence_references=list(prev_rec.evidence_references),
                        notes=prev_rec.notes,
                    )

        # 4. Gather Discovery Resources
        associated_resources: set[str] = set()
        for s_id in active_source_classes:
            s_def = self.source_registry.get_source(s_id)
            if s_def:
                for res_id in s_def.discovery_resources:
                    associated_resources.add(res_id)

        # 5. Generate Deduplicated Queries (preserving execution metadata if replanning)
        planned_queries = self.query_planner.generate_queries_for_vendor(
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            domain=domain,
            active_source_classes=active_source_classes,
            product_name=svc_prod,
        )
        if existing_plan and existing_plan.planned_queries:
            prev_by_rendered = {
                pq.rendered_query.lower(): pq for pq in existing_plan.planned_queries
            }
            seen_rendered: set[str] = set()
            for pq in planned_queries:
                key = pq.rendered_query.lower()
                seen_rendered.add(key)
                prev_pq = prev_by_rendered.get(key)
                if prev_pq is not None:
                    pq.execution_status = prev_pq.execution_status
                    pq.provider = prev_pq.provider
                    pq.executed_at = prev_pq.executed_at
                    pq.result_count = prev_pq.result_count
                    pq.attempts = prev_pq.attempts
                    pq.error_message = prev_pq.error_message
            for prev_pq in existing_plan.planned_queries:
                if (
                    prev_pq.rendered_query.lower() not in seen_rendered
                    and prev_pq.execution_status.value != "PLANNED"
                ):
                    planned_queries.append(prev_pq)

        preserved_candidate_urls = (
            [c.model_copy(deep=True) for c in existing_plan.candidate_urls]
            if existing_plan and existing_plan.candidate_urls
            else []
        )
        preserved_collection_summary = (
            dict(existing_plan.last_collection_summary)
            if existing_plan and existing_plan.last_collection_summary
            else {}
        )

        # 6. Configure POC Timebox Budget deterministically from depth profile (preserving timer history if replanning)
        tb_cfg = profile_cfg.get("timebox", {})
        planned_mins = int(tb_cfg.get("planned_minutes", 10))
        prev_tb = existing_plan.timebox if existing_plan else None

        timebox = TimeboxBudget(
            planning_target=tb_cfg.get("planning_target", f"{planned_mins} minutes"),
            planned_minutes=planned_mins,
            elapsed_wall_clock_minutes=prev_tb.elapsed_wall_clock_minutes if prev_tb else 0.0,
            actual_analyst_effort_minutes=prev_tb.actual_analyst_effort_minutes if prev_tb else 0.0,
            timer_state=prev_tb.timer_state if prev_tb else TimerState.NOT_STARTED,
            first_started_at=prev_tb.first_started_at if prev_tb else None,
            active_segment_started_at=prev_tb.active_segment_started_at if prev_tb else None,
            last_updated_at=prev_tb.last_updated_at if prev_tb else None,
            accumulated_active_seconds=prev_tb.accumulated_active_seconds if prev_tb else 0.0,
            min_hours=float(tb_cfg.get("min_hours", round(planned_mins / 60.0, 2))),
            max_hours=float(tb_cfg.get("max_hours", round(planned_mins / 60.0, 2))),
            is_provisional=bool(tb_cfg.get("is_provisional", True)),
            actual_analyst_time_hours=prev_tb.actual_analyst_time_hours if prev_tb else 0.0,
        )
        timebox.is_exhausted = StopConditionEvaluator.is_timebox_exhausted(timebox)

        # 7. Configure Stopping Condition
        sc_cfg = profile_cfg.get("stopping_condition", {})
        stop_rule_id = sc_cfg.get("rule_id", "STOP_DEFAULT")
        stop_rule_desc = sc_cfg.get("description", "Follow standard rulebook stopping condition.")

        # 8. Setup G1–G4 Checklist (preserving existing gate statuses if replanning)
        prev_gates = {g.gate_id: g for g in existing_plan.gate_checklist} if existing_plan else {}
        gate_checklist = [
            GateChecklistItem(
                gate_id="G1",
                title="AI Operational Use",
                question="Is artificial intelligence / machine learning actually used by the vendor?",
                status=prev_gates["G1"].status if "G1" in prev_gates else GateStatus.NOT_INVESTIGATED,
                investigative_objectives=[
                    "Locate direct admissions of AI feature availability",
                    "Identify model API endpoints or client SDK integration",
                    "Distinguish marketing claims (S12) from verified technical documentation (S2)",
                ],
                qualifying_criteria="Requires Grade A or corroborated Grade B evidence confirming functional AI execution.",
                evidence_notes=prev_gates["G1"].evidence_notes if "G1" in prev_gates else "",
            ),
            GateChecklistItem(
                gate_id="G2",
                title="Service Scope Relevance",
                question="Is AI used inside the specific service/product Meridian purchases?",
                status=prev_gates["G2"].status if "G2" in prev_gates else GateStatus.NOT_INVESTIGATED,
                investigative_objectives=[
                    f"Determine if AI features apply specifically to: '{svc_prod or 'Purchased service'}'",
                    "Check if AI is optional or mandatory in contracted workflow",
                ],
                qualifying_criteria="Evidence directly linking operational AI to Meridian's contracted scope.",
                evidence_notes=prev_gates["G2"].evidence_notes if "G2" in prev_gates else "",
            ),
            GateChecklistItem(
                gate_id="G3",
                title="Customer Data Exposure",
                question="Can Meridian financial / customer data touch the AI processing path?",
                status=prev_gates["G3"].status if "G3" in prev_gates else GateStatus.NOT_INVESTIGATED,
                investigative_objectives=[
                    "Inspect Data Processing Agreements and trust centers (S1)",
                    "Verify prompt and output retention windows",
                    "Check customer-data model fine-tuning and training exclusion clauses (S3)",
                ],
                qualifying_criteria="Contractual or technical disclosure establishing data ingestion boundaries.",
                evidence_notes=prev_gates["G3"].evidence_notes if "G3" in prev_gates else "",
            ),
            GateChecklistItem(
                gate_id="G4",
                title="Supply Chain & Hyperscaler Identity",
                question="Which specific model provider, foundation model, and cloud hosting infrastructure is involved?",
                status=prev_gates["G4"].status if "G4" in prev_gates else GateStatus.NOT_INVESTIGATED,
                investigative_objectives=[
                    "Identify third-party model providers (OpenAI, Anthropic, Bedrock, Vertex AI)",
                    "Locate subprocessor register disclosures for AI hyperscalers (S1)",
                    "Determine geographical data processing boundaries",
                ],
                qualifying_criteria="Explicit subprocessor entry or verified technical architectural specification.",
                evidence_notes=prev_gates["G4"].evidence_notes if "G4" in prev_gates else "",
            ),
        ]

        # 9. Check for unresolved planning issues
        unresolved_issues: list[str] = []
        if not domain:
            unresolved_issues.append("Missing public domain — domain-restricted queries ('site:') cannot be rendered.")
        if approval_status == CriticalityApprovalStatus.PROPOSED_UNREVIEWED:
            unresolved_issues.append("Criticality used for depth planning is proposed and unreviewed by an analyst.")

        # 10. Evaluate initial or updated stop status
        if existing_plan is not None:
            stop_status, stop_notes = StopConditionEvaluator.evaluate(
                profile_name=profile_name,
                gate_checklist=gate_checklist,
                coverage_records=coverage_records,
                timebox=timebox,
            )
        else:
            stop_status = AssessmentStopStatus.IN_PROGRESS
            stop_notes = f"Investigation planned. Awaiting OSINT collection across {len(required_sources)} required source classes."

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
            candidate_urls=preserved_candidate_urls,
            last_collection_summary=preserved_collection_summary,
            timebox=timebox,
            stopping_rule_id=stop_rule_id,
            stopping_rule_description=stop_rule_desc,
            stop_status=stop_status,
            stop_progress_notes=stop_notes,
            reviewer_requirements=profile_cfg.get("review_requirements", {}),
            gate_checklist=gate_checklist,
            unresolved_planning_issues=unresolved_issues,
            generated_at=datetime.now().isoformat(),
        )

    # ─── Lightweight JSON Persistence Across Reruns, Reloads, and Restarts ───

    @staticmethod
    def save_plan_state(
        plan: OSINTInvestigationPlan,
        state_path: Path = DEFAULT_OSINT_STATE_PATH,
    ) -> None:
        """Persist an OSINTInvestigationPlan to disk so timer effort and coverage survive page reloads and app restarts."""
        state_path.parent.mkdir(parents=True, exist_ok=True)
        existing_data: dict[str, Any] = {}
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f) or {}
            except Exception:
                existing_data = {}

        plans_dict = existing_data.get("plans", {})
        plans_dict[plan.vendor_id] = plan.model_dump(mode="json")
        existing_data["plans"] = plans_dict
        existing_data["updated_at"] = datetime.now().isoformat()

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

    @staticmethod
    def load_plan_state(
        vendor_id: str,
        state_path: Path = DEFAULT_OSINT_STATE_PATH,
        pause_if_running: bool = False,
    ) -> Optional[OSINTInvestigationPlan]:
        """Load a previously persisted OSINTInvestigationPlan for a vendor from disk.

        If `pause_if_running=True` (used on cold process restart), transitions any RUNNING timer to PAUSED
        while preserving `accumulated_active_seconds` and `actual_analyst_effort_minutes` so that offline
        process downtime is neither counted as active analyst effort nor reset to zero.
        """
        if not state_path.exists():
            return None
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            raw_plan = data.get("plans", {}).get(vendor_id)
            if not raw_plan:
                return None
            plan = OSINTInvestigationPlan.model_validate(raw_plan)
            if pause_if_running and plan.timebox.timer_state == TimerState.RUNNING:
                plan.timebox.timer_state = TimerState.PAUSED
                plan.timebox.active_segment_started_at = None
            return plan
        except Exception:
            return None

    @staticmethod
    def load_all_saved_plans(
        state_path: Path = DEFAULT_OSINT_STATE_PATH,
        pause_if_running: bool = False,
    ) -> dict[str, OSINTInvestigationPlan]:
        """Load all persisted OSINTInvestigationPlans from disk."""
        if not state_path.exists():
            return {}
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            result: dict[str, OSINTInvestigationPlan] = {}
            for v_id, raw_plan in data.get("plans", {}).items():
                plan = OSINTInvestigationPlan.model_validate(raw_plan)
                if pause_if_running and plan.timebox.timer_state == TimerState.RUNNING:
                    plan.timebox.timer_state = TimerState.PAUSED
                    plan.timebox.active_segment_started_at = None
                result[v_id] = plan
            return result
        except Exception:
            return {}
