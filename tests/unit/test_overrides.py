"""Unit tests for automatic safeguards O1-O4 and O5 human governance."""
import pytest
from meridian_assessment.models.factor_result import (
    CriticalityLevel,
    DeterminationMethod,
    FactorResult,
    HumanReviewRecord,
)
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine


def make_test_vendor(
    vendor_id="V-TEST",
    vendor_name="Test Vendor",
    operational_dependency="Moderate",
    data_classification="Internal operational records",
    data_volume="N/A",
    service="IT operations",
    business_process="General support",
    description="A standard vendor",
) -> MeridianVendor:
    return MeridianVendor(
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        operational_dependency=operational_dependency,
        data_classification_accessed=data_classification,
        data_volume_annual=data_volume,
        service_product_provided=service,
        business_process_supported=business_process,
        vendor_description=description,
    )


class TestOverrideFramework:
    @pytest.fixture
    def engine(self):
        return MeridianCriticalityEngine()

    # --- O1 Tests ---
    def test_o1_triggers_on_explicit_privileged_access(self, engine):
        v = make_test_vendor(
            operational_dependency="Low",
            data_classification="No direct customer data. Privileged access to production job schedules.",
        )
        res = engine.evaluate(v)
        o1 = next(o for o in res.overrides_evaluated if o.override_id == "O1")
        assert o1.triggered is True
        # Minimum proposed floor must be High
        assert res.proposed_criticality in (CriticalityLevel.HIGH, CriticalityLevel.CRITICAL)

    def test_o1_does_not_trigger_on_normal_production_access(self, engine):
        v = make_test_vendor(
            operational_dependency="Low",
            data_classification="Standard production database access for querying user metrics.",
        )
        res = engine.evaluate(v)
        o1 = next(o for o in res.overrides_evaluated if o.override_id == "O1")
        assert o1.triggered is False

    def test_o1_does_not_trigger_on_sensitive_data_alone(self, engine):
        v = make_test_vendor(
            operational_dependency="Low",
            data_classification="Customer account numbers and balances.",
        )
        res = engine.evaluate(v)
        o1 = next(o for o in res.overrides_evaluated if o.override_id == "O1")
        assert o1.triggered is False

    # --- O2 Tests ---
    def test_o2_triggers_on_sole_source_plus_o3(self, engine):
        v = make_test_vendor(
            operational_dependency="Critical",  # O=3
            description="The only provider and sole-source supplier for core platform communications.",
        )
        res = engine.evaluate(v)
        o2 = next(o for o in res.overrides_evaluated if o.override_id == "O2")
        assert o2.triggered is True
        assert res.proposed_criticality == CriticalityLevel.CRITICAL

    def test_o2_does_not_trigger_on_sole_source_plus_o2(self, engine):
        v = make_test_vendor(
            operational_dependency="High",  # O=2
            description="The only provider and sole-source supplier for core platform communications.",
        )
        res = engine.evaluate(v)
        o2 = next(o for o in res.overrides_evaluated if o.override_id == "O2")
        assert o2.triggered is False

    def test_o2_does_not_trigger_on_o3_without_sole_source(self, engine):
        v = make_test_vendor(
            operational_dependency="Critical",  # O=3
            description="A major enterprise vendor supporting core operations.",
        )
        res = engine.evaluate(v)
        o2 = next(o for o in res.overrides_evaluated if o.override_id == "O2")
        assert o2.triggered is False

    def test_o2_unknown_sole_source_does_not_trigger(self, engine):
        v = make_test_vendor(
            operational_dependency="Critical",
            description="Widely used market leader in core infrastructure.",
        )
        res = engine.evaluate(v)
        o2 = next(o for o in res.overrides_evaluated if o.override_id == "O2")
        assert o2.triggered is False

    # --- O3 Tests ---
    def test_o3_triggers_on_d3_plus_v3(self, engine):
        v = make_test_vendor(
            operational_dependency="Low",
            data_classification="Full customer master, balances and transactions for 1.4 million customers.",  # D=3
            data_volume="240 million transactions",  # V=3
        )
        res = engine.evaluate(v)
        assert res.D.score == 3
        assert res.V.score == 3
        o3 = next(o for o in res.overrides_evaluated if o.override_id == "O3")
        assert o3.triggered is True
        assert res.proposed_criticality in (CriticalityLevel.HIGH, CriticalityLevel.CRITICAL)

    def test_o3_does_not_trigger_on_d3_plus_v2(self, engine):
        v = make_test_vendor(
            operational_dependency="Low",
            data_classification="Full customer master, balances and transactions.",  # D=3
            data_volume="2.1 million messages",  # V=2
        )
        res = engine.evaluate(v)
        o3 = next(o for o in res.overrides_evaluated if o.override_id == "O3")
        assert o3.triggered is False

    def test_o3_does_not_trigger_on_d2_plus_v3(self, engine):
        v = make_test_vendor(
            operational_dependency="Low",
            data_classification="Portfolio holdings, revenue data.",  # D=2
            data_volume="19 million documents",  # V=3
        )
        res = engine.evaluate(v)
        o3 = next(o for o in res.overrides_evaluated if o.override_id == "O3")
        assert o3.triggered is False

    def test_o3_does_not_trigger_on_d3_plus_unknown_v(self, engine):
        v = make_test_vendor(
            operational_dependency="Low",
            data_classification="Full customer master data.",  # D=3
            data_volume="12 monthly cycles",  # V=UNKNOWN
        )
        res = engine.evaluate(v)
        o3 = next(o for o in res.overrides_evaluated if o.override_id == "O3")
        assert o3.triggered is False

    # --- O4 Tests ---
    def test_o4_triggers_on_p3_plus_o2(self, engine):
        v = make_test_vendor(
            operational_dependency="High",  # O=2
            service="Real Time Payments RTP payment transmission services",  # P=3
            business_process="Payment transmission",
        )
        res = engine.evaluate(v)
        assert res.P.score == 3
        assert res.O.score == 2
        o4 = next(o for o in res.overrides_evaluated if o.override_id == "O4")
        assert o4.triggered is True
        assert res.proposed_criticality in (CriticalityLevel.HIGH, CriticalityLevel.CRITICAL)

    def test_o4_triggers_on_p3_plus_o3(self, engine):
        v = make_test_vendor(
            operational_dependency="Critical",  # O=3
            service="ACH and RTP clearing and settlement",  # P=3
            business_process="Payment clearing and settlement",
        )
        res = engine.evaluate(v)
        assert res.P.score == 3
        assert res.O.score == 3
        o4 = next(o for o in res.overrides_evaluated if o.override_id == "O4")
        assert o4.triggered is True

    def test_o4_does_not_trigger_on_p3_plus_o1(self, engine):
        v = make_test_vendor(
            operational_dependency="Moderate",  # O=1
            service="ACH and RTP clearing and settlement",  # P=3
            business_process="Payment clearing and settlement",
        )
        res = engine.evaluate(v)
        o4 = next(o for o in res.overrides_evaluated if o.override_id == "O4")
        assert o4.triggered is False

    def test_o4_does_not_trigger_on_p2_plus_o3(self, engine):
        v = make_test_vendor(
            operational_dependency="Critical",  # O=3
            service="Core banking transaction processing and account servicing",  # P=2
            business_process="Core account processing",
        )
        res = engine.evaluate(v)
        o4 = next(o for o in res.overrides_evaluated if o.override_id == "O4")
        assert o4.triggered is False

    # --- O5 Human Governance Tests ---
    def test_o5_keep_proposed_preserves_proposed_value(self, engine):
        v = make_test_vendor(
            operational_dependency="High",
            data_classification="No direct customer data. Privileged access to production credentials.",
        )
        initial = engine.evaluate(v)
        assert initial.proposed_criticality == CriticalityLevel.HIGH

        reviewed = engine.apply_human_override(
            initial,
            decision="KEEP_PROPOSED",
        )
        assert reviewed.final_criticality == CriticalityLevel.HIGH
        assert reviewed.human_review is not None
        assert reviewed.human_review.user_decision == "KEEP_PROPOSED"

    def test_o5_change_criticality_with_valid_rationale(self, engine):
        v = make_test_vendor(
            operational_dependency="High",
            data_classification="No direct customer data. Privileged access to production credentials.",
        )
        initial = engine.evaluate(v)
        assert initial.proposed_criticality == CriticalityLevel.HIGH

        # Committee escalates to Critical due to unique risk appetite
        reviewed = engine.apply_human_override(
            initial,
            decision="CHANGE_CRITICALITY",
            new_criticality=CriticalityLevel.CRITICAL,
            rationale="Escalated by Third-Party Risk Committee due to active automation modernization dependency.",
        )
        assert reviewed.final_criticality == CriticalityLevel.CRITICAL
        # Base score and underlying factors remain unaltered
        assert reviewed.base_score == initial.base_score
        assert reviewed.D.score == initial.D.score
        assert reviewed.human_review.original_criticality == CriticalityLevel.HIGH
        assert reviewed.human_review.final_criticality == CriticalityLevel.CRITICAL

    def test_o5_change_criticality_requires_rationale(self, engine):
        v = make_test_vendor()
        initial = engine.evaluate(v)
        with pytest.raises(ValueError, match="written rationale is required"):
            engine.apply_human_override(
                initial,
                decision="CHANGE_CRITICALITY",
                new_criticality=CriticalityLevel.HIGH,
                rationale="   ",  # blank
            )

    def test_o5_change_criticality_requires_controlled_selection(self, engine):
        v = make_test_vendor()
        initial = engine.evaluate(v)
        with pytest.raises(ValueError, match="Invalid criticality selection"):
            engine.apply_human_override(
                initial,
                decision="CHANGE_CRITICALITY",
                new_criticality="ArbitraryExtremeRiskLevel",
                rationale="Valid rationale.",
            )
