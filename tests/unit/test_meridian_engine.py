"""Unit tests for MeridianCriticalityEngine against the Meridian vendor dataset."""
import pytest
from pathlib import Path
from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine
from meridian_assessment.models.factor_result import CriticalityLevel


class TestMeridianEngine:
    @pytest.fixture
    def engine(self):
        return MeridianCriticalityEngine()

    @pytest.fixture
    def vendors(self):
        loader = MeridianCSVVendorLoader()
        return loader.load(Path("data/input/meridian_vendors.csv"))

    def test_vendor_dataset_loaded(self, vendors):
        assert len(vendors) == 6
        ids = [v.vendor_id for v in vendors]
        assert ids == ["V-001", "V-002", "V-003", "V-004", "V-005", "V-006"]

    def test_automworx_sanity(self, engine, vendors):
        v = next(x for x in vendors if x.vendor_id == "V-001")
        res = engine.evaluate(v)
        # O = 2 (High)
        assert res.O.score == 2
        # D = 3 (production service credentials)
        assert res.D.score == 3
        # V = 0 (Not applicable)
        assert res.V.score == 0
        # P = 0 (Batch scheduling is NOT payment)
        assert res.P.score == 0
        # O1 Safeguard triggered (privileged access to production)
        o1 = next(o for o in res.overrides_evaluated if o.override_id == "O1")
        assert o1.triggered is True
        assert res.proposed_criticality == CriticalityLevel.HIGH
        # Rationale consistency: clearly explains credentials while acknowledging no direct customer data
        assert "No direct customer data identified" in res.D.rationale
        assert "credentials" in res.D.rationale

    def test_fiserv_sanity(self, engine, vendors):
        v = next(x for x in vendors if x.vendor_id == "V-002")
        res = engine.evaluate(v)
        # O = 3 (Critical)
        assert res.O.score == 3
        # D = 3 (customer master, balances, transactions)
        assert res.D.score == 3
        # V = 3 (240 million transactions)
        assert res.V.score == 3
        # P = 2 (material transaction processing, not direct P3 clearing/settlement)
        assert res.P.score == 2
        # O3 triggered: D=3 and V=3
        o3 = next(o for o in res.overrides_evaluated if o.override_id == "O3")
        assert o3.triggered is True
        assert res.proposed_criticality == CriticalityLevel.HIGH

    def test_fssi_sanity(self, engine, vendors):
        v = next(x for x in vendors if x.vendor_id == "V-003")
        res = engine.evaluate(v)
        # O = 1 (Moderate)
        assert res.O.score == 1
        # D = 3 (entire customer base, SSNs)
        assert res.D.score == 3
        # V = 3 (19 million documents)
        assert res.V.score == 3
        # R = 2 (regulatory notice generation)
        assert res.R.score == 2
        # P = 1 (statement production without transaction processing)
        assert res.P.score == 1
        # O3 triggered (D=3 and V=3) -> High floor
        o3 = next(o for o in res.overrides_evaluated if o.override_id == "O3")
        assert o3.triggered is True
        assert res.proposed_criticality == CriticalityLevel.HIGH

    def test_terrapin_sanity(self, engine, vendors):
        v = next(x for x in vendors if x.vendor_id == "V-004")
        res = engine.evaluate(v)
        # O = 1 (Moderate)
        assert res.O.score == 1
        # D = 2 (portfolio holdings, advisor compensation)
        assert res.D.score == 2
        # V = UNKNOWN (12 monthly cycles must NOT be treated as annual volume = 12)
        assert res.V.volume_type == "FREQUENCY"
        assert res.V.is_unknown is True
        # R should not be R3
        assert res.R.score <= 1
        # Proposed criticality remains Low (provisional based on known contributions)
        assert res.proposed_criticality == CriticalityLevel.LOW

    def test_bny_sanity(self, engine, vendors):
        v = next(x for x in vendors if x.vendor_id == "V-005")
        res = engine.evaluate(v)
        # O = 2 (High)
        assert res.O.score == 2
        # D = 3 (payment instructions, account identifiers)
        assert res.D.score == 3
        # P = 3 (RTP transmission)
        assert res.P.score == 3
        # V = 2 (2.1 million messages)
        assert res.V.score == 2
        # O4 Safeguard triggered: P=3 and O=2 -> High floor
        o4 = next(o for o in res.overrides_evaluated if o.override_id == "O4")
        assert o4.triggered is True
        # O2 must NOT trigger (O is not 3, no sole source)
        o2 = next(o for o in res.overrides_evaluated if o.override_id == "O2")
        assert o2.triggered is False
        assert res.proposed_criticality == CriticalityLevel.HIGH

    def test_clearing_house_sanity(self, engine, vendors):
        v = next(x for x in vendors if x.vendor_id == "V-006")
        res = engine.evaluate(v)
        # O = 3 (Critical)
        assert res.O.score == 3
        # P = 3 (ACH and RTP clearing and settlement)
        assert res.P.score == 3
        # V = 3 (14 million messages)
        assert res.V.score == 3
        # R = 3 (Performs critical regulated function / clearing and settlement operation)
        assert res.R.score == 3
        assert "critical_regulated_function" in res.R.matched_concepts
        # O4 Safeguard triggered: P=3 and O=3
        o4 = next(o for o in res.overrides_evaluated if o.override_id == "O4")
        assert o4.triggered is True
        # Proposed criticality is High
        assert res.proposed_criticality == CriticalityLevel.HIGH
