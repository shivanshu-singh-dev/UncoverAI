"""Vendor data model and validation."""

from enum import StrEnum
from typing import Self
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SeverityLevel(StrEnum):
    """Categorical severity levels used across criticality criteria."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

    @classmethod
    def _missing_(cls, value: object) -> "SeverityLevel | None":
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        return None


class Vendor(BaseModel):
    """Internal normalized vendor model validated against domain rules."""

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="ignore",
        frozen=True,
    )

    vendor_id: str = Field(..., description="Unique alphanumeric identifier for the vendor", min_length=1)
    vendor_name: str = Field(..., description="Official trading or corporate name of the vendor", min_length=1)
    domain: str = Field(..., description="Primary website domain of the vendor", min_length=3)

    # 5 Core Criticality Criteria
    data_sensitivity: SeverityLevel = Field(
        ...,
        description="Sensitivity of accessed data (Regulated financial records, PII, IP, Auth/Security systems)",
    )
    payment_flows: SeverityLevel = Field(
        ...,
        alias="payment_involvement",
        description="Involvement in payment processing, routing, reconciliation, or monetary transactions",
    )
    regulatory_exposure: SeverityLevel = Field(
        ...,
        description="Regulatory exposure impact (SEC, FINRA, GDPR, NYDFS, PCI-DSS)",
    )
    operational_dependency: SeverityLevel = Field(
        ...,
        description="Operational dependency and disruption potential on critical paths",
    )
    customer_data_volume: SeverityLevel = Field(
        ...,
        description="Scale and volume of customer/identity records processed",
    )

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        domain = v.strip().lower()
        if domain.startswith("http://") or domain.startswith("https://"):
            # Strip protocol if accidentally included
            domain = domain.split("://", 1)[1]
        domain = domain.split("/")[0]  # strip path if included
        if "." not in domain or len(domain) < 3:
            raise ValueError(f"Invalid domain format: '{v}'")
        return domain

    @model_validator(mode="after")
    def validate_non_empty_identifiers(self) -> Self:
        if not self.vendor_id.strip():
            raise ValueError("vendor_id cannot be blank or whitespace")
        if not self.vendor_name.strip():
            raise ValueError("vendor_name cannot be blank or whitespace")
        return self
