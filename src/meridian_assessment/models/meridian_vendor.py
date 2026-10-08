"""Meridian vendor input model — accepts free-text fields for the new criticality methodology."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MeridianVendor(BaseModel):
    """Vendor input record from the Meridian dataset.
    
    Accepts the actual Meridian spreadsheet column names.
    All text fields are free-text and will be processed by the criticality engine.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="ignore",
    )

    # Identity
    vendor_id: str = Field(..., description="Unique vendor identifier (e.g. V-001)")
    vendor_name: str = Field(..., description="Official vendor name")
    domain: str = Field(default="", description="Primary domain (optional for Meridian dataset)")

    # Descriptive fields (context only — not used to inflate scores)
    category: Optional[str] = Field(default=None, description="Vendor category — contextual only")
    vendor_description: Optional[str] = Field(default=None, description="General vendor description — contextual only")

    # Primary scoring inputs
    service_product_provided: Optional[str] = Field(
        default=None,
        description="Service or product provided to Meridian — primary source for P and R"
    )
    business_process_supported: Optional[str] = Field(
        default=None,
        description="Business process supported at Meridian — primary source for P and R"
    )
    data_classification_accessed: Optional[str] = Field(
        default=None,
        description="Data classification accessed — PRIMARY source for D factor"
    )
    operational_dependency: Optional[str] = Field(
        default=None,
        description="Operational dependency level — PRIMARY source for O factor (Low/Moderate/High/Critical)"
    )
    data_volume_annual: Optional[str] = Field(
        default=None,
        description="Annual data volume — PRIMARY source for V factor (free text)"
    )

    @field_validator("vendor_id")
    @classmethod
    def validate_vendor_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vendor_id cannot be blank")
        return v.strip()

    @field_validator("vendor_name")
    @classmethod
    def validate_vendor_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vendor_name cannot be blank")
        return v.strip()
