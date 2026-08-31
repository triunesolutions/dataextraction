"""Pydantic models that double as the JSON schema handed to the model."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Equipment(BaseModel):
    schedule: str = Field(description="The schedule title this row came from, e.g. 'FAN SCHEDULE'")
    tag: str = Field(description="Equipment tag / mark exactly as printed, e.g. 'RTU-7' or 'RG-1 THRU RG-7'")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer / basis-of-design brand")
    model: Optional[str] = Field(default=None, description="Model number as printed")
    size_capacity: Optional[str] = Field(default=None, description="Size or capacity, e.g. '2,400 CFM' or '7.5 ton'")


class EquipmentPage(BaseModel):
    """What the model returns for a single schedule page."""
    equipment: List[Equipment] = Field(default_factory=list)


class TeamMember(BaseModel):
    role: str = Field(
        description="Role in the project, e.g. 'Owner', 'Architect of Record', "
        "'Civil Engineer', 'Structural Engineer', 'Mechanical Engineer', 'MEP Engineer'"
    )
    firm: Optional[str] = Field(default=None, description="Company / firm name")
    address: Optional[str] = Field(default=None, description="Street address")
    city_state_zip: Optional[str] = Field(default=None, description="City, State, ZIP")
    phone: Optional[str] = None
    contact: Optional[str] = Field(default=None, description="Contact person name")
    email: Optional[str] = None


class ProjectMetadata(BaseModel):
    project_name: Optional[str] = None
    location: Optional[str] = Field(default=None, description="City/State or project location")
    address: Optional[str] = Field(default=None, description="Full project street address")
    project_number: Optional[str] = None
    drawing_date: Optional[str] = None
    revision_date: Optional[str] = None
    revision: Optional[str] = Field(default=None, description="Latest revision, e.g. 'Addendum #1' or 'Rev 4'")
    issue_status: Optional[str] = Field(default=None, description="Issue status, e.g. 'Issued for Construction'")
    team: List[TeamMember] = Field(
        default_factory=list,
        description="Project team / directory: owner, architect, civil, structural, mechanical/MEP engineers",
    )
