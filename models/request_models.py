from pydantic import BaseModel
from typing import Optional


class Department(BaseModel):
    id: str
    name: str


class DiagramGenerationRequest(BaseModel):
    description: str
    departments: list[Department]
    policyName: Optional[str] = None
