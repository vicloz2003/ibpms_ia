from pydantic import BaseModel
from typing import Optional


class FormField(BaseModel):
    id: str
    type: str  # TEXT, TEXTAREA, NUMBER, DATE, SELECT, FILE, SIGNATURE
    label: str
    required: bool
    options: list[str] = []


class FormSchema(BaseModel):
    fields: list[FormField] = []


class ActivityPartition(BaseModel):
    id: str
    label: str
    departmentId: str


class ActivityNode(BaseModel):
    id: str
    label: str
    partitionId: str
    type: str  # INITIAL_NODE, ACTION, DECISION, MERGE, FORK, JOIN, FLOW_FINAL, ACTIVITY_FINAL
    formSchema: FormSchema = FormSchema()
    metadata: dict = {}


class ControlFlow(BaseModel):
    id: str
    sourceNodeId: str
    targetNodeId: str
    guardCondition: Optional[str] = None


class DiagramGenerationResponse(BaseModel):
    name: str
    description: Optional[str] = None
    partitions: list[ActivityPartition]
    nodes: list[ActivityNode]
    flows: list[ControlFlow]
    suggestedBpmnXml: Optional[str] = None
