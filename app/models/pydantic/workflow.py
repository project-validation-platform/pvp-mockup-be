from pydantic import BaseModel, Field, EmailStr
from app.models.pydantic.base import ConfigBase
from app.models.pydantic.block import Block
from typing import Optional, Dict, Any, List
from datetime import date

class WorkflowBase(BaseModel):
    name: str
    experiment_id: Optional[str] = None

class WorkflowCreate(WorkflowBase):
    pass

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    experiment_id: Optional[str] = None

class Workflow(WorkflowBase, ConfigBase):
    workflow_id: str
    # Include all blocks, which can be ordered in the service
    blocks: List[Block] = []