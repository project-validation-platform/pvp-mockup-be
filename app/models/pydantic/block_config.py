from pydantic import BaseModel, Field, EmailStr
from app.models.pydantic.base import ConfigBase
from typing import Optional, Dict, Any, List
from datetime import date

class BlockConfigBase(BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]] = None
    workflow_id: str

class BlockConfigCreate(BlockConfigBase):
    pass

class BlockConfigUpdate(BaseModel):
    name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class BlockConfig(BlockConfigBase, ConfigBase):
    config_id: str