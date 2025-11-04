from pydantic import BaseModel, Field, EmailStr
from app.models.pydantic.base import ConfigBase
from app.models.pydantic.block_config import BlockConfig
from typing import Optional, Dict, Any, List
from datetime import date

class BlockBase(BaseModel):
    name: str
    workflow_id: str
    config_id: Optional[str] = None
    next_block: Optional[str] = None
    prev_block: Optional[str] = None

class BlockCreate(BlockBase):
    pass

class BlockUpdate(BaseModel):
    name: Optional[str] = None
    config_id: Optional[str] = None
    next_block: Optional[str] = None
    prev_block: Optional[str] = None

class Block(BlockBase, ConfigBase):
    block_id: str
    # Include the full config in the response
    config: Optional[BlockConfig] = None