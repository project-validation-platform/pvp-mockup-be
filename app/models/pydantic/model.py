from pydantic import BaseModel, Field, EmailStr
from app.models.pydantic.base import ConfigBase
from typing import Optional, Dict, Any, List
from datetime import date

class ModelBase(BaseModel):
    name: str
    path: str
    dataset_id: Optional[str] = None

class ModelCreate(ModelBase):
    pass

class ModelUpdate(BaseModel):
    name: Optional[str] = None
    path: Optional[str] = None
    dataset_id: Optional[str] = None

class Model(ModelBase, ConfigBase):
    model_id: str