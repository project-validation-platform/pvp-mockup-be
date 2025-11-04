from pydantic import BaseModel, Field, EmailStr
from app.models.pydantic.base import ConfigBase
from typing import Optional, Dict, Any, List
from datetime import date

class DatasetBase(BaseModel):
    name: str
    path: str

class DatasetCreate(DatasetBase):
    pass

class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    path: Optional[str] = None

class Dataset(DatasetBase, ConfigBase):
    dataset_id: str