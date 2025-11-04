from pydantic import BaseModel, Field, EmailStr
from app.models.pydantic.base import ConfigBase
from app.models.pydantic.model import Model
from app.models.pydantic.dataset import Dataset
from typing import Optional, Dict, Any, List
from datetime import date

class ExperimentBase(BaseModel):
    name: str
    user_id: str

class ExperimentCreate(ExperimentBase):
    pass

class ExperimentUpdate(BaseModel):
    name: Optional[str] = None

class Experiment(ExperimentBase, ConfigBase):
    experiment_id: str
    # These fields can be populated by your service
    models: List[Model] = []
    datasets: List[Dataset] = []