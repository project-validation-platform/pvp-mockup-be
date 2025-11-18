from pydantic import BaseModel, Field, EmailStr
from app.models.pydantic.base import ConfigBase
from typing import Optional, Dict, Any, List
from datetime import date

class UserBase(BaseModel):
    email: EmailStr
    dob: Optional[date] = None

class UserCreate(UserBase):
    # In a real app, you'd add a password field here
    pass

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    dob: Optional[date] = None

class User(UserBase, ConfigBase):
    user_id: str