from pydantic import BaseModel

# Using from_attributes=True allows the models to be created
# from SQLAlchemy ORM objects (e.g., User.model_validate(db_user))
class ConfigBase(BaseModel):
    class Config:
        from_attributes = True