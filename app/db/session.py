from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings 

# Create the engine
# Use 'postgresql+asyncpg' for async or 'postgresql+psycopg2' for sync
engine = create_engine(
    settings.DATABASE_URL, 
    pool_pre_ping=True
)

# Create a session factory
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

class Base(DeclarativeBase):
    pass

# Create an init function to create tables
def init_db():
    from app.models import db_models

    Base.metadata.create_all(bind=engine)

# Create a dependency for FastAPI endpoints
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()