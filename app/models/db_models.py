import uuid
from sqlalchemy import Column, String, Date, ForeignKey, Table
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.db.session import Base

# --- Junction Tables ---
# Define the many-to-many association tables before the models that use them

experiments_models = Table(
    'experiments_models',
    Base.metadata,
    Column('experiment_id', String, ForeignKey('experiments.experiment_id'), primary_key=True),
    Column('model_id', String, ForeignKey('models.model_id'), primary_key=True)
)

experiments_datasets = Table(
    'experiments_datasets',
    Base.metadata,
    Column('experiment_id', String, ForeignKey('experiments.experiment_id'), primary_key=True),
    Column('dataset_id', String, ForeignKey('datasets.dataset_id'), primary_key=True)
)

# --- Main Table Models ---

class User(Base):
    __tablename__ = 'users'
    
    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    dob = Column(Date)
    
    # Define the one-to-many relationship: User -> Experiment
    experiments = relationship("Experiment", back_populates="user")
    
    def __repr__(self):
        return f"<User(user_id='{self.user_id}', email='{self.email}')>"

class Experiment(Base):
    __tablename__ = 'experiments'
    
    experiment_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    
    # Define the many-to-one relationship: Experiment -> User
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    user = relationship("User", back_populates="experiments")
    
    # Define the one-to-many relationship: Experiment -> Workflow
    workflows = relationship("Workflow", back_populates="experiment")
    
    # Define the many-to-many relationships
    models = relationship("Model", secondary=experiments_models, back_populates="experiments")
    datasets = relationship("Dataset", secondary=experiments_datasets, back_populates="experiments")

    def __repr__(self):
        return f"<Experiment(experiment_id='{self.experiment_id}', name='{self.name}')>"

class Dataset(Base):
    __tablename__ = 'datasets'
    
    dataset_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), unique=True, nullable=False)
    path = Column(String, unique=True, nullable=False)
    
    # Define the one-to-many relationship: Dataset -> Model
    models = relationship("Model", back_populates="dataset")
    
    # Define the many-to-many relationship: Dataset -> Experiment
    experiments = relationship("Experiment", secondary=experiments_datasets, back_populates="datasets")

    def __repr__(self):
        return f"<Dataset(dataset_id='{self.dataset_id}', name='{self.name}')>"

class Model(Base):
    __tablename__ = 'models'
    
    model_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), unique=True, nullable=False)
    path = Column(String, unique=True, nullable=False)
    
    # Define the many-to-one relationship: Model -> Dataset
    dataset_id = Column(String, ForeignKey('datasets.dataset_id'))
    dataset = relationship("Dataset", back_populates="models")
    
    # Define the many-to-many relationship: Model -> Experiment
    experiments = relationship("Experiment", secondary=experiments_models, back_populates="models")

    def __repr__(self):
        return f"<Model(model_id='{self.model_id}', name='{self.name}')>"

class Workflow(Base):
    __tablename__ = 'workflows'
    
    workflow_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), unique=True, nullable=False)
    
    # Define the many-to-one relationship: Workflow -> Experiment
    experiment_id = Column(String, ForeignKey('experiments.experiment_id'))
    experiment = relationship("Experiment", back_populates="workflows")
    
    # Define the one-to-many relationships: Workflow -> Block and Workflow -> BlockConfig
    blocks = relationship("Block", back_populates="workflow", cascade="all, delete-orphan")
    block_configs = relationship("BlockConfig", back_populates="workflow", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Workflow(workflow_id='{self.workflow_id}', name='{self.name}')>"

class BlockConfig(Base):
    __tablename__ = 'block_configs'
    
    config_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    parameters = Column(JSONB)
    
    # Define the many-to-one relationship: BlockConfig -> Workflow
    workflow_id = Column(String, ForeignKey('workflows.workflow_id'), nullable=False)
    workflow = relationship("Workflow", back_populates="block_configs")
    
    # Define the one-to-many relationship: BlockConfig -> Block
    blocks = relationship("Block", back_populates="config")

    def __repr__(self):
        return f"<BlockConfig(config_id='{self.config_id}', name='{self.name}')>"

class Block(Base):
    __tablename__ = 'blocks'
    
    block_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    
    # Define the many-to-one relationship: Block -> Workflow
    workflow_id = Column(String, ForeignKey('workflows.workflow_id'), nullable=False)
    workflow = relationship("Workflow", back_populates="blocks")
    
    # Define the many-to-one relationship: Block -> BlockConfig
    config_id = Column(String, ForeignKey('block_configs.config_id'))
    config = relationship("BlockConfig", back_populates="blocks")
    
    # --- Self-Referential Relationships for Linked List ---
    # Store the ID of the next/prev block
    next_block_id = Column(String, ForeignKey('blocks.block_id'))
    prev_block_id = Column(String, ForeignKey('blocks.block_id'))

    # Define the relationships for next/prev
    # remote_side=[block_id] tells SQLAlchemy that block_id is the "remote" side
    # of this one-to-one relationship.
    next_block = relationship("Block", foreign_keys=[next_block_id], remote_side=[block_id], uselist=False, post_update=True)
    prev_block = relationship("Block", foreign_keys=[prev_block_id], remote_side=[block_id], uselist=False, post_update=True)

    def __repr__(self):
        return f"<Block(block_id='{self.block_id}', name='{self.name}')>"