-- Table: users
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    dob DATE
);

-- Table: experiments
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    user_id TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

-- Table: datasets
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    path TEXT UNIQUE NOT NULL
);

-- Table: models
CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    path TEXT UNIQUE NOT NULL,
    dataset_id TEXT,
    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
);

-- Table: experiments_models (Junction Table)
CREATE TABLE experiments_models (
    experiment_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    PRIMARY KEY (experiment_id, model_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
    FOREIGN KEY (model_id) REFERENCES models (model_id)
);

-- Table: experiments_datasets (Junction Table)
CREATE TABLE experiments_datasets (
    experiment_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    PRIMARY KEY (experiment_id, dataset_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
);

-- Table: workflows
CREATE TABLE workflows (
    workflow_id TEXT PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- Table: experiments_workflows (Junction Table)
CREATE TABLE experiments_workflows (
    experiment_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    PRIMARY KEY (experiment_id, workflow_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
);

-- Table: block_configs
CREATE TABLE block_configs (
    config_id TEXT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    parameters JSONB, -- JSONB for flexible parameter storage
    workflow_id TEXT NOT NULL,
    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
);

-- Table: blocks
CREATE TABLE blocks (
    block_id TEXT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    next_block TEXT,
    prev_block TEXT,
    workflow_id TEXT NOT NULL,
    config_id TEXT,
    FOREIGN KEY (next_block) REFERENCES blocks (block_id),
    FOREIGN KEY (prev_block) REFERENCES blocks (block_id),
    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id),
    FOREIGN KEY (config_id) REFERENCES block_configs (config_id)
);