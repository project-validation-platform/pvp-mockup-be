import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from sdv.metadata import SingleTableMetadata

from app.core.config import settings
from app.core.exceptions import DatasetNotFoundError, ModelNotFoundError, ModelTrainingError, ModelNotFittedError
from app.core.synthesizers.synthetic_data_generation_model import PATEGANSynthesizer
from app.models.dataset import ModelTrainingConfig, ModelInfo, ModelStatus
from app.services.storage_service import storage_service

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self._active_models: Dict[str, PATEGANSynthesizer] = {}
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load all existing trained models into memory"""
        try:
            for dataset_info in storage_service.list_datasets():
                dataset_id = dataset_info.dataset_id
                if storage_service.model_exists(dataset_id):
                    try:
                        model = storage_service.load_model(dataset_id)
                        self._active_models[dataset_id] = model
                        logger.info(f"Loaded existing model for dataset {dataset_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load model for dataset {dataset_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load existing models: {str(e)}")
    
    def train_model(self, dataset_id: str, config: Optional[ModelTrainingConfig] = None) -> ModelInfo:
        """Train a new PATE-GAN model for the given dataset"""
        # Validate dataset exists
        dataset_info = storage_service.get_dataset_info(dataset_id)
        
        # Use default config if none provided
        if config is None:
            config = ModelTrainingConfig()
        
        # Update model status to training
        storage_service.update_model_status(
            dataset_id, 
            ModelStatus.TRAINING,
            config=config.model_dump(),
            training_started_at=datetime.now().isoformat()
        )
        
        try:
            start_time = time.time()
            
            # Load dataset
            df = storage_service.get_dataset(dataset_id)
            
            # Create SDV metadata
            sdv_metadata = SingleTableMetadata()
            sdv_metadata.detect_from_dataframe(data=df)
            
            # Initialize synthesizer with config
            synthesizer = PATEGANSynthesizer(
                metadata=sdv_metadata,
                latent_dim=config.latent_dim,
                teacher_epochs=config.teacher_epochs,
                student_epochs=config.student_epochs,
                generator_epochs=config.generator_epochs,
                num_teachers=config.num_teachers,
                epsilon=config.epsilon,
                delta=config.delta,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                teacher_batch_size=config.teacher_batch_size,
                lambda_gradient_penalty=config.lambda_gradient_penalty,
                noise_multiplier=config.noise_multiplier,
                verbose=settings.DEBUG
            )
            
            # Train the model
            logger.info(f"Starting training for dataset {dataset_id}")
            synthesizer.fit(df)
            
            training_time = time.time() - start_time
            
            # Create model info
            model_info = ModelInfo(
                dataset_id=dataset_id,
                status=ModelStatus.TRAINED,
                config=config,
                trained_at=datetime.utcnow().isoformat(),
                training_time=training_time,
                privacy_spent=getattr(synthesizer, 'privacy_spent', None)
            )
            
            # Save model and update metadata
            storage_service.save_model(dataset_id, synthesizer, model_info)
            
            # Store in active models
            self._active_models[dataset_id] = synthesizer
            
            logger.info(f"Training completed for dataset {dataset_id} in {training_time:.2f} seconds")
            return model_info
            
        except Exception as e:
            # Update status to error
            storage_service.update_model_status(
                dataset_id,
                ModelStatus.ERROR,
                error_message=str(e),
                error_at=datetime.utcnow().isoformat()
            )
            
            logger.error(f"Training failed for dataset {dataset_id}: {str(e)}")
            raise ModelTrainingError(str(e), dataset_id)
    
    def get_model_info(self, dataset_id: str) -> ModelInfo:
        """Get model information"""
        try:
            return storage_service.get_model_info(dataset_id)
        except ModelNotFoundError:
            # Return not trained status if model doesn't exist
            return ModelInfo(
                dataset_id=dataset_id,
                status=ModelStatus.NOT_TRAINED
            )
    
    def generate_samples(self, dataset_id: str, num_rows: int, conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate synthetic samples using trained model"""
        # Check if model exists and is trained
        if dataset_id not in self._active_models:
            if not storage_service.model_exists(dataset_id):
                raise ModelNotFittedError(dataset_id)
            
            # Try to load model
            try:
                model = storage_service.load_model(dataset_id)
                self._active_models[dataset_id] = model
            except Exception as e:
                raise ModelNotFoundError(dataset_id)
        
        try:
            synthesizer = self._active_models[dataset_id]
            
            # Generate samples
            logger.info(f"Generating {num_rows} samples for dataset {dataset_id}")
            synthetic_df = synthesizer._sample(num_rows, conditions)
            
            # Convert to list of dictionaries
            samples = synthetic_df.to_dict('records')
            
            # Update privacy spent if available
            if hasattr(synthesizer, 'privacy_spent'):
                storage_service.update_model_status(
                    dataset_id,
                    ModelStatus.TRAINED,
                    privacy_spent=synthesizer.privacy_spent,
                    last_sampled_at=datetime.utcnow().isoformat()
                )
            
            logger.info(f"Generated {len(samples)} samples for dataset {dataset_id}")
            return samples
            
        except Exception as e:
            logger.error(f"Sample generation failed for dataset {dataset_id}: {str(e)}")
            raise ModelTrainingError(f"Sample generation failed: {str(e)}", dataset_id)
    
    def delete_model(self, dataset_id: str):
        """Delete trained model"""
        # Remove from active models
        if dataset_id in self._active_models:
            del self._active_models[dataset_id]
        
        # Remove from storage (handled by storage_service.delete_dataset)
        logger.info(f"Deleted model for dataset {dataset_id}")
    
    def list_models(self) -> List[ModelInfo]:
        """List all models with their status"""
        models = []
        for dataset_info in storage_service.list_datasets():
            dataset_id = dataset_info.dataset_id
            try:
                model_info = storage_service.get_model_info(dataset_id)
                models.append(model_info)
            except ModelNotFoundError:
                # Add not trained status
                models.append(ModelInfo(
                    dataset_id=dataset_id,
                    status=ModelStatus.NOT_TRAINED
                ))
        return models
    
    def get_privacy_spent(self, dataset_id: str) -> float:
        """Get current privacy budget spent for a model"""
        if dataset_id in self._active_models:
            return getattr(self._active_models[dataset_id], 'privacy_spent', 0.0)
        
        try:
            model_info = storage_service.get_model_info(dataset_id)
            return model_info.privacy_spent or 0.0
        except ModelNotFoundError:
            return 0.0

# Global model service instance
model_service = ModelService()