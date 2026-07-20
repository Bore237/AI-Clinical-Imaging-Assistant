from pathlib import Path
from typing import Union
import uuid

from src.entity.cls_entity import (
    ClsDataIngestionConfig,
    ClsModelConfig,
    ClsTransformationConfig,
)
from src.entity.train_entity import (
    EarlyStoppingConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    WandbConfig,
)
from src.utils.common import create_directories, read_yaml


class ConfigurationManager:
    """Manages system configuration parsing, workspace folder setup, and data class conversions.

    Reads incoming structural YAML configuration states and exposes them as strongly typed 
    data configuration objects tailored for specific orchestration blocks.
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        """Initializes the configuration workspace infrastructure.

        Args:
            config_path (Union[str, Path]): Path pointing to the base system configuration file.
        """
        self.config = read_yaml(Path(config_path))
        
        # Cache a single persistent UUID tag to ensure runtime tracking consistency
        self._uuid_tag = uuid.uuid4().hex[:6]
        
        # Ensure root artifact drop folders are present at boot time
        create_directories([self.config.trainer_config.artifacts_root])

    def get_data_validation_config(self) -> None:
        """Parses operational constraints for data quality validation checks.
        
        Note:
            This workflow placeholder is currently under development.
        """
        pass

    def get_data_ingestion_config(self) -> ClsDataIngestionConfig:
        """Constructs configuration data classes for the data ingestion stage.

        Returns:
            ClsDataIngestionConfig: Fully initialized ingestion parameter wrapper.
        """
        config = self.config.data_ingestion
        return ClsDataIngestionConfig(
            ext=config.ext,
            csv_col=config.csv_col,
            path_root=Path(config.path_root),
            multiclass=config.multiclass,
            paths_img=tuple(config.paths_img),
            paths_csv=tuple(config.paths_csv),
            samples_rate=tuple(config.samples_rate),
            seed=config.seed
        )
    
    def get_data_transformation_config(self) -> ClsTransformationConfig:
        """Constructs configuration containers for target augmentation and preprocessing pipelines.

        Returns:
            ClsTransformationConfig: Pipeline data transformation configurations.
        """
        config = self.config.data_transformation
        return ClsTransformationConfig(
            image_size=tuple(config.image_size),
            cache_rate=tuple(config.cache_rate),
            cache_num_workers=config.cache_num_workers,
            flip=config.flip,
            contrast=config.contrast,
            gaussian_noise=config.gaussian_noise,
            affine=config.affine,
        )
    
    def get_model_config(self) -> ClsModelConfig:
        """Extracts neural network parameter blocks and anchors a unique tracking identity tag.

        Returns:
            ClsModelConfig: Parameters configuration containing deep network metrics.
        """
        config = self.config.model_params
        return ClsModelConfig(
            dropout_rate=tuple(config.dropout_rate), 
            in_chans=config.in_chans,
            feature_head=config.feature_head,
            model_name=config.model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes, 
            uuid_tag=self._uuid_tag, 
        )
    
    def get_wandb_config(self) -> WandbConfig:
        """Extracts Weights & Biases remote logging access configurations.

        Returns:
            WandbConfig: Parameter bindings mapped to experimental log environments.
        """
        config = self.config.wandb_logging
        return WandbConfig(
            project_name=config.project_name,
            entity=config.entity,
            config=self.config, 
            run_name=config.run_name,
        )
    
    def get_trainer_config(self) -> TrainerConfig:
        """Compiles structural orchestration criteria settings to route model training engines.

        Returns:
            TrainerConfig: Compiled deep training configuration parameters.
        """
        config = self.config.trainer_config
        return TrainerConfig(
            epochs=config.epochs,
            accumulation_steps=config.accumulation_steps,
            lr=float(config.lr),
            max_grad_norm=config.max_grad_norm,
            artifacts_root=Path(config.artifacts_root),
            step_freq_save=config.step_freq_save,
            amp=config.amp,
            compile_model=config.compile_model,
        )
        
    def get_early_stopping_config(self) -> EarlyStoppingConfig:
        """Extracts performance evaluation conditions used to trigger early run termination.

        Returns:
            EarlyStopingConfig: Parametric thresholds for early stopping conditions.
        """
        config = self.config.early_stopping
        return EarlyStoppingConfig(
            patience=config.patience,
            min_delta=float(config.min_delta),
            monitor=config.monitor,
            mode=config.mode, 
        )
    
    def get_loader_config(self) -> LoaderConfig:
        """Constructs data streaming parameter settings for PyTorch DataLoader creation.

        Returns:
            LoaderConfig: Stream parsing layout settings mapping resource allocations.
        """
        config = self.config.loader
        return LoaderConfig(
            batch_size=config.batch_size,
            loader_params=config.loader_params
        )
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """Resolves target functional hyperparameter sets for optimization algorithms.

        Returns:
            OptimizerConfig: Algorithmic weight step update settings.
        """
        config = self.config.optimizer
        return OptimizerConfig(
            name=config.name,
            optimizer_params=config.optimizer_params
        )

    def get_scheduler_config(self) -> SchedulerConfig:
        """Resolves structural update constraints for learning rate scheduling systems.

        Returns:
            SchedulerConfig: Hyperparameter settings tracking dynamic schedule rates.
        """
        config = self.config.scheduler
        return SchedulerConfig(
            name=config.name,
            warmup_epochs=config.warmup_epochs,
            scheduler_params=config.scheduler_params
        )