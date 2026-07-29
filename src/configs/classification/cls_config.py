"""
Description:
    This module implements the core configuration broker (`ConfigurationManager`) for deep 
    learning classification workflows. It reads raw hierarchical configuration specifications 
    from YAML storage layers and systematically parses, validates, and transforms them into 
    strongly-typed parameter schemas (`dataclasses`) optimized for downstream ingestion, 
    pre-processing, training, and tracking tasks.

Main components :

    * ConfigurationManager: The central coordination engine handling workspace infrastructure, 
      identity allocation, and config entity conversions.
    * ClsDataIngestionConfig, ClsTransformationConfig, ClsModelConfig: Data containers defining 
      input file topologies, structural data augmentations, and neural network attributes.
    * TrainerConfig, LoaderConfig, OptimizerConfig, SchedulerConfig, EarlyStoppingConfig: Parametric 
      schemas controlling the convergence optimization loop and data streaming performance.
    * WandbConfig: Configuration interface managing credentials and project boundaries for remote 
      cloud tracking dashboards.

Main features :

    * Strong Parameter Typing: Converts untyped dictionary representations (`ConfigBox`) into concrete, 
      read-only typed structures, eliminating key lookup errors during pipeline execution.
    * Cryptographic Experiment Lineage: Provisions and caches a unique short hexadecimal tracking 
      string signature (`_uuid_tag`) at instantiation to unify model identification across tracking layers.
    * Proactive Workspace Verification: Automatically checks for and builds required directory structures 
      on the local filesystem during the initialization phase to prevent delayed IO failures.
    * Data Normalization: Enforces safe data structures across tracking layers by parsing paths into 
      concrete `Path` objects and lists into immutable Python `tuple` targets.

General architecture :
    The manager serves as an isolated mediation layer decoupling the raw filesystem storage formats 
    from the runtime components. It centralizes parsing logic so that ingestion engines, model compilers, 
    and optimization routines remain completely agnostic of raw YAML configurations:
    [Configuration YAML on Disk] ➔ ConfigurationManager ➔ [Decoupled Strongly-Typed Dataclass Entities]

General data flow :

    .. code-block:: text 

        ┌─────────────────────────┐
        │       config.yaml       │
        └────────────┬────────────┘
                     │ read_yaml()
                     ▼
        ┌─────────────────────────┐
        │  ConfigurationManager   │
        └──────┬───────────┬──────┘
               │           ├───────────────────────────────┐
               ▼           ▼                               ▼
         [Data Configs]  [Model Config]             [Trainer Configs]
         - Ingestion     - Hyperparameters          - Optimizer / Scheduler
         - Augmentation  - uuid_tag Registration    - Early Stopping / Loaders

Optimisations :
    * Memory and Immutability Safeguards: Sequence items are explicitly cast into native Python tuples 
      rather than lists, ensuring configuration parameters cannot be mutated downstream.
    * Centralized Resource Identity: Caching the unique tracking token within the single manager instance 
      guarantees consistency across disparate logging plugins (e.g., matching local metrics with Wandb logs).
    * Early Fail Design: Instantiating structural folders at early boot ensures permissions issues or 
      path violations drop runtime execution blocks before intensive hardware resource allocations begin.

Example:

    .. code-block:: python

        from pathlib import Path
        from src.configs.classification.configuration_manager import ConfigurationManager

        # Initialize the broker with a target workspace configuration file
        config_mgr = ConfigurationManager(config_path="configs/spine_xr_config.yaml")

        # Extract specific pipeline configuration entities
        ingestion_cfg = config_mgr.get_data_ingestion_config()
        transformation_cfg = config_mgr.get_data_transformation_config()
        model_cfg = config_mgr.get_model_config()

        print(f"Initialized run with unique experiment tag: {model_cfg.uuid_tag}")

Note:
    The `get_model_config` method actively injects the cached unique tracking token (`_uuid_tag`). While 
    most returned entities map directly to values parsed from disk, the model configuration explicitly links 
    the static architecture layout to this dynamically generated runtime run identifier.

References:

    * Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
    * Python Configurations and Dataclasses: https://docs.python.org/3/library/dataclasses.html

Author:
    Goudjou Borel

Version:
    1.0.0
"""

from pathlib import Path
from typing import Union
import uuid

from src.entity.cls_entity import (
    ClsDataIngestionConfig,
    ClsModelConfig,
    ClsTransformationConfig,
    LossConfig
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

    Attributes:
        config (Any): Global hierarchical configuration box wrapping the parsed YAML file content.
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        """Initializes the configuration workspace infrastructure and registers runtime properties.

        Parses the system-level configuration parameters from disk into an attribute-accessible 
        container, registers a persistent unique tracking string signature for model lineage, 
        and verifies the workspace environment layout.

        Args:
            config_path (Union[str, Path]): Target filesystem location pointing directly 
                to the core configuration YAML asset.

        Raises:
            ValueError: If the targeted configuration file is empty or structurally corrupted.
            Exception: Re-raises any underlying system or filesystem access errors.
        """
        self.config = read_yaml(Path(config_path))
        
        # Cache a single persistent UUID tag to ensure runtime tracking consistency
        self._uuid_tag = uuid.uuid4().hex[:6]
        
        # Ensure root artifact drop folders are present at boot time
        create_directories([self.config.trainer_config.artifacts_root])

    def get_data_validation_config(self) -> None:
        """Parses operational constraints for data quality validation checks.
        
        Note:
            This workflow placeholder is currently under active development and does not 
            return or alter pipeline environments yet.
        """
        pass

    def get_data_ingestion_config(self) -> ClsDataIngestionConfig:
        """Extracts and constructs the configuration entity for the data ingestion stage.

        Ensures that relative or absolute path declarations are cast into structural 
        ``Path`` instances, and that sequence lists are converted into immutable python tuples.

        Returns:
            ClsDataIngestionConfig: Fully initialized data ingestion parameter schema mapping 
            source split paths and sample fractions.
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

        Extracts input size metrics, multi-threaded caching limits, and spatial transform dictionary 
        configurations required by MONAI or custom data transformations.

        Returns:
            ClsTransformationConfig: Pipeline data transformation configurations containing explicit 
            augmentation settings.
        """
        config = self.config.data_transformation
        return ClsTransformationConfig(
            load_image=config.load_image,
            cache_rate=tuple(config.cache_rate),
            cache_num_workers=config.cache_num_workers,
            flip=config.flip,
            contrast=config.contrast,
            gaussian_noise=config.gaussian_noise,
            affine=config.affine,
        )
    
    def get_model_config(self) -> ClsModelConfig:
        """Extracts neural network parameter blocks and anchors a unique tracking identity tag.

        Binds structural model network metrics (e.g., input channel counts, target backbone identities, 
        dropout distributions) with a localized cryptographic identifier.

        Returns:
            ClsModelConfig: Parameters configuration tracking neural architecture settings and the 
            global execution UUID tag.
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
        """Extracts Weights & Biases remote logging access and project workspace configurations.

        Returns:
            WandbConfig: Dynamic parameters mapping the systemic execution profile to cloud tracking dashboards.
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

        Unpacks convergence epoch limits, floating-point optimization metrics, mixed precision (AMP) 
        execution modes, and system artifact root path bindings.

        Returns:
            TrainerConfig: Compiled deep learning optimization framework configuration parameters.
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

        Monitors validation trajectories against specified patience bounds and significant delta improvements.

        Returns:
            EarlyStoppingConfig: Parametric thresholds determining convergence monitoring properties.
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

        Determines tensor batch aggregation sizes and passes supplementary hardware-level runtime keywords.

        Returns:
            LoaderConfig: Stream parsing layout settings mapping resource allocations.
        """
        config = self.config.loader
        return LoaderConfig(
            display=config.display,
            batch_size=config.batch_size,
            loader_params=config.loader_params
        )
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """Resolves target functional hyperparameter sets for optimization algorithms.

        Returns:
            OptimizerConfig: Weight coefficient update configurations tracking optimizer names and 
            keyword parameters.
        """
        config = self.config.optimizer
        return OptimizerConfig(
            name=config.name,
            optimizer_params=config.optimizer_params
        )

    def get_scheduler_config(self) -> SchedulerConfig:
        """Resolves structural update constraints for learning rate scheduling systems.

        Returns:
            SchedulerConfig: Hyperparameter settings tracking dynamic policy adjustments and warmups.
        """
        config = self.config.scheduler
        return SchedulerConfig(
            name=config.name,
            warmup_epochs=config.warmup_epochs,
            scheduler_params=config.scheduler_params
        )

    def get_loss_config(self) -> LossConfig:
        """Build the loss configuration.

        Reads the loss-related settings from the parsed configuration file and
        returns them as a :class:`LossConfig` instance.

        Returns:
            LossConfig: Loss configuration.
        """
        config = self.config.losses
        return LossConfig(
            gamma=config.gamma,
            class_weight=config.class_weight,
            pos_weight=config.pos_weight,
            label_smoothing=config.label_smoothing or 0.0,
            reduction=config.reduction or "mean"
        )