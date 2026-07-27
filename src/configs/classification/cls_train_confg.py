r"""
Description:
    This module implements the training orchestrator component for deep learning classification pipelines. 
    It serves as a centralized factory class (`TrainerManager`) that dynamically parses system configurations, 
    resolves dataset split transformations, balances class distributions via weighted loss modules, 
    and instantiates optimization schedules and performance metrics for both multiclass and multilabel 
    experimental setups.

Main components :

    * TrainerManager: The primary factory engine responsible for constructing and assembling isolated 
      PyTorch components required by down-stream training loops.
    * ClsDataIngestion: Discovers raw file structures, handles metadata schemas, and extracts analytical 
      class balancing coefficients.
    * ClsDataTransformation: Builds MONAI-augmented training and validation dataset arrays with 
      high-performance caching.
    * MetricCollection: Aggregates evaluation metrics (Accuracy, F1-Score, Recall) into a single 
      optimized execution container.

Main features :

    * Reflection-Driven Instantiation: Dynamically resolves optimizer and learning rate scheduler classes 
      directly from the `torch.optim` registry using text-based configuration signatures.
    * Duality Optimization: Seamlessly toggles execution flows between Multiclass categorical networks 
      (Softmax dependent) and Multilabel tracking arrays (Sigmoid dependent).
    * Composite Scheduling: Automatically structures multi-stage training profiles by prepending 
      linear warmup cycles before traditional learning rate decay milestones.
    * Automatic Imbalance Compensation: Injects inverse frequency tensors into cross-entropy calculations 
      and positive weights into binary formulations to stabilize models running on unaligned datasets.

General architecture :

    The class functions as a structural mediator between the system configuration registry (`ConfigurationManager`) 
    and the actual execution trainer runtime loops. It pulls properties from the data ingestion layer to build 
    decoupled runtime resources:
    [Configuration Context] ➔ TrainerManager ➔ [Loss, Optimizers, Schedulers, Metrics, DataLoaders]

General data flow :

    .. code-block:: text 

        ┌────────────────────────────────────────────────────────┐
        │                  ConfigurationManager                  │
        └───────────────────────────┬────────────────────────────┘
                                    │ Provides configuration & metadata
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │                     TrainerManager                     │
        └───────┬───────────────┬───────────────┬───────────────┘
                │               │               │               │
                ▼               ▼               ▼               ▼
          [DataLoaders]    [Optimizer]     [Scheduler]   [Loss & Metrics]
          (Train & Val)   (Dynamic AdamW) (Warmup + LR)  (Macro Average)

Optimisations :

    * Automatic adjustment of scheduler limits (:math:`T_{max}`) to match real remaining steps, preventing 
      epoch overflow conditions when custom warmup boundaries are explicitly set.
    * Macro-averaging applied across all metric criteria to guarantee minority classes contribute 
      equally to calculated validation accuracy scores.
    * Injected label smoothing (:math:`0.1`) on multiclass arrangements to constrain logit values, preventing 
      overfitting and improving out-of-distribution robustness.

Example:

    .. code-block:: python

        from src.configs.classification.cls_config import ConfigurationManager
        from src.components.classification.trainer_manager import TrainerManager

        # Initialize global configuration workspace context
        config_mgr = ConfigurationManager(config_filepath="path/to/config.yaml")

        # Instantiate the training manager factory engine
        manager = TrainerManager(config_manager=config_mgr)

        # Resolve isolated PyTorch modules ready for training loops
        criterion, task_type = manager.get_loss()
        optimizer = manager.get_optimizer(model.parameters())
        scheduler = manager.get_scheduler(optimizer)
        metrics = manager.get_evaluation_metrics()
        train_loader, val_loader = manager.get_dataloaders()

Note:
    When setting up composite schedulers with linear warmup, the adjustments to internal parameters like 
    :math:`T_{\text{max}}` are calculated automatically using the relationship:

    .. math::
        
        T_{\text{max, adjusted}} = T_{\text{max}} - \text{warmup\_epochs}

    This calculation relies on raw values provided within the scheduler parameter blocks.

References:
    * PyTorch Optimization Registry: https://pytorch.org/docs/stable/optim.html
    * TorchMetrics Classification Suite: https://torchmetrics.readthedocs.io/
    * Google Python Documentation Style Guide: https://google.github.io/styleguide/pyguide.html

Author:
    Goudjou Borel

Version:
    1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MultilabelAccuracy,
    MultilabelF1Score,
    MulticlassRecall,
    MultilabelRecall,
)
from typing import Any, Tuple, Iterable
from src.components.classification.cls_data_ingestion import ClsDataIngestion
from src.components.classification.cls_data_transform import ClsDataTransformation
from src.configs.classification.cls_config import ConfigurationManager
from src.utils.cls_losses import MultiClassFocalLoss, MultiLabelFocalLoss


class TrainerManager:
    """A factory engine that builds and configures PyTorch components for training loops.

    Consolidates dataset orchestration, loss function selection, optimization routines,
    learning rate scheduling setups, and evaluation metrics for both multiclass and multilabel contexts.

    Attributes:
        config_manager (ConfigurationManager): System configuration context provider managing pipeline metadata access.
        config (Any): Global configuration profile mapping structural parameters across modules.
        data_ingestion (Any): Resolved ingestion data object housing file source properties.
        multiclass (bool): If True, configures standard multiclass behaviors (softmax-dependent); 
            if False, switches to a multilabel processing pipeline (sigmoid-dependent).
        img_files (list): Collection of discovered absolute or relative image file references.
        labels (list): Target ground-truth matrices or integer indices matching raw assets.
        weights (dict): Class frequency mappings used to calculate balancing parameters.
        pos_weights (list): Positive class weight coefficients utilized in multilabel cross-entropy calculations.
        number_class (int): The explicit total number of uniquely identified target classes inferred from data.
        class_label (dict): Key-value lookup dictionary mapping textual labels to clean integer indices.
    """

    def __init__(self, config_manager: ConfigurationManager) -> None:
        """Initializes dependencies and prepares structural metadata from ingested datasets.

        Args:
            config_manager (ConfigurationManager): The active global workspace configuration manager.
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.data_ingestion = config_manager.get_data_ingestion_config()
        
        cls_data_ingestion = ClsDataIngestion(self.data_ingestion)
        self.multiclass = self.data_ingestion.multiclass

        ingestion_result = cls_data_ingestion.get_files()
        self.img_files = ingestion_result.images
        self.labels = ingestion_result.labels
        self.weights = ingestion_result.weights
        self.pos_weights = ingestion_result.pos_weight
        self.number_class = len(list(self.weights.keys()))
        self.class_label = {k: idx for idx, k in enumerate(self.weights.keys())}

    def get_loss(self) -> Tuple[nn.Module, str]:
        """Create the focal loss function for the current classification task.

        Returns either a ``MultiClassFocalLoss`` or a ``MultiLabelFocalLoss``
        depending on whether the task is multiclass or multilabel. Optional
        class and positive class weights are applied according to the
        configuration.

        Returns:
            Tuple[nn.Module, str]: A tuple containing:

                * **loss** (*nn.Module*): Instantiated ``MultiClassFocalLoss`` or
                ``MultiLabelFocalLoss``.
                * **cls_type** (*str*): Classification type, either
                ``"multiclass"`` or ``"multilabel"``.
        """
        config = self.config_manager.get_loss_config()

        weight_tensor = (torch.tensor(list(self.weights.values()), dtype=torch.float32) if config.class_weight else None)

        pos_weight_tensor = (torch.tensor(self.pos_weights, dtype=torch.float32) if config.pos_weight  else None)

        if self.multiclass:
            loss = MultiClassFocalLoss(config.gamma, weight_tensor,  config.label_smoothing,  config.reduction) 
            cls_type = "multiclass"
        else:
            loss = MultiLabelFocalLoss(config.gamma, pos_weight_tensor,  weight_tensor, config.label_smoothing,  config.reduction) 
            cls_type = "multilabel"
        
        return loss, cls_type

    def get_optimizer(self, model_parameters: Iterable[nn.Parameter]) -> optim.Optimizer:
        """Constructs the optimization engine using the configuration hyperparameters.

        Inspects the runtime configuration context using reflection to identify the matching 
        PyTorch optimization implementation class, unpacking its associated custom parameter blocks.

        Args:
            model_parameters (Iterable[nn.Parameter]): Target parameter tensors requiring 
                gradient tracking updates within the training loop.

        Returns:
            optim.Optimizer: A fully initialized PyTorch optimization engine (e.g., ``optim.AdamW``).
        """
        opt_config = self.config_manager.get_optimizer_config() 
        opt_class = getattr(optim, opt_config.name)
        
        return opt_class(params=model_parameters, **opt_config.optimizer_params)

    def get_scheduler(self, optimizer: optim.Optimizer) -> Any:
        r"""Constructs the learning rate scheduler pipeline, incorporating optional warmup logic.

        If a warmup interval greater than zero is specified, this method constructs a composite 
        ``lr_scheduler.SequentialLR`` pipeline. It prepends a linear growth cycle rising from :math:`0.0`
        up to the initial learning rate limit, shifts the tracking timeline constraints of the 
        main structural policy to protect total epoch bounds via:
    
        .. math::

            T_{\text{max, adjusted}} = T_{\text{max}} - \text{warmup\_epochs}
        
        and transitions across milestones smoothly.

        Args:
            optimizer (optim.Optimizer): The active optimization instance bound to the network parameters.

        Returns:
            Any: A standard isolated PyTorch learning rate scheduler or a composite 
            ``lr_scheduler.SequentialLR`` structural pipeline container.
        """
        sched_config = self.config_manager.get_scheduler_config() 
        sched_class = getattr(lr_scheduler, sched_config.name)
        warmup_epochs = getattr(sched_config, "warmup_epochs", 0)

        if warmup_epochs > 0:
            # Set up a linear warmup phase rising from 0.0 to the initial learning rate ceiling
            warmup_lambda = lambda epoch: float(epoch) / float(max(1, warmup_epochs))
            warmup_sched = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
            
            params = dict(sched_config.scheduler_params)
            if "T_max" in params:
                params["T_max"] = max(1, params["T_max"] - warmup_epochs)
                
            main_sched = sched_class(optimizer=optimizer, **params)
            return lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup_epochs]
            )
        else:
            return sched_class(optimizer=optimizer, **sched_config.scheduler_params)

    def get_evaluation_metrics(self) -> MetricCollection:
        """Constructs classification metrics tailored to the dataset execution mode.

        Bundles performance trackers into an integrated ``MetricCollection`` container. 
        All tracking metrics are assigned a macro-averaging structure, computing statistics 
        independently per target feature channel before taking an unweighted mean. This prevents 
        frequent baseline categories from artificially masking low validation accuracies in minority classes.

        Returns:
            MetricCollection: A grouped collection of isolated TorchMetrics tracking 
            Accuracy, Macro F1-Score, and Macro Recall.
        """
        if self.multiclass:
            return MetricCollection({
                "accuracy": MulticlassAccuracy(num_classes=self.number_class, average="macro"),
                "f1_macro": MulticlassF1Score(num_classes=self.number_class, average="macro"),
                "recal_macro": MulticlassRecall(num_classes=self.number_class, average="macro")
            })
        else:
            return MetricCollection({
                "accuracy": MultilabelAccuracy(num_labels=self.number_class, average="macro"),
                "f1_macro": MultilabelF1Score(num_labels=self.number_class, average="macro"),
                "recal_macro": MultilabelRecall(num_labels=self.number_class, average="macro")
            })

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Runs image transforms and wraps target datasets in ready-to-stream PyTorch DataLoaders.

        Triggers data pipeline conversions via ``ClsDataTransformation``, building memory-cached, 
        augmented training and validation datasets. These datasets are then encapsulated in standard 
        PyTorch DataLoaders using the multi-processing and memory-pinning configurations defined 
        in the structural loader properties.

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple containing:

                * **train_loader** (*DataLoader*): The active streaming training DataLoader with shuffling enabled.
                * **valid_loader** (*DataLoader*): The active evaluation validation DataLoader with shuffling disabled.
        """
        cls_data_transformation = ClsDataTransformation(
            self.config_manager.get_data_transformation_config(), self.multiclass
        )
        train_ds, valid_ds = cls_data_transformation.transforms(self.img_files, self.labels)

        loader_config = self.config_manager.get_loader_config()
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=loader_config.batch_size, 
            shuffle=True, 
            **loader_config.loader_params
        )
        valid_loader = DataLoader(
            valid_ds, 
            batch_size=loader_config.batch_size, 
            shuffle=False, 
            **loader_config.loader_params
        )

        return train_loader, valid_loader