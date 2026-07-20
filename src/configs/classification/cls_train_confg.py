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
)
from typing import Any, Tuple, Iterable

from src.constants import *
from src.components.classification.cls_data_ingestion import ClsDataIngestion
from src.components.classification.cls_data_transform import ClsDataTransformation
from src.configs.classification.cls_config import ConfigurationManager


class TrainerManager:
    """A factory engine that builds and configures PyTorch components for training loops.

    Consolidates dataset orchestration, loss function selection, optimization routines,
    learning rate scheduling setups, and evaluation metrics for both multiclass 
    and multilabel contexts.

    Attributes:
        config_manager (ConfigurationManager): System configuration context provider.
        config (Any): Reference profile mapping structural parameters.
        data_ingestion (Any): Parsed ingestion ruleset configurations.
        multiclass (bool): If True, configures standard multiclass behaviors; 
            if False, switches to a multilabel processing pipeline.
        img_files (list): Collection of discovered image file references.
        labels (list): Target ground-truth matrices matching raw assets.
        weights (dict): Class frequency mappings used to offset data imbalances.
        pos_weights (list): Positive class weights specifically utilized in multilabel setups.
        number_class (int): The explicit total number of target classes.
        class_label (dict): Key-value lookup dictionary mapping labels to integer indices.
    """

    def __init__(self, config_manager: ConfigurationManager) -> None:
        """Initializes dependencies and prepares structural metadata from ingested datasets."""
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
        """Resolves the objective loss function based on the active classification style.

        Returns:
            Tuple[nn.Module, str]: The un-placed loss module instance along with a 
                string label identifying the pipeline profile type ("multiclass" or "multilabel").
        """
        if self.multiclass:
            weight_tensor = torch.tensor(list(self.weights.values()), dtype=torch.float32)
            loss = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
            cls_type = "multiclass"
        else:
            pos_weight_tensor = torch.tensor(self.pos_weights, dtype=torch.float32)
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            cls_type = "multilabel"
        
        return loss, cls_type

    def get_optimizer(self, model_parameters: Iterable[nn.Parameter]) -> optim.Optimizer:
        """Constructs the optimization engine using the configuration hyperparameters.

        Args:
            model_parameters (Iterable[nn.Parameter]): Target parameter tensors requiring gradients.

        Returns:
            optim.Optimizer: Fully initialized PyTorch optimization engine.
        """
        opt_config = self.config_manager.get_optimizer_config() 
        opt_class = getattr(optim, opt_config.name)
        
        return opt_class(params=model_parameters, **opt_config.optimizer_params)

    def get_scheduler(self, optimizer: optim.Optimizer) -> Any:
        """Constructs the learning rate scheduler pipeline, incorporating optional warmup logic.

        Args:
            optimizer (optim.Optimizer): The active optimization instance bound to the network parameters.

        Returns:
            Any: A standard PyTorch learning rate scheduler or a composite SequentialLR instance.
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

        Returns:
            MetricCollection: A grouped container of TorchMetrics objects tracking accuracy and F1 scores.
        """
        if self.multiclass:
            return MetricCollection({
                "accuracy": MulticlassAccuracy(num_classes=self.number_class, average="macro"),
                "f1_macro": MulticlassF1Score(num_classes=self.number_class, average="macro")
            })
        else:
            return MetricCollection({
                "accuracy": MultilabelAccuracy(num_labels=self.number_class, average="macro"),
                "f1_macro": MultilabelF1Score(num_labels=self.number_class, average="macro")
            })

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Runs image transforms and wraps target datasets in ready-to-stream PyTorch DataLoaders.

        Returns:
            Tuple[DataLoader, DataLoader]: Prepared training and validation DataLoader objects.
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