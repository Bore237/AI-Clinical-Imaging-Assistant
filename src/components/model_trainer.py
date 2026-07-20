import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Any, cast
from accelerate import Accelerator

from src.entity.train_entity import TrainerConfig
from src.callbacks.callbacks import TrainState, CallbackHandler, Callback, StepLogCbk


class ModelTrainer:
    """A high-level wrapper that manages distributed model training and evaluation loops.

    Leverages Hugging Face's Accelerate library to handle mixed-precision execution, 
    gradient accumulation, distributed communications, and callback scheduling seamlessly.

    Attributes:
        config (TrainerConfig): Configuration containing hyperparameters, paths, and training flags.
        accelerator (Accelerator): Hugging Face Accelerator managing distributed runtime states.
        device (torch.device): The target compute hardware device assigned by the accelerator.
        train_metrics (Dict[str, Any]): Collection of metric functions to compute during training.
        scheduler_requires_metric (bool): True if the learning rate scheduler requires a validation 
            loss score input to step (e.g., ReduceLROnPlateau).
        model (nn.Module): Prepared neural network undergoing optimization.
        optimizer (torch.optim.Optimizer): Prepared optimization manager.
        scheduler (Any): Prepared learning rate scheduling engine.
        criterion (nn.Module): Prepared objective loss function module.
        state (TrainState): Centralized training loop runtime data context shared across callbacks.
        handle_cbk (CallbackHandler): Event dispatcher executing pipeline tracking hooks.
    """

    def __init__(
        self, 
        accelerator: Accelerator,
        config: TrainerConfig, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: Any, 
        criterion: nn.Module,
        train_metrics: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """Initializes the ModelTrainer execution environment and prepares dependencies."""
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.train_metrics = train_metrics or {} 
        self.scheduler_requires_metric = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

        # Broadcast variables to targets managed by the distributed orchestration engine
        self.model, self.optimizer, self.scheduler, self.criterion = self.accelerator.prepare(
            model, optimizer, scheduler, criterion
        )
        
        self.state = TrainState(self.accelerator, self.model)
        
        # Setup and register standard callback infrastructure
        callbacks = callbacks or []
        callbacks.append(StepLogCbk())
        self.handle_cbk = CallbackHandler(callbacks)

        # Handle runtime ahead-of-time (AOT) model compilation enhancements
        if self.config.compile_model:
            if self.accelerator.is_main_process:
                self.accelerator.print("[MLOps Info] Compiling model graph via torch.compile()...")
            self.model = torch.compile(self.model)
        else:
            if self.accelerator.is_main_process:
                self.accelerator.print("[MLOps Info] Standard eager execution mode active.")
    
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Runs training forward, backward, and tracking pipelines over a single dataset epoch.

        Args:
            loader (DataLoader): The distributed-prepared training dataset loader split.
            epoch (int): Index profile value of the current active training epoch loop.

        Returns:
            Dict[str, float]: Aggregated average metric scores computed across this entire epoch.
        """
        self.model.train()
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        metric_accumulators = {name: [] for name in self.train_metrics.keys()}

        pbar = tqdm(
            enumerate(loader), 
            total=len(loader), 
            desc=f"Epoch {epoch}/{self.config.epochs}", 
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in pbar:
            with self.accelerator.accumulate(self.model):
                # Flatten 5D sequence tensors [B, N, C, H, W] to 4D batch inputs [B * N, C, H, W]
                if batch.get("image") is not None and batch["image"].dim() == 5:
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.flatten(0, 1)
                            
                images = batch["image"]
                labels = batch["label"]

                with self.accelerator.autocast():
                    preds = self.model(images)
                    loss = self.criterion(preds, labels)

                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                detached_loss = loss.detach()
                total_loss += detached_loss.float()

                # Increment global running steps continuously across training epochs
                self.state.step += 1

                if step % self.config.step_freq_save == 0:
                    current_loss_val = detached_loss.item()
                    pbar.set_postfix({"loss": f"{current_loss_val:.4f}"})

                    with torch.no_grad():
                        for name, metric_fn in self.train_metrics.items():
                            metric_accumulators[name].append(metric_fn(preds, labels))

                    # Update and dispatch structural data state updates to logging hooks
                    self.state.loss_step = current_loss_val
                    self.state.epoch = epoch
                    self.handle_cbk("on_step_end", self.state)

        # Normalize total training loss over total dataset length steps
        local_avg_loss = total_loss / len(loader)
        avg_loss = cast(torch.Tensor, self.accelerator.gather_for_metrics(local_avg_loss)).mean().item()

        # Gather and calculate average validation scores across distributed ranks
        avg_metrics: Dict[str, float] = {}
        for name, values in metric_accumulators.items():
            if values:
                tensors = [v if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device) for v in values]
                local_metric_mean = torch.stack(tensors).mean()
                avg_metrics[name] = cast(torch.Tensor, self.accelerator.gather_for_metrics(local_metric_mean)).mean().item()
        
        return {"loss": avg_loss, **avg_metrics}
    
    def train(self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None) -> None:
        """Executes the complete multi-epoch training lifecycle pipeline.

        Args:
            train_loader (DataLoader): Source data loader split containing training assets.
            valid_loader (Optional[DataLoader], optional): Source dataset loader split containing 
                validation metrics assets. Defaults to None.

        Raises:
            ValueError: If a dynamic metric-based learning rate schedule is active 
                but no validation dataloader split was provided.
        """
        train_loader = self.accelerator.prepare(train_loader)
        if valid_loader is not None:
            valid_loader = self.accelerator.prepare(valid_loader)

        self.state.valid_loader = valid_loader
        self.handle_cbk("on_train_start", self.state)

        # Loop from epoch 1 up to and including the target epoch limit count
        for epoch in range(1, self.config.epochs + 1):
            self.handle_cbk("on_epoch_start", self.state)

            epoch_train_metrics = self.train_one_epoch(train_loader, epoch)

            self.state.metrics["train"] = epoch_train_metrics
            self.state.lr = self.optimizer.param_groups[0]["lr"]
            
            # Epoch evaluation callbacks run internally during this state event hook
            self.handle_cbk("on_epoch_end", self.state)  

            # Progress the active learning rate scheduler step paths
            if self.scheduler_requires_metric:
                if valid_loader is None:
                    self.accelerator.print("[Error] This schedule requires validation data to adjust rates.")
                    raise ValueError("Provide a validation dataset loader split to execute this schedule.")
                
                # Update scheduler parameters based on validation metrics
                self.scheduler.step(self.state.metrics["val"]["loss"])
            else:
                self.scheduler.step()

            # Early termination handling triggered by callback rulesets
            if self.state.stop_training:
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"Early stopping condition triggered. Halting at epoch {epoch}.")
                break
        
        self.handle_cbk("on_train_end", self.state)
        
        # Clean up memory profiles across distributed worker states
        self.accelerator.free_memory()
        self.accelerator.end_training()
    
    def find_parameter(self, trial):
        #import optuna
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        # Entraînement du modèle ici...
        accuracy = self.train_one_epoch(lr, batch_size)

        return accuracy

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        study.best_params


