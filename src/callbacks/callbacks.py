from typing import Any, Dict, List, Optional
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from src.entity.train_entity import EarlyStoppingConfig

class TrainState:
    """Tracks and holds the live runtime state of a model training loop.

    Acts as a centralized data class that is continuously updated throughout 
    the training lifespan and shared across all active callbacks.

    Attributes:
        accelerator (Accelerator): Hugging Face Accelerator instance managing distributed environments.
        model (nn.Module): The neural network model undergoing optimization.
        stop_training (bool): Flag that signals the loop to halt immediately if set to True.
        epoch (int): Index of the current active training epoch.
        step (int): Cumulative global step count across all processed batches.
        loss_step (Optional[float]): The calculated loss value from the most recent training iteration.
        metrics (Dict[str, Dict[str, Any]]): Categorized dictionary holding tracking values 
            for 'train' and 'val' groups.
        valid_loader (Optional[DataLoader]): Reference to the validation dataset data loader structure.
        lr (Optional[float]): The live learning rate setting of the target optimizer.
        image (Optional[Any]): Optional reference tensor for validation visualization or tracking.
    """

    def __init__(self, accelerator: Accelerator, model: nn.Module) -> None:
        """Initializes the training loop state tracker.

        Args:
            accelerator (Accelerator): Accelerator container instance.
            model (nn.Module): The target model architecture instance.
        """
        self.accelerator = accelerator
        self.model = model
        self.stop_training: bool = False

        self.epoch: int = 0
        self.step: int = 0 
        self.loss_step: Optional[float] = None
        self.metrics: Dict[str, Dict[str, Any]] = {"train": {}, "val": {}}
        self.valid_loader: Optional[DataLoader] = None
        self.lr: Optional[float] = None

class Callback:
    """Abstract base class designed to construct hooks into the model execution pipeline."""

    def on_train_start(self, state: TrainState) -> None:
        """Triggered at the absolute beginning of the training loop initialization sequence."""
        pass

    def on_epoch_start(self, state: TrainState) -> None:
        """Triggered at the beginning of each sequential training epoch."""
        pass

    def on_step_end(self, state: TrainState) -> None:
        """Triggered immediately at the end of each optimization iteration (batch forward/backward path)."""
        pass

    def on_epoch_end(self, state: TrainState) -> None:
        """Triggered at the conclusion of each epoch immediately after the validation processes terminate."""
        pass

    def on_train_end(self, state: TrainState) -> None:
        """Triggered when the total execution loop processes conclude entirely."""
        pass


class CallbackHandler:
    """Manages collection groups of callback hooks and dispatches lifecycle event signals sequentially."""

    def __init__(self, callbacks: List[Callback]) -> None:
        """Initializes the callback management loop ecosystem.

        Args:
            callbacks (List[Callback]): Collection containing operational pipeline hooks.
        """
        self.callbacks = callbacks

    def __call__(self, event: str, state: TrainState) -> None:
        """Invokes a targeted lifecycle string event broadcast across all registered callback modules.

        Args:
            event (str): The method string key name to identify the call target (e.g., 'on_step_end').
            state (TrainState): The active shared state instance context.
        """
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn is not None:
                fn(state)


class StepLogCbk(Callback):
    """Callback wrapper focusing on recording and shipping iteration tracking values to Accelerator trackers."""

    def on_step_end(self, state: TrainState) -> None:
        """Dispatches real-time batch analytics to backend data log structures."""
        state.accelerator.log({
            "train/loss_step": state.loss_step, 
            "batch_step": state.step, 
            "epoch": state.epoch
        })


class EarlyStoppingCbk(Callback):
    """Monitors configured execution metrics to stop training runs early when progress stalls.

    Checks evaluation metrics at the close of every evaluation loop phase, tracking performance 
    ceilings to interrupt training if the threshold stalls over a series of epochs.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        """Initializes structural constraints for the early termination mechanism.

        Args:
            config (EarlyStoppingConfig): Hyperparameter object defining evaluation windows, 
                tolerance configurations, and monitoring directions.
        """
        self.patience: int = config.patience
        self.min_delta: float = config.min_delta
        self.mode: str = config.mode
        self.counter: int = 0
        
        # Determine the metric to follow, defaulting safely to "val/loss" if missing from config
        self.monitor_metric: str = getattr(config, "monitor", "loss")

        if self.mode == "min":
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best_score: float = float("inf")
        elif self.mode == "max":
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best_score: float = float("-inf")
        else:
            raise ValueError("The configuration parameter 'mode' must be specified as either 'min' or 'max'.")

    def on_epoch_end(self, state: TrainState) -> None:
        """Evaluates target validation metrics against historical benchmarks to detect plateau states."""
        current_score = state.metrics["val"].get(self.monitor_metric, None)
        
        if current_score is None:
            return 

        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

            if state.accelerator.is_main_process:
                state.accelerator.print(
                    f"--- EarlyStopping Counter updated: {self.counter}/{self.patience} "
                    f"(Best historical score: {self.best_score:.6f}) ---"
                )
                state.accelerator.log({
                    "early_stopping/counter": self.counter,
                    "early_stopping/best_score": self.best_score,
                    "early_stopping/patience": self.patience,
                })

            if self.counter >= self.patience:
                if state.accelerator.is_main_process:
                    state.accelerator.print("Early stopping criteria met. Terminating training pipeline.")
                state.stop_training = True