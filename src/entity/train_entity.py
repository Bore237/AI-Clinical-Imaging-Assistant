from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from beartype import beartype
from box import ConfigBox


@beartype
@dataclass(frozen=True)
class TrainerConfig:
    """
    Configuration of the training loop.

    Defines optimization parameters, artifact management, and runtime
    optimizations used during model training.

    Attributes:
        epochs (int): Number of training epochs.
        accumulation_steps (int): Number of steps used for gradient accumulation
            before updating model parameters.
        lr (float): Initial learning rate.
        max_grad_norm (float): Maximum gradient norm used for gradient clipping.
        artifacts_root (Path): Root directory where training artifacts are stored.
        step_freq_save (int): Frequency (in steps) at which checkpoints are saved.
        amp (str): Automatic mixed precision mode ('fp16', 'bf16', or 'no').
        compile_model (bool): Whether to compile the model to improve execution
            performance.
    """

    epochs: int
    accumulation_steps: int
    lr: float
    max_grad_norm: float
    artifacts_root: Path
    step_freq_save: int
    amp: str
    compile_model: bool


@beartype
@dataclass(frozen=True)
class LoaderConfig:
    """
    Configuration of the data loader.

    Defines parameters required to initialize the data loading pipeline.

    Attributes:
        batch_size (int): Number of samples processed in each batch.
        loader_params (dict): Additional parameters passed to the data loader.
    """

    batch_size: int
    loader_params: dict


@beartype
@dataclass(frozen=True)
class EarlyStoppingConfig:
    """
    Configuration of the early stopping mechanism.

    Controls when training should stop if the monitored metric no longer
    improves.

    Attributes:
        patience (int): Number of epochs without improvement before stopping.
        min_delta (float): Minimum improvement considered significant.
        monitor (str): Name of the metric to monitor.
        mode (Literal['min', 'max']): Optimization direction of the monitored
            metric. Use 'min' for metrics to minimize and 'max' for metrics
            to maximize.
    """

    patience: int
    min_delta: float
    monitor: str
    mode: Literal['min', 'max']


@beartype
@dataclass(frozen=True)
class OptimizerConfig:
    """
    Configuration of the optimizer.

    Defines the optimizer type and its initialization parameters.

    Attributes:
        name (str): Name of the optimizer to use.
        optimizer_params (dict): Additional parameters passed to the optimizer.
    """

    name: str
    optimizer_params: dict


@beartype
@dataclass(frozen=True)
class SchedulerConfig:
    """
    Configuration of the learning rate scheduler.

    Defines the scheduler strategy and its associated parameters.

    Attributes:
        name (str): Name of the scheduler to use.
        warmup_epochs (int): Number of warmup epochs before applying the
            scheduler policy.
        scheduler_params (dict): Additional parameters passed to the scheduler.
    """

    name: str
    warmup_epochs: int
    scheduler_params: dict


@beartype
@dataclass(frozen=True)
class WandbConfig:
    """
    Configuration for Weights & Biases experiment tracking.

    Defines the metadata required to initialize a tracking run and store
    experiment configurations.

    Attributes:
        project_name (str): Name of the Weights & Biases project.
        entity (str): User or organization owning the project.
        run_name (str): Name of the current experiment run.
        config (ConfigBox): Complete experiment configuration.
    """

    project_name: str
    entity: str
    run_name: str
    config: ConfigBox