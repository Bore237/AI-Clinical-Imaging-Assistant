r"""
Description:
    Configuration data schemas and type-enforced validation models governing the 
    hyperparameter spaces for medical image classification workflows. This module leverages 
    dataclasses and runtime type enforcement to secure execution parameters.

Main components :

    * TrainerConfig: Core runtime tracking and hardware optimization parameter group.
    * LoaderConfig: Data stream batch size and worker thread configuration space.
    * EarlyStoppingConfig: Convergence validation threshold and tracking mechanics.
    * OptimizerConfig: Algorithmic optimization update weights hyperparameter schema.
    * SchedulerConfig: Learning rate adaptation curve strategy and warmup steps.
    * WandbConfig: Remote MLOps monitoring and experiment tracking access parameters.

Main features :

    * Strict runtime type validation enforced via the `@beartype` ecosystem.
    * Structural immutability secured by `frozen=True` to prevent hyperparameter tampering.
    * Native integration with nested attribute-accessible data structures (`ConfigBox`).

General architecture :
    Acts as the immutable data contract layer of the application. It receives unstructured 
    dictionary inputs from configuration streams and maps them safely into validated, 
    statically checkable data structures used across trainers and pipelines.

General data flow :

    .. code-block:: text 

        [Unstructured Config Dict] ---> ConfigurationManager ---> Type Verification
                                                                         |
                                                                         v
        [ModelTrainer / Loggers]   <--- Statically Checked <--- [Instantiated Dataclasses]

    
Optimisations :

    * Immutability ensures complete runtime integrity of hyperparameters across asynchronous workers.
    * Hardware-agnostic typing for path instances using unified `pathlib.Path` components.
    

Example:

    .. code-block:: python

        from pathlib import Path
        from src.configs.classification.cls_config_schemas import TrainerConfig

        config = TrainerConfig(
            epochs=100,
            accumulation_steps=2,
            lr=1e-4,
            max_grad_norm=1.0,
            artifacts_root=Path("artifacts/train_run"),
            step_freq_save=500,
            amp="fp16",
            compile_model=True
        )
        print(f"Target Epochs: {config.epochs}")


Note:
    Always wrap complex downstream training loops using these validated entities rather than 
    raw dictionaries to prevent silent key lookup or invalid type evaluation failures.
    

References:

    * Python PEP 557 (Dataclasses): https://peps.python.org/pep-0557/
    * Beartype Runtime Type Enforcement: https://github.com/beartype/beartype


Author:
    Goudjou Borel

Version:
    1.0.0
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from beartype import beartype
from box import ConfigBox


@beartype
@dataclass(frozen=True)
class TrainerConfig:
    """Configuration schema governing the central optimization and training loop.

    Defines execution counts, architectural runtime optimizations (such as graph compilation), 
    checkpoint saving cadences, and mixed-precision constraints to optimize hardware throughput.

    Attributes:
        epochs (int): Total number of complete iterations over the entire dataset.
        accumulation_steps (int): Number of forward/backward batches to accumulate gradients 
            before triggering an optimizer step. Useful for simulating larger batch sizes.
        lr (float): Base learning rate allocated to the parameter optimization algorithm.
        max_grad_norm (float): Upper threshold value used for gradient norm clipping to prevent 
            exploding gradients.
        artifacts_root (Path): System directory location designated for storing checkpoints, 
            telemetry reports, and model binaries.
        step_freq_save (int): Batch step interval cadence at which intermediate model states 
            are committed to disk.
        amp (str): Automated Mixed Precision hardware mode. Accepted strings are typically 
            ``'fp16'``, ``'bf16'``, or ``'no'``.
        compile_model (bool): If True, passes the neural network through ``torch.compile`` 
            to execute kernel fusion and accelerate training performance.
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
    """Configuration schema controlling data stream loaders and batch generation pipelines.

    Encapsulates memory boundaries and worker threads essential to feed the training infrastructure.

    Attributes:
        batch_size (int): Number of distinct data samples grouped together in a single forward pass.
        loader_params (dict): Supplementary dictionary configurations passed natively to the backend 
            data framework loader (e.g., ``num_workers``, ``pin_memory``, or ``drop_last``).
    """

    batch_size: int
    loader_params: dict


@beartype
@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration schema regulating the execution termination trigger mechanism.

    Controls early training exit routines when the monitored performance metric reaches 
    a plateau, optimizing compute budgets against overfitting tendencies.

    Attributes:
        patience (int): Number of successive evaluation checks allowed without any recorded 
            improvement before triggering an execution halt.
        min_delta (float): Minimum absolute metric change required to qualify as a legitimate 
            statistical improvement.
        monitor (str): Identifying string name of the specific validation metric targeted 
            for tracking (e.g., ``'val_loss'`` or ``'val_accuracy'``).
        mode (Literal['min', 'max']): Directional optimization goal. Set to ``'min'`` for 
            metrics intended to be minimized (e.g., loss), or ``'max'`` for metrics intended 
            to be maximized (e.g., F1-score).
    """

    patience: int
    min_delta: float
    monitor: str
    mode: Literal['min', 'max']


@beartype
@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration schema managing the algorithmic weight update parameters.

    Attributes:
        name (str): The identifying name of the target optimization algorithm (e.g., ``'AdamW'``, ``'SGD'``).
        optimizer_params (dict): Explicit hyperparameter values directly passed to the optimizer 
            constructor (e.g., ``weight_decay``, ``betas``, or ``momentum``).
    """

    name: str
    optimizer_params: dict


@beartype
@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration schema managing dynamic learning rate scheduling policies.

    Attributes:
        name (str): The identifying name of the target scheduling strategy (e.g., ``'CosineAnnealingLR'``).
        warmup_epochs (int): Number of initial training epochs dedicated to progressively scaling 
            the learning rate up before initiating standard decay curves.
        scheduler_params (dict): Structural dictionary payload mapping operational execution parameters 
            required by the scheduler instance.
    """

    name: str
    warmup_epochs: int
    scheduler_params: dict


@beartype
@dataclass(frozen=True)
class WandbConfig:
    """Configuration schema orchestrating external experiment telemetry tracking via MLOps pipelines.

    Houses access parameters and comprehensive execution parameters required to build remote 
    dashboards on Weights & Biases.

    Attributes:
        project_name (str): High-level operational container workspace identifier inside the WandB cloud.
        entity (str): Target user identity profile or corporate team namespace owning the destination repository.
        run_name (str): Unique descriptive label mapping to the singular live experiment iteration.
        config (ConfigBox): Complete global config dictionary wrapped in an attribute-accessible 
            container for reproducibility logging.
    """

    project_name: str
    entity: str
    run_name: str
    config: ConfigBox