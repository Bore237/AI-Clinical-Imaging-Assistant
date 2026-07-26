"""
Description:
    Orchestration layer managing the end-to-end training, validation, and inference 
    lifecycles of medical image classification models. This engine wraps Hugging Face's 
    Accelerate library to enable seamless mixed-precision and distributed environment executions.

Main components :

    * ClsPipeline.__init__: Configuration parser and dataset stream allocator.
    * ClsPipeline.train: High-throughput multi-epoch deep learning optimization executor.
    * ClsPipeline.predict: Memory-safe evaluation and logit extraction matrix generator.

Main features :

    * Automatic Mixed Precision (AMP) native selection matching target runtime profiles.
    * Hugging Face Accelerate abstraction decoupling training mechanics from hardware structures.
    * Dynamic classification head channel binding synchronized with metadata class counts.
    * Multi-callback architectural pipeline (Early Stopping, Evaluation, MLOps logging).

General architecture :
    Encapsulates orchestration logic by parsing inputs through configuration managers, 
    setting up operational accelerators, routing telemetry to Weights & Biases (W&B), 
    and passing downstream control loops to a centralized ModelTrainer.

General data flow :

    .. code-block:: text 

        [Config Path] ---> ConfigurationManager ---> Extract Hyperparameters
                                                            |
                                                            v
        [DataLoader]  ---> Accelerator (AMP/DDP) <--- ClassifierModel
                                     |
                                     v
                       ModelTrainer (Callbacks Loop) ---> Weights & Biases / Checkpoints

Optimisations :

    * Device-agnostic tensor allocation switching to evaluation state (.eval()) during inference.
    * Context-managed memory safety via torch.no_grad() preventing backpropagation memory leaks.
    * Primary-process isolation (is_main_process) for intensive MLOps tracing hooks (wandb.watch).

Example:

    .. code-block:: python

        from pathlib import Path
        import torch
        from src.pipelines.classification.cls_pipeline import ClsPipeline

        pipeline = ClsPipeline(config_path="configs/spine_classification.yaml")
        
        # Run training loop
        pipeline.train()

        # Run out-of-sample inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits, targets = pipeline.predict(device=device, checkpoint_path="models/best_model.pth")

Note:
    Ensure that when executing in a distributed data-parallel environment, trackers are 
    initialized globally across all nodes while heavy gradient visualization hooks remain 
    bound solely to the master node.

References:

    * Hugging Face Accelerate Engine: https://huggingface.co/docs/accelerate
    * Weights & Biases ML Experiment Tracking: https://docs.wandb.ai/

Author:
    Goudjou Borel

Version:
    1.0.0
"""

import torch
from pathlib import Path
from tqdm.auto import tqdm
from accelerate import Accelerator
from typing import Union, Tuple, Optional

from src.configs import CONFIG_FILE_PATH
from src.callbacks.callbacks import EarlyStoppingCbk
from src.components.model_trainer import ModelTrainer
from src.callbacks.cls_callback import EvaluationCbk, LogMetricsCbk
from src.components.classification.cls_model import ClassifierModel
from src.configs.classification.cls_config import ConfigurationManager
from src.configs.classification.cls_train_confg import TrainerManager


class ClsPipeline:
    """Orchestrates the entire deep learning model optimization and evaluation lifecycle.

    This class coordinates configuration loading, data streaming generation, distributed 
    tracking registration, network initialization, and multi-callback training routines 
    using Hugging Face's Accelerate engine. It decouples high-level engineering pipelines 
    from explicit hardware declarations.

    Attributes:
        config_manager (ConfigurationManager): System instance managing parsing operations 
            for runtime and hyperparameter declarations.
        train_manager (TrainerManager): Subsystem instance responsible for allocating 
            dataloaders, optimizers, learning rate schedulers, and loss criterions.
        config (dict): Global dictionary payload mapping active structural configuration keys.
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        """Initializes the pipeline tracking managers and workspace settings.

        Args:
            config_path (Union[str, Path]): Target filesystem path location pointing to the 
                active project system YAML/JSON configuration file.
        """
        self.config_manager = ConfigurationManager(config_path=Path(config_path))
        self.train_manager = TrainerManager(config_manager=self.config_manager)
        self.config = self.config_manager.config

    def train(self) -> None:
        """Configures the distributed orchestration space and executes the training loop.

        This method coordinates the setup for multi-GPU or single-node training pipelines. 
        It configures Automated Mixed Precision (AMP), dynamically sets model target layers 
        matching target dataset dimensions, registers logging backends, and passes execution 
        control to the core structural ``ModelTrainer``.

        Note:

            * Gradient tracking hooks via ``wandb.watch`` are safely executed exclusively 
              on the main driver process (rank 0 node) to avoid tracking redundancy.
            * Early stopping, evaluation metrics, and disk telemetry logs are handled 
              via modular callbacks injected directly into the execution loop.

        Raises:
            ValueError: If configurations are incompatible or hardware assignment fails.
        """
        trainer_config = self.config_manager.get_trainer_config()
        wandb_config = self.config_manager.get_wandb_config()
        model_config = self.config_manager.get_model_config()
        early_stopping_config = self.config_manager.get_early_stopping_config()

        # Handle mixed precision states natively matching target run profiles
        amp_mode = trainer_config.amp
        accelerator = Accelerator(
            mixed_precision=amp_mode if amp_mode in ["bf16", "fp16"] else None,
            gradient_accumulation_steps=trainer_config.accumulation_steps,
            log_with="wandb"
        )

        # Synchronize dynamic dataset metadata attributes with model constraints
        model_config.num_classes = self.train_manager.number_class
        self.config["class_label_map"] = self.train_manager.class_label
        
        # Instantiate model directly onto the correct target hardware device context
        model = ClassifierModel(model_config).to(accelerator.device)
        
        # Initialize telemetry trackers on all active nodes safely
        accelerator.init_trackers(
            project_name=wandb_config.project_name,
            config=self.config, 
            init_kwargs={"wandb": {
                "entity": wandb_config.entity,
                "name": wandb_config.run_name,
                "reinit": True 
            }}
        )

        # Apply structural gradient and parameter tracking hooks on the primary worker node
        if accelerator.is_main_process:
            try:
                wandb_tracker = accelerator.get_tracker("wandb")
                wandb_tracker.run.watch(model, log="all", log_freq=100) # type: ignore
            except Exception as e:
                accelerator.print(f"[MLOps Warning] Could not hook wandb.watch onto model: {e}")

        # Resolve core training modules from the manager factory
        classification_metrics = self.train_manager.get_evaluation_metrics() 
        optimizer = self.train_manager.get_optimizer(model.parameters())
        scheduler = self.train_manager.get_scheduler(optimizer)
        criterion, cls_type = self.train_manager.get_loss()

        # Build training pipeline logging and functional callback lists
        log_metrics_cbk = LogMetricsCbk()
        early_stopping_cbk = EarlyStoppingCbk(early_stopping_config)
        evaluation_cbk = EvaluationCbk(criterion=criterion, metrics_collection=classification_metrics, cls_type=cls_type, threshold=0.50)

        # Configure the central high-level orchestration trainer wrapper
        trainer = ModelTrainer(
            accelerator=accelerator,
            config=trainer_config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_metrics=classification_metrics,
            callbacks=[evaluation_cbk, log_metrics_cbk, early_stopping_cbk] 
        )

        # Gather dataset streaming channels
        train_loader, valid_loader = self.train_manager.get_dataloaders()

        # Run multi-epoch model optimization and performance valuation phases
        trainer.train(train_loader=train_loader, valid_loader=valid_loader)

    def predict(
        self, 
        device: torch.device, 
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Executes inference predictions against targeted out-of-sample data points.

        This method routes evaluation or production targets through the optimized model state 
        to compute raw network output logits. It handles activation mappings and tensor shifts 
        across processing devices safely without leaking GPU memory.

        Mathematical Output Space:
            Depending on the configuration of your classification head, predictions :math:`\hat{y}` 
            are returned as unnormalized logit scores. Downstream consumers can apply activation 
            mappings to compute formal probabilities:

            * For Binary / Multi-label Tasks: :math:`P(y_i = 1 | x) = \sigma(\hat{y}_i)`
            * For Multiclass Tasks: :math:`P(y = c | x) = \text{softmax}(\hat{y}_c)`

        Args:
            device (torch.device): Compute target hardware infrastructure routing the inference 
                calculations (e.g., ``torch.device('cuda')`` or ``torch.device('cpu')``).
            dataloader (torch.utils.data.DataLoader, optional): Explicit data stream pipeline 
                to evaluate. If None, dynamically resolves the validation streaming channel 
                from the training manager. Defaults to None.
            checkpoint_path (Union[str, Path], optional): Specific filesystem path to a serialized 
                ``.pt`` or ``.pth`` model state dictionary checkpoint to load before inference. 
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two aggregated tensor payloads:
            
                * **all_predictions** (torch.Tensor): Continuous unnormalized network logits mapped 
                  over the entire dataset dimensions.
                * **all_targets** (torch.Tensor): Core ground truth target arrays if present in the 
                  stream context, otherwise returns an empty tensor layer.

        Example:
            >>> import torch
            >>> pipeline = ClsPipeline(config_path="config.yaml")
            >>> target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            >>> logits, targets = pipeline.predict(device=target_device, checkpoint_path="best_model.pth")
        """
        model_config = self.config_manager.get_model_config()
        model_config.num_classes = self.train_manager.number_class
        
        # Instantiate and map the computational architecture onto the target execution hardware
        model = ClassifierModel(model_config)
        
        if checkpoint_path is not None:
            state_dict = torch.load(Path(checkpoint_path), map_location=device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict)
            
        model = model.to(device)
        model.eval()

        if dataloader is None:
            _, dataloader = self.train_manager.get_dataloaders()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running Pipeline Inference", unit="batch"):
                if isinstance(batch, dict):
                    inputs = batch["image"].to(device)
                    targets = batch.get("label", None)
                else:
                    inputs, targets = batch[0].to(device), batch[1]

                outputs = model(inputs)
                
                all_predictions.append(outputs.cpu())
                if targets is not None:
                    if isinstance(targets, torch.Tensor):
                        all_targets.append(targets.cpu())
                    else:
                        all_targets.append(torch.tensor(targets).cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0)

        return all_predictions, all_targets


if __name__ == "__main__":
    pipeline = ClsPipeline(config_path=CONFIG_FILE_PATH)
    pipeline.train()