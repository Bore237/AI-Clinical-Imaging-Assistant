from typing import Any, Dict, cast
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.callbacks.callbacks import Callback, TrainState


class LogMetricsCbk(Callback):
    """Flattens, routes, and displays tracked training and validation metrics every epoch."""

    def on_epoch_end(self, state: TrainState) -> None:
        """Flattens the nested metrics dictionary structure and logs it to configured tracking backends."""
        # Convert nested structure (e.g., {"train": {"loss": 0.4}}) to flat keys ("train/loss")
        flattened_metrics: Dict[str, Any] = {}
        for group_name, metrics_dict in state.metrics.items():
            for metric_key, value in metrics_dict.items():
                flattened_metrics[f"{group_name}/{metric_key}"] = value

        # Inject learning rate tracking if available
        if state.lr is not None:
            flattened_metrics["train/lr_scheduler"] = state.lr

        # Ship flattened dictionary to tracking tools (e.g., TensorBoard, WandB, MLflow)
        state.accelerator.log(flattened_metrics)

        # Extract values safely using fallbacks to avoid potential KeyErrors
        train_loss = state.metrics.get("train", {}).get("loss", float("nan"))
        val_loss = state.metrics.get("val", {}).get("loss", float("nan"))

        state.accelerator.print(
            f"Epoch {state.epoch} - Train/Val Loss Ratio: {train_loss:.4f} / {val_loss:.4f}"
        )
        state.accelerator.print(f"Epoch {state.epoch} - Detailed Validation Metrics: {state.metrics.get('val', {})}")


class EvaluationCbk(Callback):
    """Executes the validation pipeline on the evaluation dataset split.

    Handles inference distribution across devices, accumulates tracking losses, 
    and handles prediction tensor conversions based on the targeted classification mode.
    """

    def __init__(self,  criterion: nn.Module,    metrics_collection: Any, 
                cls_type: str = "multilabel",    threshold: float = 0.50
    ) -> None:
        """Initializes the verification callback runner.

        Args:
            criterion (nn.Module): The evaluation loss criteria function.
            metrics_collection (Any): A metric accumulator object (e.g., torchmetrics.MetricCollection).
            cls_type (str, optional): Target task type, either "multilabel" or "multiclass". 
                Defaults to "multilabel".
            threshold (float, optional): Operational decision boundary cut-off for probability 
                activations in multilabel setups. Defaults to 0.50.
        """
        super().__init__()
        self.criterion = criterion 
        self.metrics_collection = metrics_collection
        self.cls_type = cls_type.lower()
        self.threshold = threshold

    @torch.no_grad()
    def on_epoch_end(self, state: TrainState) -> None:
        """Runs validation inference, gathers metrics across distributed devices, and saves the results."""
        if state.valid_loader is None:
            state.accelerator.print("[Warning] Missing validation DataLoader reference inside active TrainState.")
            return 
        
        self.metrics_collection.reset()
        device = state.accelerator.device
        model = state.model.eval()
        loader = cast(DataLoader, state.valid_loader)
        total_loss: torch.Tensor = torch.zeros((), device=device, dtype=torch.float32)
        pbar = tqdm(loader, total=len(loader), desc=f"Validation Epoch {state.epoch}", disable=not state.accelerator.is_local_main_process)
        
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with state.accelerator.autocast():
                logits = model(images)
                loss = self.criterion(logits, labels)
            
            pbar.set_postfix({"batch_loss": loss.item()})
            total_loss += loss.float()

            gathered_logits, gathered_labels = state.accelerator.gather_for_metrics((logits, labels))
            if self.cls_type == "multiclass":
                preds = torch.argmax(gathered_logits, dim=-1)
            else:
                preds = torch.sigmoid(gathered_logits) > self.threshold
            
            self.metrics_collection.update(preds, gathered_labels)
            
        # Standardize loss tracking across all distributed execution processes
        epoch_val_loss = total_loss / len(loader)
        gathered_loss = state.accelerator.gather_for_metrics(epoch_val_loss)
        
        if isinstance(gathered_loss, torch.Tensor):
            final_val_loss = gathered_loss.mean().item()
        else:
            final_val_loss = torch.stack(list(gathered_loss)).mean().item()

        # Compute metric aggregates across all updates
        computed_metrics = self.metrics_collection.compute()
        results = {key: val.item() for key, val in computed_metrics.items()}
        results["loss"] = final_val_loss
        
        state.metrics["val"] = results