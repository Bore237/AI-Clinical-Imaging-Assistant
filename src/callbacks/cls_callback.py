"""
Description:
    Ce module fournit les hooks opérationnels de pipeline (callbacks) conçus pour exécuter 
    l'inférence de validation sous des conditions d'exécution distribuée (DDP), appliquer 
    les frontières de décision spécifiques aux tâches (multi-classe vs multi-label), et 
    acheminer la télémétrie de performance aplatie vers divers backends de suivi (TensorBoard, 
    WandB, MLflow).

Main components:

    * LogMetricsCbk: Télémétrie middleware responsable de l'aplatissement des structures de 
      dictionnaires imbriquées et de la synchronisation des affichages sur le processus principal.
    * EvaluationCbk: Pilote de l'évaluation distribuée prenant en charge la synchronisation 
      inter-nœuds, la précision mixte automatique (autocast) et le calcul découplé des métriques.

Main features:

    * Aplatissement Dynamique: Conversion automatique des métriques groupées (ex: `{"val": {"loss": 0.2}}`) 
      en chemins d'accès normalisés (ex: `"val/loss"`) requis par les outils tiers.
    * Inférence Distribuée Sécurisée: Utilisation de `gather_for_metrics` pour centraliser les logits 
      et les étiquettes cibles sans dupliquer les résidus de fins de batchs distribués.
    * Frontières de Décision Flexibles: Routage mathématique automatique selon le type de tâche 
      médicale (Argmax pour le multi-classe, Sigmoïde avec seuil opérationnel pour le multi-label).

General architecture:
    Les sous-systèmes héritent de la classe de base `Callback` et interagissent de manière asynchrone 
    avec le contexte partagé `TrainState` à chaque fin d'époque.

General data flow:

    .. code-block:: text

        ┌────────────────────────────────────────────────────────┐
        │                 Fin d'Époque (Epoch)                   │
        └──────────────────────────┬─────────────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │                     EvaluationCbk                      │
        │  Inference DDP ➔ Autocast ➔ Gather ➔ Metrics Compute  │
        └──────────────────────────┬─────────────────────────────┘
                                   │ (Mise à jour de TrainState.metrics)
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │                     LogMetricsCbk                      │
        │  Flatten dict ➔ Add Learning Rate ➔ Accelerator.log()  │
        └────────────────────────────────────────────────────────┘

Optimisations:

    * Précision Mixte (AMP): Inférence encapsulée dans le gestionnaire de contexte `autocast()` d'Accelerate.
    * Protection Console: Utilisation de `state.accelerator.print` et restriction de la barre 
      de progression `tqdm` au processus local principal (`is_local_main_process`) pour éviter la 
      pollution des logs de grappe de calcul.

Example:

    .. code-block:: python

        from src.callbacks.logging_cbk import EvaluationCbk, LogMetricsCbk
        
        eval_cbk = EvaluationCbk(
            criterion=torch.nn.BCEWithLogitsLoss(),
            metrics_collection=my_torchmetrics_collection,
            cls_type="multilabel",
            threshold=0.50
        )
        log_cbk = LogMetricsCbk()
        
        callbacks = [eval_cbk, log_cbk]

References:

    * Hugging Face Accelerate Documentation: https://huggingface.co/docs/accelerate
    * Sphinx Napoleon Extension: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

Author:
    Goudjou Borel

Version:
    1.0.0
"""

from typing import Any, Dict, cast
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.callbacks.callbacks import Callback, TrainState


class LogMetricsCbk(Callback):
    """Flattens, routes, and displays tracked training and validation metrics every epoch.

    Acts as a telemetry middleware that transforms nested structured group maps 
    into standard flat tracking paths, appends optimization metadata such as 
    learning rates, and prints summarized performance ratios to the primary process console.
    """

    def on_epoch_end(self, state: TrainState) -> None:
        """Flattens the nested metrics dictionary structure and logs it to configured tracking backends.

        Args:
            state (TrainState): The live shared state instance context tracking current model steps.
        """
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
    r"""Executes the validation pipeline on the evaluation dataset split.

    Handles inference distribution across multi-GPU environments, manages autocast 
    precision states, synchronizes prediction and target arrays across devices, and 
    updates target metric accumulators using task-specific activation constraints.

    Mathematical Decision Boundaries:
    
        *   **Multiclass Classification**:
            Predictions are evaluated by extracting the highest activation index over the channels:
            
            .. math::

                \text{preds} = \arg\max_{c} (\text{logits}_c)
            
        *   **Multilabel Classification**:
            Predictions are converted via an element-wise Sigmoid function :math:`\sigma(x) = \frac{1}{1 + e^{-x}}` 
            and thresholded against an operational cut-off value :math:`\tau`:
            
            .. math::

                \text{preds} = \sigma(\text{logits}) > \tau

    Distributed Loss Aggregation:
        The final validation loss is reduced across all :math:`K` execution nodes and :math:`B` total batches:
        
        .. math::

            \mathcal{L}_{\text{val}} = \frac{1}{K} \sum_{k=1}^{K} \left( \frac{1}{B} \sum_{b=1}^{B} \mathcal{L}_{k, b} \right)

    Attributes:
        criterion (nn.Module): The objective loss criteria function used for evaluation.
        metrics_collection (Any): A collective metric calculator structure (e.g., ``torchmetrics.MetricCollection``).
        cls_type (str): Normalization signature tracker set to either ``"multilabel"`` or ``"multiclass"``.
        threshold (float): Decision cut-off value :math:`\tau` used for mapping probabilities to binary states.
    """

    def __init__(
        self, 
        criterion: nn.Module, 
        metrics_collection: Any, 
        cls_type: str = "multilabel", 
        threshold: float = 0.50
    ) -> None:
        """Initializes the verification callback runner.

        Args:
            criterion (nn.Module): The evaluation loss criteria function.
            metrics_collection (Any): A metric accumulator object.
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
        """Runs validation inference, gathers metrics across distributed devices, and saves the results.

        Args:
            state (TrainState): The live shared state instance context tracking current model steps.
        """
        if state.valid_loader is None:
            state.accelerator.print("[Warning] Missing validation DataLoader reference inside active TrainState.")
            return 
        
        self.metrics_collection.reset()
        device = state.accelerator.device
        model = state.model.eval()
        loader = cast(DataLoader, state.valid_loader)
        total_loss: torch.Tensor = torch.zeros((), device=device, dtype=torch.float32)
        
        # Configure standard progress bar bounded to the main local node execution track
        pbar = tqdm(
            loader, 
            total=len(loader), 
            desc=f"Validation Epoch {state.epoch}", 
            disable=not state.accelerator.is_local_main_process
        )
        
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with state.accelerator.autocast():
                logits = model(images)
                loss = self.criterion(logits, labels)
            
            pbar.set_postfix({"batch_loss": loss.item()})
            total_loss += loss.float()

            # Synchronize tensor shards across all available hardware environments
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

        # Compute metric aggregates across all parallel updates
        computed_metrics = self.metrics_collection.compute()
        results = {key: val.item() for key, val in computed_metrics.items()}
        results["loss"] = final_val_loss
        
        state.metrics["val"] = results