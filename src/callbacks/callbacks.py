"""
Description:
    Ce module fournit un écosystème de programmation événementielle complet, léger et hautement 
    extensible conçu pour orchestrer et suivre les cycles de vie des boucles d'entraînement 
    de modèles PyTorch. Tirant parti de Hugging Face Accelerate, cette infrastructure sépare 
    strictement le flux logique d'optimisation central des effets secondaires de télémétrie, 
    de journalisation et de régulation précoce de l'entraînement.

Main components :
    * TrainState : Registre de données centralisé et mutable encapsulant l'état d'exécution en temps réel.
    * Callback : Interface abstraite définissant les points d'ancrage (hooks) événementiels structurels.
    * CallbackHandler : Gestionnaire d'exécution assurant la distribution séquentielle des signaux.
    * StepLogCbk : Observateur concret dédié à la publication des métriques d'itérations.
    * EarlyStoppingCbk : Contrôleur d'arrêt précoce basé sur la stagnation de métriques cibles.

Main features :
    * Synchronisation unifiée de la télémétrie en environnement de calcul distribué (DDP, DeepSpeed).
    * Découplage architectural strict via le patron de conception Observateur (Observer Pattern).
    * Gestion native bidirectionnelle des objectifs d'optimisation (minimisation de perte ou maximisation de score).
    * Introspection dynamique (reflection) limitant les structures conditionnelles redondantes.

General architecture :
    Le système s'articule autour d'un modèle d'écoute événementielle. La boucle d'entraînement principale 
    agit en tant qu'émetteur en notifiant le `CallbackHandler` lors des transitions clés de son cycle de vie. 
    Le `CallbackHandler` inspecte dynamiquement les instances de `Callback` via introspection (`getattr`) et 
    exécute les hooks correspondants. L'ensemble des callbacks partagent une référence unique vers l'instance 
    `TrainState`, ce qui permet des mutations coordonnées et sécurisées de l'état global.

General data flow :
    .. code-block:: text 

        [Boucle d'entraînement] ─── Exécute un événement (ex: on_step_end) ───► [CallbackHandler]
                                                                                        │
                                                                               Dispatch séquentiel
                                                                                        │
                                                                                        ▼
        [TrainState] ◄─── Lit & modifie le contexte global (loss, métriques) ─── [Callbacks Concrets]

Optimisations :
    * Résolution des hooks par chaîne de caractères via `getattr` évitant le surcoût de structures `if/else` imbriquées.
    * Instanciation à chaud d'opérateurs lambda d'évaluation (`monitor_op`) éliminant les aiguillages logiques à chaque pas.
    * Contrôle de flux centralisé sur le processus maître (`is_main_process`) limitant les opérations d'I/O concurrentes en contexte distribué.

Example:
    .. code-block:: python

        from accelerate import Accelerator
        from torch import nn
        from src.entity.train_entity import EarlyStoppingConfig

        # 1. Instanciation des dépendances de base
        accelerator = Accelerator()
        model = nn.Linear(10, 2)
        config = EarlyStoppingConfig(patience=3, min_delta=1e-4, mode="min", monitor="loss")

        # 2. Initialisation du gestionnaire et de l'état
        state = TrainState(accelerator, model)
        handler = CallbackHandler([StepLogCbk(), EarlyStoppingCbk(config)])

        # 3. Intégration dans la boucle d'entraînement
        handler("on_train_start", state)
        for epoch in range(5):
            state.epoch = epoch
            handler("on_epoch_start", state)
            
            # Simulation d'un batch d'entraînement
            state.step += 1
            state.loss_step = 0.314
            handler("on_step_end", state)
            
            # Simulation de la phase de validation
            state.metrics["val"]["loss"] = 0.285
            handler("on_epoch_end", state)
            
            if state.stop_training:
                break
                
        handler("on_train_end", state)

Note:
    Pour garantir l'efficacité opérationnelle du module `EarlyStoppingCbk`, le dictionnaire imbriqué 
    `state.metrics["val"]` doit obligatoirement être alimenté avec une clé textuelle correspondant 
    exactement à la valeur configurée dans l'attribut `monitor` de l'entité de configuration.

References:
    * Hugging Face Accelerate Lifecycle Design: https://huggingface.co/docs/accelerate
    * Keras/PyTorch Lightning Callback Architecture Specification.

Author:
    Goudjou Borel

Version:
    1.0.0
"""

from typing import Any, Dict, List, Optional
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from src.entity.train_entity import EarlyStoppingConfig


class TrainState:
    """Tracks and holds the live runtime state of a model training loop.

    Acts as a centralized data class that is continuously updated throughout 
    the training lifespan and shared across all active callbacks to sync 
    distributed metrics, telemetry parameters, and optimization checkpoints.

    Attributes:
        accelerator (Accelerator): Hugging Face Accelerator instance managing 
            distributed training execution graphs (DDP, DeepSpeed, etc.).
        model (nn.Module): The target neural network architecture undergoing optimization.
        stop_training (bool): Interruption flag. When flipped to ``True`` by an internal 
            monitoring callback, it signals the active training execution loop to immediately halt.
        epoch (int): Index of the current active training epoch.
        step (int): Cumulative global step counter tracking total processed training batches.
        loss_step (Optional[float]): The raw scalar loss value calculated from the most 
            recent forward/backward optimization path step.
        metrics (Dict[str, Dict[str, Any]]): Categorized structured dictionary isolating evaluation 
            records. Accessible keys are ``"train"`` and ``"val"`` containing metric name maps.
        valid_loader (Optional[DataLoader]): Reference to the validation dataset iterable structure.
        lr (Optional[float]): The current operational learning rate set inside the core optimizer.
    """

    def __init__(self, accelerator: Accelerator, model: nn.Module) -> None:
        """Initializes the training loop runtime state tracker.

        Args:
            accelerator (Accelerator): Hugging Face orchestration engine instance.
            model (nn.Module): The target network module undergoing parameter adjustment.
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
    """Abstract base class designed to construct hooks into the model execution pipeline.

    Derive from this class to inject telemetry trackers, automated loggers, 
    dynamic learning rate adjustment logic, or custom checkpointing mechanics 
    at distinct steps of the iteration sequence.
    """

    def on_train_start(self, state: TrainState) -> None:
        """Triggered at the absolute beginning of the training loop initialization sequence.

        Args:
            state (TrainState): The active shared state instance context.
        """
        pass

    def on_epoch_start(self, state: TrainState) -> None:
        """Triggered at the beginning of each sequential training epoch.

        Args:
            state (TrainState): The active shared state instance context.
        """
        pass

    def on_step_end(self, state: TrainState) -> None:
        """Triggered immediately at the end of each optimization iteration.

        Executes right after the backward gradient update path and prior to 
        the subsequent data loader collection cycle.

        Args:
            state (TrainState): The active shared state instance context.
        """
        pass

    def on_epoch_end(self, state: TrainState) -> None:
        """Triggered at the conclusion of each epoch immediately after validation processes terminate.

        Args:
            state (TrainState): The active shared state instance context.
        """
        pass

    def on_train_end(self, state: TrainState) -> None:
        """Triggered when the total execution loop processes conclude entirely.

        Args:
            state (TrainState): The active shared state instance context.
        """
        pass


class CallbackHandler:
    """Manages collection groups of callback hooks and dispatches lifecycle event signals sequentially."""

    def __init__(self, callbacks: List[Callback]) -> None:
        """Initializes the callback management loop ecosystem.

        Args:
            callbacks (List[Callback]): Collection containing operational pipeline hooks 
                to evaluate across training lifecycle steps.
        """
        self.callbacks = callbacks

    def __call__(self, event: str, state: TrainState) -> None:
        """Invokes a targeted lifecycle string event broadcast across all registered callback modules.

        Iterates sequentially over the internal list array and routes context information using 
        reflection hooks (`getattr`).

        Args:
            event (str): The method string key name identifying the destination target 
                (e.g., ``"on_step_end"``, ``"on_epoch_end"``).
            state (TrainState): The live shared state instance context tracking current model steps.
        """
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn is not None:
                fn(state)


class StepLogCbk(Callback):
    """Callback wrapper focusing on recording and shipping iteration tracking values to Accelerator trackers."""

    def on_step_end(self, state: TrainState) -> None:
        """Dispatches real-time batch analytics to backend data log structures.

        Args:
            state (TrainState): The live shared state instance context tracking current model steps.
        """
        state.accelerator.log({
            "train/loss_step": state.loss_step, 
            "batch_step": state.step, 
            "epoch": state.epoch
        })


class EarlyStoppingCbk(Callback):
    r"""Monitors configured execution metrics to stop training runs early when progress stalls.

    Checks evaluation metrics at the close of every evaluation loop phase, tracking performance 
    ceilings to interrupt training if the metric stalls without a meaningful improvement over a 
    predefined window of consecutive epochs.

    Mathematical Improvement Tracking Logics:
        *   **Minimization Objective (e.g., loss)**:
            An improvement is declared if and only if the current validation score satisfies:
            
            .. math::

                S_{\text{current}} < S_{\text{best}} - \Delta_{\text{min}}
            
        *   **Maximization Objective (e.g., accuracy, AUC)**:
            An improvement is declared if and only if the current validation score satisfies:
            
            .. math::

                S_{\text{current}} > S_{\text{best}} + \Delta_{\text{min}}

    Attributes:
        patience (int): Number of consecutive epochs to wait without reaching a new performance 
            ceiling before triggering an early termination sequence.
        min_delta (float): Minimum absolute threshold shift required to qualify as an actual score improvement.
        mode (str): Evaluation optimization trajectory mode direction; either ``"min"`` or ``"max"``.
        counter (int): Number of consecutive epochs that failed to yield a structural improvement.
        monitor_metric (str): Key matching the exact validation dictionary path identifier to track.
        best_score (float): Internal historical top performance score benchmark. Initialized to positive 
            or negative infinity depending on the operational optimization mode.
        monitor_op (callable): Evaluator lambda function mapping operational bounds checking.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        """Initializes structural constraints for the early termination mechanism.

        Args:
            config (EarlyStoppingConfig): Hyperparameter object defining evaluation windows, 
                tolerance configurations, and monitoring directions.

        Raises:
            ValueError: If the configuration parameter ``mode`` is not strictly declared 
                as ``"min"`` or ``"max"``.
        """
        self.patience: int = config.patience
        self.min_delta: float = config.min_delta
        self.mode: str = config.mode
        self.counter: int = 0
        
        # Determine the metric to follow, defaulting safely to "loss" if missing from config
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
        """Evaluates target validation metrics against historical benchmarks to detect plateau states.

        Increments the tracking stall counter if progress criteria are unfulfilled. When 
        the counter threshold matches or exceeds ``patience``, the shared state interception flag 
        (``stop_training``) is flipped to true.

        Args:
            state (TrainState): The live shared state instance context tracking current model steps.
        """
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