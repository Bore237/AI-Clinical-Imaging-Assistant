"""
Description:
    Ce module implémente la classe centrale de modélisation (`ClassifierModel`) pour les 
    architectures de classification deep learning. Il encapsule l'instanciation dynamique 
    de backbones de pointe (ex: EfficientNet, RegNet, DenseNet) via la bibliothèque `timm` 
    et propose un basculement transparent entre la tête linéaire native du modèle et une 
    tête de classification perceptron multicouche (MLP) personnalisée.

Main components:

    * ClassifierModel: Wrapper PyTorch (`nn.Module`) unifiant l'extraction de caractéristiques 
      et les projections linéaires adaptées aux configurations d'imagerie.

Main features:

    * Intégration Native timm: Chargement dynamique de modèles pré-entraînés ou non, avec configuration 
      séparée du Stochastic Depth (drop path) et du Dropout classique.
    * Tête MLP Optionnelle: Génération automatique d'un bloc de projection robuste composé de couches 
      `Linear`, `LayerNorm`, `ReLU` et `Dropout` pour affiner les représentations complexes.
    * Gestion Flexible des Canaux: Configuration explicite des canaux d'entrée (`in_chans`), idéale 
      pour traiter des radiographies ou volumes spécifiques.

General architecture:
    Le modèle agit comme un conteneur modulaire. Si une dimension de tête (`feature_head`) est spécifiée, 
    la tête native de `timm` est désactivée via `num_classes=0` pour ne renvoyer que le vecteur de caractéristiques 
    global, qui est ensuite injecté dans la séquence personnalisée :
    [Tenseur d'Image Input] ➔ timm Encoder ➔ [Tête Custom MLP / Tête Native] ➔ [Logits de Classification]

General data flow:

    .. code-block:: text

        ┌─────────────────────────┐
        │   Input (B, C, H, W)    │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │      timm Encoder       │
        └──────┬───────────┬──────┘
               │           │
               │ (feature_head is None)
               │           │ ───➔ [Native Linear Head] ───➔ Logits (B, Classes)
               ▼
         Features (B, Hidden)
               │
               ▼
         [Dropout Layer]
               │
               ▼
         [Custom MLP Head] ➔ Linear ➔ LayerNorm ➔ ReLU ➔ Dropout ➔ Linear ➔ Logits (B, Classes)

Optimisations:

    * Extraction Efficace: L'utilisation de `num_classes=0` dans `timm` court-circuite la création de la couche 
      linéaire par défaut, économisant des ressources mémoire et du calcul inutile.
    * Normalisation des Caractéristiques: L'inclusion d'une couche `LayerNorm` au sein de la tête personnalisée 
      stabilise la distribution des activations et accélère la convergence des gradients sur les cibles 
      médicales hautement résolues.

Example:

    .. code-block:: python

        from src.entity.cls_entity import ClsModelConfig
        from src.models.classification import ClassifierModel

        config = ClsModelConfig(
            model_name="efficientnet_b0",
            pretrained=True,
            in_chans=1,
            num_classes=3,
            dropout_rate=(0.2, 0.3),
            feature_head=512,
            uuid_tag="a1b2c3"
        )
        model = ClassifierModel(config_model=config)
        outputs = model(dummy_tensor)

Note:
    Lors du passage à `num_classes=0`, la plupart des architectures de `timm` appliquent automatiquement 
    leur pooling global par défaut (ex: Global Average Pooling) pour renvoyer un tenseur 2D de forme `[B, Features]`. 
    Assure-toi que le backbone choisi respecte cette propriété sous peine de devoir aplatir manuellement le tenseur.

References:

    * Ross Wightman PyTorch Image Models (timm): https://github.com/huggingface/pytorch-image-models
    * Google Python Style Guide: https://google.github.io/styleguide/pyguide.html

Author:
    Goudjou Borel

Version:
    1.0.0
"""

from typing import cast
import torch
import torch.nn as nn
import timm

from src.entity.cls_entity import ClsModelConfig


class ClassifierModel(nn.Module):
    """Deep learning backbone classifier built on top of the `timm` library.

    This architecture encapsulates a convolutional or transformer encoder backbone. 
    It dynamically toggles between using the native timm classification head or constructing 
    a custom multi-layer classification head based on configuration properties.

    Attributes:
        feature_head (Union[int, None]): The hidden layer dimension size of the custom head, 
            or None if utilizing the model's native linear head.
        encoder (nn.Module): The core feature extraction backbone instantiated via `timm`.
        drop (nn.Dropout): Dropout regularization layer applied right before the linear projections.
        head (nn.Sequential): Optional multi-layer classification sequence consisting of 
            Linear projections, LayerNorm, ReLU, and Dropout layers.
    """

    def __init__(self, config_model: ClsModelConfig) -> None:
        """Initializes the classification model topology.

        Args:
            config_model (ClsModelConfig): Configuration entity containing architectural attributes 
                such as backbone name, input channels, target class count, and dropout rates.
        """
        super().__init__()
        self.feature_head = config_model.feature_head
        
        # If a custom structural feature head is specified, the native timm head is disabled (num_classes=0).
        self.encoder = timm.create_model(
            config_model.model_name, 
            pretrained=config_model.pretrained, 
            in_chans=config_model.in_chans,
            drop_path_rate=config_model.dropout_rate[0],  # Stochastic depth / drop path rate
            drop_rate=config_model.dropout_rate[1],       # Backbone classification dropout rate
            num_classes=config_model.num_classes if config_model.feature_head is None else 0
        )

        # Retrieve the encoder output feature map dimension for proper layer pairing
        num_features = cast(int, self.encoder.num_features)
        self.drop = nn.Dropout(p=config_model.dropout_rate[1])
        
        # Build custom multilayer perceptron (MLP) classification head if requested
        if config_model.feature_head is not None:
            self.head = nn.Sequential(
                nn.Linear(num_features, config_model.feature_head), 
                nn.LayerNorm(config_model.feature_head),
                nn.ReLU(),          
                nn.Dropout(config_model.dropout_rate[1]),    
                nn.Linear(config_model.feature_head, config_model.num_classes)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes the forward propagation path across the model network.

        Args:
            x (torch.Tensor): Input batch image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Raw logits tensor of shape (batch_size, num_classes).
        """
        if self.feature_head is None:
            # Route processing directly through the native timm architecture
            return self.encoder(x) 
        
        # Route processing through the feature extractor backbone followed by the custom head layers
        features = self.encoder(x) 
        features = self.drop(features)
        logits = self.head(features)
        return logits