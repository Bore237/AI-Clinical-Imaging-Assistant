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