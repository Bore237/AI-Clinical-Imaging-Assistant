"""
Description:
    Advanced custom loss modules for PyTorch, optimizing classification performance
    under heavy class imbalance in both multi-label and multi-class target configurations.
    These modules are designed for safety, precision, and numerical stability.

Main components :

    * MultiLabelFocalLoss: Focal Loss for multi-label classification tasks.
    * MultiClassFocalLoss: Focal Loss for multi-class classification tasks.

Main features :

    * Dynamic sample-level loss modulation based on prediction difficulty.
    * Uncoupled label smoothing execution preventing target masking during focal factor extraction.
    * Fully vectorized parallel matrix operations for minimal execution overhead.

General architecture :
    Standard PyTorch `nn.Module` subclasses implementing optimized `forward` methods 
    leveraging high-performance underlying ATen/CUDA tensor operations via PyTorch functional API.

General data flow :

    .. code-block:: text

        [MultiLabel Data Flow]
        Logits ----> Sigmoid --------------------------> Probabilities ----> pt Extraction
                                                                               |
        Logits + Smoothed Targets ----> F.binary_cross_entropy_with_logits ---> Multiplied by (1-pt)^gamma

        [MultiClass Data Flow]
        Logits ----> Softmax --------------------------> Probabilities ----> Tensor Gather (targets) -> pt
                                                                               |
        Logits + Int Targets ---------> F.cross_entropy (none) --------------> Multiplied by (1-pt)^gamma

Optimisations :
    * Vectorized spatial masking using `torch.where` eliminates multi-label loops.
    * Dimension mining via `torch.gather` removes iterative multi-class index matching.
    * Delayed micro-reductions to maximize CUDA kernel execution throughput.

Example:
    .. code-block:: python

        import torch
        from losses import MultiLabelFocalLoss, MultiClassFocalLoss

        # Multi-Label Verification
        ml_logits = torch.randn(4, 5, requires_grad=True)
        ml_targets = torch.randint(0, 2, (4, 5)).float()
        ml_criterion = MultiLabelFocalLoss(gamma=2.0, label_smoothing=0.05)
        ml_loss = ml_criterion(ml_logits, ml_targets)
        ml_loss.backward()

        # Multi-Class Verification
        mc_logits = torch.randn(4, 3, requires_grad=True)
        mc_targets = torch.tensor([0, 2, 1, 2], dtype=torch.long)
        mc_criterion = MultiClassFocalLoss(gamma=2.0, reduction='mean')
        mc_loss = mc_criterion(mc_logits, mc_targets)
        mc_loss.backward()

Note:
    Combining Label Smoothing and Focal Loss creates a theoretical conflict: Label Smoothing
    penalizes absolute confidence ($p \to 1$), while Focal Loss forces the model to become highly
    confident on easy examples to down-weight their loss. Use them together only if substantial
    label noise (e.g., inter-observer variability in clinical annotations) is present.

References:

    * Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense 
      object detection. In Proceedings of the IEEE international conference on computer vision.
    * Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P., & Dokania, P. (2020). 
      Calibrating deep neural networks using focal loss. NeurIPS.

Author:
    Goudjou Borel

Version:
    1.0.0
"""

import torch
import torch.nn as nn
from typing import Sequence
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    r"""Focal Loss implementation for Multi-Label classification configurations.

    This loss extends the standard Binary Cross-Entropy by adding a modulating factor
    :math:`(1 - p_t)^{\gamma}` to dynamically scale the loss based on prediction confidence.
    It isolates the True Class Probability :math:`(p_t)` calculation from the smoothed targets 
    tensor to prevent target masking bugs during loss backpropagation.

    Mathematical Definition:

        .. math::

            FL(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)

        Where :math:`p_t` is computed as:

        .. math::
            
            p_t = y \cdot \sigma(x) + (1 - y) \cdot (1 - \sigma(x))

        And :math:`\sigma(x)` represents the sigmoid activation function applied to raw logits.

    Args:
        gamma (float): Focusing parameter adjusting the down-weighting rate of easy examples.
            Higher values scale down easy examples faster. Defaults to 2.0.
        pos_weight (torch.Tensor, Sequence, optional): A weight tensor of shape `(num_classes,)` 
            to rescale the positive class loss. Useful for handling severe static internal 
            imbalance within individual classes. Defaults to None.
        class_weight (torch.Tensor, Sequence, optional): A manual scaling weight tensor of shape `(num_classes,)` 
                    assigning relative importance values across classes. Defaults to None.
        label_smoothing (float): Uniform smoothing factor applied to the binary targets.
            Transforms targets to :math:`\epsilon / 2` and :math:`1 - \\epsilon / 2`. Defaults to 0.0.
        reduction (str): Post-processing reduction strategy applied to the calculated tensor.
            Supported configurations are: 'none', 'mean', or 'sum'. Defaults to 'mean'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | Sequence[float] | float |None = None,
        class_weight: torch.Tensor | Sequence[float] | float | None = None,
        label_smoothing: float = 0.0,
        transform_weight: str = "log",
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if pos_weight is not None:
            pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32)

            if transform_weight == "log":
                pos_weight_tensor = torch.log1p(pos_weight_tensor)
            elif transform_weight == "sqrt":
                pos_weight_tensor = torch.sqrt(pos_weight_tensor)
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.register_buffer("pos_weight", None)

        if class_weight is not None:
            self.register_buffer("class_weight", torch.as_tensor(class_weight, dtype=torch.float32))
        else:
            self.register_buffer("class_weight", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Executes the mathematical forward pass for Multi-Label Focal Loss.

        Args:
            logits (torch.Tensor): Unnormalized network outputs of shape `(Batch Size, Num Classes)`.
            targets (torch.Tensor): Binary ground-truth matrix of shape `(Batch Size, Num Classes)`
                containing strictly `0.0` or `1.0` float entries.

        Returns:
            torch.Tensor: Computed scalar loss value or raw matrix depending on the chosen reduction method.
        """
        # Compute exact pt probabilities using uncorrupted binary ground-truth targets
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)

        # Inject label smoothing explicitly into a decoupled target representation
        if self.label_smoothing > 0.0:
            smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed_targets = targets

        # Compute element-wise Binary Cross Entropy with raw logit configurations
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            smoothed_targets,
            pos_weight=self.pos_weight, # type: ignore
            weight=self.class_weight, # type: ignore
            reduction="none"
        )

        # Apply the mathematical focal down-weighting scaling factor
        focal_loss = ((1.0 - pt) ** self.gamma) * bce_loss

        # Apply execution reduction paths
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

from typing import Sequence
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Computes the Asymmetric Loss (ASL) for multi-label classification.

    Asymmetric Loss addresses the positive-negative imbalance by applying 
    asymmetric focusing and asymmetric probability clipping to decouple the 
    treatment of positive and negative samples.

    Reference:
        Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification",
        ICCV 2021. https://arxiv.org/abs/2009.14119

    Attributes:
        gamma_neg: Focusing parameter for negative samples (gamma-).
        gamma_pos: Focusing parameter for positive samples (gamma+).
        clip: Probability shift threshold for negative sample clipping.
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'.
        eps: Small constant to avoid numerical instability in log operations.
        pos_weight: Weight tensor applied to positive samples for balance.
    """

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 0.0,
        clip: float = 0.15,
        reduction: str = "mean",
        transform_weight: str = "log",
        pos_weight: torch.Tensor | Sequence[float] | float | None = None,
        eps: float = 1e-8,
    ) -> None:
        """Initializes the AsymmetricLoss module.

        Args:
            gamma_neg: Decay rate for easy negative samples. Defaults to 2.0.
            gamma_pos: Decay rate for hard positive samples. Defaults to 0.0.
            clip: Margin threshold for negative probability shifting. If > 0,
                discards easy negatives below this threshold. Defaults to 0.15.
            reduction: Specifies the reduction to apply to the output:
                'none', 'mean', or 'sum'. Defaults to "mean".
            pos_weight: A weight for positive targets. If provided, must be a 
                scalar or a sequence with shape equal to the number of classes.
                Defaults to None.
            eps: Epsilon value for log numerical stability. Defaults to 1e-8.

        Raises:
            ValueError: If an unsupported reduction method is provided.
        """
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"Invalid reduction mode: '{reduction}'. "
                "Supported modes are 'none', 'mean', or 'sum'."
            )

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.eps = eps

        if pos_weight is not None:
            pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32)
            
            if transform_weight == "log":
                pos_weight_tensor = torch.log1p(pos_weight_tensor)
            elif transform_weight == "sqrt":
                pos_weight_tensor = torch.sqrt(pos_weight_tensor)
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.register_buffer("pos_weight", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the asymmetric loss between logits and binary targets.

        Args:
            logits: Unnormalized model predictions of shape 
                `(batch_size, num_classes)`.
            targets: Binary ground-truth labels of shape 
                `(batch_size, num_classes)` containing values in {0, 1}.

        Returns:
            Calculated loss tensor. Returns a scalar if reduction is 'mean' 
            or 'sum', or a tensor of shape `(batch_size, num_classes)` if 'none'.
        """
        # Calculate probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping (margin shift for negative samples)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic Binary Cross-Entropy components
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        # Apply positive class weighting if defined
        if self.pos_weight is not None:
            los_pos = los_pos * self.pos_weight # type: ignore

        loss = los_pos + los_neg

        # Asymmetric Focusing (Focal loss factor)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)

            loss = loss * one_sided_w

        # Compute loss (negative log-likelihood)
        loss = -loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
            
class MultiClassFocalLoss(nn.Module):
    r"""Focal Loss implementation for Multi-Class categorical classification configurations.

    This loss scales categorical Cross-Entropy dynamically, focusing the gradient step
    on hard examples while dampening the penalty assigned to correctly classified majority 
    classes. True class probabilities :math:`(p_t)` are extracted via highly parallelized matrix gathering.

    Mathematical Definition:
        
        .. math::
        
            (p_t) = -(1 - p_t)^{\gamma} \log(p_t)

        Where :math:`p_t` is extracted via index gathering matching the ground-truth categorical label:
        
        .. math::
            
            p_t = \text{Softmax}(\mathbf{x})_{[\text{batch}, \text{target}]}

    Args:
        gamma (float): Focusing parameter adjusting the down-weighting rate of easy examples.
            Setting :math:`\gamma = 0` collapses this module back into standard Categorical Cross Entropy.
            Defaults to 2.0.
        class_weight (torch.Tensor, Sequence, optional): A manual scaling weight tensor of shape `(num_classes,)` 
            assigning relative importance values across classes. Defaults to None.
        label_smoothing (float): Uniform smoothing factor applied to categorical distribution.
            Spreads a tiny probability mass uniformly across wrong labels. Defaults to 0.0.
        reduction (str): Post-processing reduction strategy applied to the calculated tensor.
            Supported configurations are: 'none', 'mean', or 'sum'. Defaults to 'mean'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weight: torch.Tensor | Sequence[float] | float | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if class_weight is not None:
            self.register_buffer("class_weight", torch.as_tensor(class_weight, dtype=torch.float32))
        else:
            self.register_buffer("class_weight", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Executes the mathematical forward pass for Multi-Class Focal Loss.

        Args:
            logits (torch.Tensor): Unnormalized network outputs of shape `(Batch Size, Num Classes)`.
            targets (torch.Tensor): Categorical target indices tensor of shape `(Batch Size)` 
                containing integers in the range :math:`[0, \text{Num Classes} - 1]`.

        Returns:
            torch.Tensor: Computed scalar loss value or raw vector depending on the chosen reduction method.
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weight, # type: ignore
            label_smoothing=self.label_smoothing,
            reduction="none"
        )

        # Extract standard categorical softmax probabilities across the class dimension
        probs = torch.softmax(logits, dim=-1)

        # Vectorized extraction of pt using parallel tensor dimension mining
        pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        # Apply execution reduction paths
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss