"""
Description:
    This module centralizes configuration data structures for a PyTorch, MONAI, and timm-based 
    image classification pipeline (e.g., medical or computer vision applications). It provides 
    strongly-typed objects (via `TypedDict` and `dataclass`) that cleanly isolate data ingestion, 
    loading, transformations, and model hyperparameters.

Main components :

    * FlipConfig, ContrastConfig, GaussianConfig, AffineConfig: Structural dictionaries (TypedDict) 
      precisely defining data augmentation parameters.
    * ClsDataIngestionConfig: Immutable dataclass managing source file access, filtering, and 
      sub-sampling ratios across data splits.
    * ClsDataLoaderConfig: Immutable dataclass encapsulating hardware execution constraints for the 
      PyTorch DataLoader runtime engine.
    * ClsTransformationConfig: Immutable dataclass driving target spatial dimensions, asynchronous 
      memory caching strategies, and augmentation pipelines.
    * ClsModelConfig: Mutable dataclass orchestrating neural network model architecture lifecycles.

Main features :

    * Strict runtime type and constraint validation powered by the `@beartype` decorator.
    * Immutability (`frozen=True`) across ingestion, loader, and transformation configs, ensuring 
      pipeline robustness and eliminating unintended side effects.
    * Controlled flexibility via mutability on the model configuration, allowing dynamic updates 
      to the target class distribution after dataset parsing.
    * Native compatibility with MONAI dictionary-based containers and `timm` backbone registries.

General architecture :
    The architecture is organized into highly decoupled configuration layers, strictly separating 
    pipeline responsibilities:
    [Ingestion Layer] ➔ [Transformation / Cache Layer] ➔ [Loading / Stream Layer] ➔ [Model Layer]

General data flow :

    .. code-block:: text 

        [Raw Data Files (Images & CSV)] ➔ ClsDataIngestionConfig (Selection & Filtering)
                                                       │
                                                       ▼
        [Asynchronous Cache Optimization] ➔ ClsTransformationConfig (Normalization & Augmentation)
                                                       │
                                                       ▼
        [PyTorch DataLoader Stream]       ➔ ClsDataLoaderConfig (Batching & Prefetching)
                                                       │
                                                       ▼
        [Neural Network (timm)]            ➔ ClsModelConfig (Inference / Training)

Optimisations :

    * IO bandwidth optimization via asynchronous and parallelized RAM caching (`cache_rate`, 
      `cache_num_workers`), optimized for high-resolution volumetric imaging.
    * Strict inter-experiment reproducibility ensured by injecting a global random anchor (`seed`).
    * Lightweight native data structures (`TypedDict`, `dataclasses`) minimizing memory overhead 
      when handling configuration metadata across distributed workers.

Example:

    .. code-block:: python

        from pathlib import Path
        from module_config import ClsDataIngestionConfig, ClsModelConfig

        # 1. Configure raw data ingestion pipelines
        ingestion_cfg = ClsDataIngestionConfig(
            ext=".nii.gz",
            csv_col={"img": "image_path", "label": "label_id"},
            path_root=Path("/data/workspace"),
            multiclass=True,
            paths_img=("train/images", "val/images"),
            paths_csv=("train_meta.csv", "val_meta.csv"),
            samples_rate=(1.0, 1.0),
            seed=42
        )

        # 2. Configure model properties (with a temporary placeholder for num_classes)
        model_cfg = ClsModelConfig(
            dropout_rate=(0.1, 0.2),
            in_chans=1,
            feature_head=None,
            model_name="vit_base_patch16_224",
            num_classes=0,  # Will be updated dynamically after parsing the CSV
            pretrained=True,
            uuid_tag="exp-cls-v1"
        )

Note:
    The `ClsModelConfig` class intentionally uses the property `frozen=False`. This design pattern 
    allows the execution engine (ingester or trainer) to dynamically parse dataset metadata (CSV), 
    infer the unique number of target classes actually present, and overwrite the `num_classes` 
    attribute seamlessly on the fly prior to neural network compilation.

References:

    * Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
    * Project MONAI Documentation: https://docs.monai.io/
    * Timm (Torch Image Models) Repository: https://github.com/huggingface/pytorch-image-models

Author:
    Goudjou Borel

Version:
    1.0.0
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple, TypedDict, Union, List, Optional
from beartype import beartype


# --- Structural Type Definitions (TypedDict) ---

class LoadImage(TypedDict):
    """Configuration for loading and preprocessing a medical image.

    Attributes:
        image_size: Target image size (H, W).
        z_score: Whether to apply z-score normalization image per image.
        mean: Mean value used for normalization. Ignored if ``z_score`` is False.
        std: Standard deviation used for normalization. Ignored if ``z_score`` is False.
        quantile: Lower and upper quantiles used for intensity clipping, e.g. [0.01, 0.99].
    """
    image_size: Tuple[int, int]
    z_score: bool
    mean:  Optional[float]
    std: Optional[float]
    quantile: List[float] | None

class FlipConfig(TypedDict):
    """Configuration mapping for random spatial flipping transforms.

    Attributes:
        prob (float): Probability threshold anchoring the structural transform trigger.
        axis (int): Spatial dimension axis index targeted for spatial inversion.
    """
    prob: float
    axis: int


class ContrastConfig(TypedDict):
    """Configuration mapping for dynamic image contrast adjustments.

    Attributes:
        prob (float): Probability of executing the contrast modification on a given sample.
        gamma (Tuple[float, float]): Bounded range scaling the nonlinear intensity adjustments.
    """
    prob: float
    gamma: Tuple[float, float]


class GaussianConfig(TypedDict):
    """Configuration mapping for additive white Gaussian noise generation.

    Attributes:
        prob (float): Execution probability threshold across the dataset stream.
        std (float): Standard deviation multiplier scaling the stochastic noise magnitude.
    """
    prob: float
    std: float


class AffineConfig(TypedDict):
    """Configuration mapping for geometric affine transformations.

    Attributes:
        prob (float): Probability of applying structural spatial deformations.
        rotate (float): Maximum absolute rotation angle variation boundary evaluated in radians.
        scale (Tuple[float, float]): Bounded min/max range constraints applied to spatial scaling.
        translate (Tuple[int, int]): Maximum spatial pixel coordinate shift offsets across axes.
        pad_mode (Literal["zeros", "border", "reflection"]): Edge padding strategy adopted to fill newly formed out-of-boundary voids.
    """
    prob: float
    rotate: float
    scale: Tuple[float, float]
    translate: Tuple[int, int]
    pad_mode: Literal["zeros", "border", "reflection"]


# --- Dataclass Entities ---

@beartype
@dataclass(frozen=True)
class ClsDataIngestionConfig:
    """Configuration entity handling raw data ingestion pipelines and source mappings.

    Manages structural data asset parsing, verification file filters, and sub-sampling ratios 
    across cross-validation splits.

    Attributes:
        ext (str): Target image format extension marker (e.g., ``'.png'``, ``'.nii.gz'``).
        csv_col (Dict[Literal["img", "label"], str]): Safe metadata schema mapping keys to explicit tracking spreadsheet column identities.
        path_root (Path): Top-level root workspace directory containing the source files.
        multiclass (bool): Toggles optimization formats between Multiclass or Multilabel settings.
        paths_img (Tuple[str, ...]): Path pointers locating data split directories relative to root.
        paths_csv (Tuple[str, ...]): File locations mapping text label frames across data splits.
        samples_rate (Tuple[float, ...]): Sub-sampling fraction rates applied across data pipelines.
        seed (int): Global random index anchor ensuring structural replication across experiments.
    """
    ext: str
    csv_col: Dict[Literal["img", "label"], str]
    path_root: Path
    multiclass: bool
    paths_img: Tuple[str, ...]
    paths_csv: Tuple[str, ...]
    samples_rate: Tuple[float, ...]
    seed: int


@beartype
@dataclass(frozen=True)
class ClsDataLoaderConfig:
    """Configuration parameters specialized for PyTorch DataLoaders.

    Manages execution constraints related to hardware data streaming performance.

    Attributes:
        batch_size (int): Number of processing samples bundled per gradient update step.
        kwarg_loader (dict): Keyword configurations transferred directly to the DataLoader runtime engine (e.g., ``num_workers``, ``pin_memory``, ``persistent_workers``).
    """
    batch_size: int
    kwarg_loader: dict


@beartype
@dataclass(frozen=True)
class ClsTransformationConfig:
    """Pipeline properties governing data normalization, scaling, and augmentations.

    Houses specialized attributes targeting spatial dimension layouts and high-speed memory caches, 
    optimized for deep learning frameworks like MONAI.

    Attributes:
        load_image (LoadImage): all information to load medical images.
        cache_rate (Tuple[float, float]): Caching capacity factor parameters matching [train, val] splits.
        cache_num_workers (int): Processing thread limits allotted to async memory caches.
        flip (FlipConfig): Structural configuration mapping for random spatial array flips.
        contrast (ContrastConfig): Structural configuration mapping for dynamic contrast alterations.
        gaussian_noise (GaussianConfig): Stochastic tracking configurations for noise addition.
        affine (AffineConfig): Geometric mapping details for spatial translations and rotations.
    """
    load_image: LoadImage
    cache_rate: Tuple[float, float]
    cache_num_workers: int
    flip: FlipConfig
    contrast: ContrastConfig
    gaussian_noise: GaussianConfig
    affine: AffineConfig


@beartype
@dataclass()
class ClsModelConfig:
    """Architectural attributes steering the neural network lifecycle and configuration.

    Note:
        Unlike other pipeline tracking states, this dataclass remains mutable (``frozen=False``). 
        This structural design allows the ingestion engine to dynamically parse the target training 
        dataset, infer the unique class output distribution, and update ``num_classes`` seamlessly 
        prior to network construction.

    Attributes:
        dropout_rate (Tuple[float, ...]): Dropout rate configuration tuple, mapped 
            as ``[drop_path_rate, standard_dropout_rate]``.
        in_chans (int): Source image input channel size (e.g., 1 for Grayscale/CT, 3 for RGB).
        feature_head (Union[int, None]): Custom hidden projection dimension size, or ``None`` 
            if routing directly through native classification heads.
        model_name (str): Backbone signature targeting specific ``timm`` library instances.
        num_classes (int): Classification output target feature map count. Planners can provide 
            an initial placeholder value as it is automatically overwritten during initialization.
        pretrained (bool): Flag toggling whether to initialize weights from fine-tuned registries.
        uuid_tag (str): Cryptographic unique string tracking experimental models across registries.
    """
    dropout_rate: Tuple[float, ...]
    in_chans: int
    feature_head: Union[int, None]
    model_name: str
    num_classes: int
    pretrained: bool
    uuid_tag: str

@beartype
@dataclass(frozen=True)
class LossConfig:
    """Configuration of the loss function.

    The loss implementation is selected automatically according to the
    classification task (multiclass or multilabel).

    Attributes:
        gamma: Focusing parameter used by focal loss.
        class_weight: Whether to apply class weights for multiclass training.
        pos_weight: Whether to apply positive class weights for multilabel
            training.
        label_smoothing: Label smoothing factor. If ``None``, label smoothing
            is disabled.
        reduction: Reduction applied to the loss. One of ``"mean"``,
            ``"sum"``, or ``"none"``.
    """

    gamma: float
    class_weight: bool
    pos_weight: bool
    label_smoothing: float = 0.0
    reduction: Literal["mean", "sum", "none"] = "mean"