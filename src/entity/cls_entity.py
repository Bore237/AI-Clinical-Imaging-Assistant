from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple, TypedDict, Union
from beartype import beartype

# --- Structural Type Definitions ---
class FlipConfig(TypedDict):
    """Configuration mapping for random spatial flipping transforms."""
    prob: float
    axis: int

class ContrastConfig(TypedDict):
    """Configuration mapping for dynamic image contrast adjustments."""
    prob: float
    gamma: Tuple[float, float]

class GaussianConfig(TypedDict):
    """Configuration mapping for additive white Gaussian noise generation."""
    prob: float
    std: float

class AffineConfig(TypedDict):
    """Configuration mapping for geometric affine transformations."""
    prob: float
    rotate: float
    scale: Tuple[float, float]
    translate: Tuple[int, int]
    pad_mode: Literal["zeros", "border", "reflection"]


# --- Dataclass Entities ---
@beartype
@dataclass(frozen=True)
class ClsDataIngestionConfig:
    """Configuration entity handling raw data ingestion pipelines.

    Attributes:
        ext (str): Targeted image file format extension (e.g., ".png", ".nii.gz").
        csv_col (Dict[Literal["img", "label"], str]): Safe metadata schema dictionary 
            mapping abstract internal data targets ("img", "label") to explicit tabular column names.
        path_root (Path): Top-level root directory containing files and subfolders.
        multiclass (bool): Toggles evaluation formats between Multiclass or Multilabel settings.
        path_img (Tuple[str, ...]): Path pointers locating file split source folders.
        path_csv (Tuple[str, ...]): Path pointers locating split target annotations files.
        sample (Tuple[float, ...]): Sub-sampling fraction rates applied across data pipelines.
        seed (int): Global random index anchor ensuring reproducibility across experiments.
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

    Attributes:
        batch_size (int): Number of processing samples bundled per gradient update step.
        kwarg_loader (dict): Keyword configurations transferred directly to the DataLoader runtime engine.
    """
    batch_size: int
    kwarg_loader: dict

@beartype
@dataclass(frozen=True)
class ClsTransformationConfig:
    """Pipeline properties governing data normalization and augmentations.

    Attributes:
        image_size (Tuple[int, ...]): Final dimension matrix target layout used during spatial scaling.
        cache_rate (Tuple[float, float]): Caching capacity factor parameters matching [train, val] splits.
        cache_num_workers (int): Processing thread limits allotted to async MONAI memory caches.
        flip (FlipConfig): Parameters matching structural random spatial array flips.
        contrast (ContrastConfig): Parameters matching structural on-the-fly contrast alterations.
        gaussian_noise (GaussianConfig): Parameters matching stochastic additive noise additions.
        affine (AffineConfig): Parameters matching custom image rotations, translations, and scaling.
    """
    image_size: Tuple[int, ...]
    cache_rate: Tuple[float, float]
    cache_num_workers: int
    flip: FlipConfig
    contrast: ContrastConfig
    gaussian_noise: GaussianConfig
    affine: AffineConfig

@beartype
@dataclass()
class ClsModelConfig:
    """Architectural attributes steering the neural network lifecycle.

    Attributes:
        dropout_rate (Tuple[float, ...]): Dropout rate configuration tuple, mapped 
            as `[drop_path_rate, standard_dropout_rate]`.
        in_chans (int): Source image input channel channel size (e.g., 1 for Grayscale/CT, 3 for RGB).
        feature_head (int | None): Dimension sizing for custom hidden layers, or None 
            if selecting native classification heads.
        model_name (str): Backbone signature targeting specific timm library instances.
        num_classes (int): Classification output target feature map count.
            **Note:** This value is automatically inferred from the training dataset
                and overwritten internally during initialization. Any placeholder value
                may be provided in the configuration.
        pretrained (bool): Flag toggling whether to initialize weights from fine-tuned registries.
        uuid_tag (str): Cryptographic unique string tracking experimental models.
    """
    dropout_rate: Tuple[float, ...]
    in_chans: int
    feature_head: Union[int, None]
    model_name: str
    num_classes: int
    pretrained: bool
    uuid_tag: str
