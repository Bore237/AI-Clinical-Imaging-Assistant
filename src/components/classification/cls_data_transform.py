"""
Description:
    This module defines the data ingestion and preprocessing pipeline for medical
    image classification (e.g., spine radiographs). It separates deterministic
    preprocessing operations from stochastic data augmentations to leverage
    MONAI's caching mechanism and improve training performance.

Main components:

    * XDataset: PyTorch ``Dataset`` wrapper around MONAI's ``CacheDataset`` including a fault-tolerant fallback mechanism for corrupted samples.
    * LoadResizeMedImg: Custom transform for loading, preprocessing, resizing and padding DICOM or NumPy medical images.
    * ClsDataTransformation: Factory class assembling deterministic loading transforms and runtime augmentation pipelines.

Main features:

    * Cached preprocessing: Image loading, intensity normalization, resizing, and padding are executed once during cache creation.
    * Runtime augmentations: Random spatial and intensity transformations are applied only during training.
    * Flexible normalization: Supports dataset-level normalization using fixed statistics or per-image z-score normalization.
    * Robust loading: Automatically skips unreadable or corrupted samples by selecting another random sample instead of interrupting training.
    * Multi-class and multi-label support: Automatically formats target tensors according to the selected classification task.

General architecture:
    The pipeline separates deterministic preprocessing from stochastic
    augmentations:

        Disk files
            │
            ▼
        LoadResizeMedImg
            │
            ▼
        CacheDataset (cached)
            │
            ▼
        Runtime augmentations
            │
            ▼
        Model input

General data flow:

    .. code-block:: text

        [Initialization]
        Disk Files
            └── LoadResizeMedImg
                    └── CacheDataset

        [Training]
        CacheDataset
            └── RandFlip
            └── RandAffine
            └── RandAdjustContrast
            └── RandGaussianNoise
                    └── Batch tensors

Optimizations:

    * Multi-process caching: Cache initialization uses all available CPU cores.
    * Cached deterministic transforms: Expensive preprocessing is executed only
      once, reducing disk I/O during training.
    * Efficient runtime pipeline: Only lightweight stochastic augmentations are
      applied after samples have been cached.

Example:

    .. code-block:: python

        from src.entity.cls_entity import ClsTransformationConfig
        from src.data.dataset import ClsDataTransformation

        config = ClsTransformationConfig(
            load_image={
                "image_size": [128, 128]
                "z_score": true
                "mean": None
                "std": None 
                "quantile": [0.01, 0.99]
            },
            cache_rate=(1.0, 0.0),
            flip={"prob": 0.5, "axis": 0},
            affine={
                "prob": 0.7,
                "rotate": 36,
                "scale": (0.8, 1.2),
                "translate": (0.1, 0.1),
                "pad_mode": "reflection",
            },
            contrast={"prob": 0.3, "gamma": (0.5, 4.5)},
            gaussian_noise={"prob": 0.2, "std": 0.1},
        )

        transforms = ClsDataTransformation(config=config, multiclass=True)
        train_ds, val_ds = transforms.transforms(img_files=([path1, path2], [path3]),  labels=(train_labels, val_labels))

References:
    * MONAI documentation: https://docs.monai.io/
    * PyTorch Dataset documentation: https://pytorch.org/docs/stable/data.html

Author:
    Goudjou Borel

Version:
    1.0.0
"""

import os
from typing import List, Tuple, Union, Any, Literal, Dict
import numpy as np
import torch
import torch.nn.functional as F
import monai.transforms as T
from monai.data import CacheDataset 
from torch.utils.data import Dataset
import pydicom

# NOTE: MEAN and STD are imported here as fixed constants.
from src.configs import MEAN, STD
from src.entity.cls_entity import ClsTransformationConfig
from src.utils.logger import get_logger


class XDataset(Dataset):
    """Custom Dataset for medical image classification using MONAI.

    Supports both multiclass and multilabel classification paradigms, and encapsulates 
    MONAI's caching pipeline (`CacheDataset`) to accelerate multi-epoch disk-to-tensor data 
    streaming.

    Attributes:
        img_files (Tuple[str, ...]): Array collection containing absolute or relative filesystem paths 
            to targeted volumetric/2D imagery.
        labels (Union[Dict[str, Any], Any]): Indexable metadata structure mapping localized structural image IDs 
            to dense target categorical arrays or continuous multi-hot lists.
        kwargs (Dict[str, Any]): Dynamic key-value mappings storing configurations such as cache rate metrics 
            or execution mode flags.
        logger (logging.Logger): Telemetry logger tracking pipeline execution status and data ingest anomalies.
        cached (CacheDataset): High-performance persistence layer executing ahead-of-time (AOT) image decodings.
        runtime (T.Compose): Executable transform sequence applied on-the-fly to cached samples during iteration.
    """

    def __init__(
        self, 
        img_files: Tuple[str, ...], 
        labels: Union[Dict[str, Any], Any], 
        load_t: T.Compose, 
        runtime_t: T.Compose,
        **kwargs: Any
    ) -> None:
        """Initializes the dataset and configures the MONAI data cache.

        Args:
            img_files (Tuple[str, ...]): File paths to the target medical images.
            labels (Union[Dict[str, Any], Any]): Structural data dictionary or frame matching image file base IDs to target targets.
            load_t (T.Compose): Deterministic transform operations processed exactly once during boot caching.
            runtime_t (T.Compose): Stochastic or volatile operations processed dynamically per batch request.
            **kwargs: Extra settings including ``cache_rate`` (float) and ``multiclass`` (bool).
        """
        self.img_files = img_files
        self.labels = labels
        self.kwargs = kwargs
        self.logger = get_logger(__name__)

        # Set up CacheDataset to accelerate future epoch training runs
        cache_rate = kwargs.get("cache_rate", 0.0)
        self.cached = CacheDataset(
            data=self.img_files, 
            transform=load_t, 
            cache_rate=cache_rate, 
            num_workers=os.cpu_count() or 1
        )
        self.runtime = runtime_t
        self.logger.info(f"Dataset successfully initialized. Cache rate: {cache_rate * 100}%")

    def __len__(self) -> int:
        """Returns the total number of samples contained within this dataset repository split.

        Returns:
            int: The total count of verified structural image file locations.
        """
        return len(self.img_files)

    def __getitem__(self, index: int) -> Dict[Literal["image", "label"], torch.Tensor]:
        """Retrieves the transformed image and its corresponding label tensor at a given index.

        Extracts samples from the memory/disk cache, applies stochastic augmentations, 
        and structures target values based on the classification mode.

        Note:
            Fault-Tolerant Mitigation: If data degradation, file format corruption, or unexpected array scaling 
            triggers a parsing exception, this loader executes an auto-fallback routine that recursively selects 
            a random sample from the index map to prevent pipeline termination.

        Args:
            index (int): Sequence lookup target position tracking the wanted image.

        Returns:
            Dict[Literal["image", "label"], torch.Tensor]: A structured dictionary containing:
            
                * ``"image"``: Prepared Pytorch floating point tensor optimized for model graph ingest.
                * ``"label"``: Multi-class indices (long integer scalar) or multi-label targets (float vector).
        """
        img_id = os.path.splitext(os.path.basename(self.img_files[index]))[0]
        
        try:
            # --- Label Extraction & Parsing ---
            if self.kwargs.get("multiclass", False):
                # Multiclass Mode: Fetch the index of the target class
                label_idx = torch.tensor(self.labels[img_id], dtype=torch.long)
            else:
                # Multilabel Mode: Extract One-Hot vector from DataFrame / dictionary
                label_idx = torch.tensor(self.labels[img_id], dtype=torch.float32)

            # --- Runtime Transformation Application ---
            img_data = self.runtime(self.cached[index]) 
            if not isinstance(img_data, torch.Tensor):
                img_data = torch.tensor(img_data)

            return {"image": img_data, "label": label_idx}
            
        except Exception as e:
            # Error Handling: Log the failure and fall back to a random fallback index
            self.logger.error(f"Failed to process image ID: {img_id}. Error details: {str(e)}")
            alt_index = int(np.random.randint(0, len(self.img_files)))
            return self.__getitem__(alt_index)

class LoadResizeMedImg:
    """Load, preprocess, resize, and pad a medical image.

    This transform supports DICOM (`.dcm`) and NumPy (`.npz`) image files.
    Images are normalized to the range [0, 1] using their bit depth,
    optionally intensity-clipped, normalized, resized while preserving the
    aspect ratio, and zero- or minimum-padded to the target spatial size.

    Args:
        spatial_size: Target output size as ``(height, width)``.
        z_score: If ``True``, apply z-score normalization using foreground
            pixels only.
        mean: Mean used for normalization when ``z_score`` is ``False``.
        std: Standard deviation used for normalization when ``z_score`` is
            ``False``.
        quantile: Optional lower and upper quantiles (e.g., ``(0.01, 0.99)``)
            used for intensity clipping on foreground pixels.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape ``(C, H, W)``.
    """
    def __init__(self, spatial_size: tuple, z_score: bool = False, mean: float | None = None, std: float | None = None,  quantile=None):
        self.mean = None if mean is None else torch.tensor(mean, dtype=torch.float32)
        self.std = None if std is None else torch.tensor(std, dtype=torch.float32)
        self.z_score = z_score
        self.spatial_size = spatial_size
        self.quantile = quantile

    def __call__(self, data):
        data = str(data)

        if data.endswith(".npz"): 
            npz = np.load(data)
            img = torch.from_numpy(npz["img"]).float()
            img.div_(2**int(npz["bitsStored"]) - 1)
        elif data.endswith(".dicom") or data.endswith(".dcm"):
            dcm = pydicom.dcmread(data)
            arr = dcm.pixel_array
            if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
                arr = np.amax(arr) - arr 

            img = torch.from_numpy(arr).float()
            bits = getattr(dcm, "BitsStored", arr.dtype.itemsize * 8)
            img.div_(2 ** bits - 1)
        else:
            raise ValueError("This library only support dicom and npz extension")

        # Foreground mask (ignore background pixels)
        fg_mask = img > img.min() + 1e-3  

        # Optional intensity clipping
        if self.quantile is not None:
            p1, p99 = torch.quantile(img[fg_mask], torch.tensor(self.quantile))
            img.clamp_(p1, p99)

        # Intensity normalization
        if self.z_score:
            mean = img[fg_mask].mean()
            std = img[fg_mask].std()
            img.sub_(mean).div_(std.clamp_min(1e-6))
        else:
            if self.mean is not None and self.std is not None:
                img.sub_(self.mean).div_(self.std)

        # Ensure channel-first format (C, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # Compute isotropic resize factor
        _, h, w = img.shape
        target_h, target_w = self.spatial_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize while preserving aspect ratio
        img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True).squeeze(0)

        # Compute symmetric padding
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if self.z_score or (self.mean is not None and self.std is not None):
            pad_value = float(img_resized.amin())
            img_padded = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom),mode="constant", value=pad_value)
        else:
            img_padded = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
            
        return img_padded

class ClsDataTransformation:
    """Data transformation pipeline manager for medical image classification tasks.

    Responsible for assembling isolated ahead-of-time (AOT) ingestion pipelines and 
    stochastic runtime augmentation configurations, then returning fully tracking-ready 
    training and validation `XDataset` object structures.

    Attributes:
        logger (logging.Logger): Dedicated logging instance for tracking configuration parsing states.
        config (ClsTransformationConfig): Strong-typed container detailing augmentation ranges.
        multiclass (bool): Execution flag defining whether targets are mutually exclusive.
        load_transform (monai.transforms.Compose): Ingest pipeline dedicated to the persistent cache.
        train_transform (monai.transforms.Compose): Augmentation transforms bound to the active training loop.
        val_transforms (monai.transforms.Compose): Standard evaluation tensor casting pipeline.
    """

    def __init__(self, config: ClsTransformationConfig, multiclass: bool) -> None:
        """Initializes the pipeline manager with operational transformation criteria.

        Args:
            config (ClsTransformationConfig): System configuration container storing user hyperparameters.
            multiclass (bool): Operational flag declaring target evaluation space shapes.
        """
        self.logger = get_logger(__name__)
        self.config = config
        self.multiclass = multiclass
        
        # Placeholders for MONAI Compose objects
        self.load_transform: T.Compose
        self.train_transform: T.Compose
        self.val_transforms: T.Compose

    def _get_data_transformation(self) -> None:
        """Builds static (cached) and dynamic (runtime) MONAI transformation chains.
        
        Isolates deterministic operations like multi-threaded image decoding, channel layout formatting, 
        and resolution scaling onto the system persistence cache layer, leaving light stochastic affine 
        and intensity alterations to process at execution runtimes.
        """
        # Initial loading pipeline
        self.load_transform = T.Compose([  
            LoadResizeMedImg(spatial_size=self.config.load_image["image_size"],
                            z_score=self.config.load_image["z_score"], 
                            mean=self.config.load_image["mean"],
                            std=self.config.load_image["std"],  
                            quantile=self.config.load_image["quantile"])
        ])

        # Data Augmentation pipeline (Executed on-the-fly during training steps)
        self.train_transform = T.Compose([  
            T.RandFlip(
                prob=self.config.flip["prob"], 
                spatial_axis=self.config.flip["axis"]
            ),  
            T.RandAffine(
                prob=self.config.affine["prob"], 
                rotate_range=(-np.pi / self.config.affine["rotate"], np.pi / self.config.affine["rotate"]), 
                scale_range=self.config.affine["scale"], 
                translate_range=self.config.affine["translate"],     
                padding_mode=self.config.affine["pad_mode"]
            ),
            T.RandAdjustContrast(
                prob=self.config.contrast["prob"], 
                gamma=self.config.contrast["gamma"]
            ),  
            T.RandGaussianNoise(
                prob=self.config.gaussian_noise["prob"], 
                mean=0.0, 
                std=self.config.gaussian_noise["std"]
            ),  
        ])

        # Validation / Test pipeline (No augmentations, standard type casting)
        self.val_transforms = T.Compose([
            T.Identity()
        ])  

    def transforms(
        self, 
        img_files: List[List[str]], 
        labels: List[Any]
    ) -> Tuple[XDataset, XDataset]:
        """Generates operational training and validation XDataset instances.

        Compiles MONAI transform states, separates file array inputs into relative paths execution paths, 
        and returns separate train/val dataset loaders.

        Args:
            img_files (Tuple[List[str], List[str]]): Nested collection holding paths segmented exactly into `(train_file_list, val_file_list)`.
            labels (Tuple[Any, Any]): Structured tuple holding target label representations split into `(train_labels, val_labels)`.

        Returns:
            Tuple[XDataset, XDataset]: Fully configured datasets prepared for DataLoader integration.

                * Index 0: Training Dataset coupled with dynamic data augmentations.
                * Index 1: Validation Dataset utilizing static, deterministic inference transforms.
        """
        self._get_data_transformation()
        
        train_img_files, val_img_files = img_files
        train_labels, val_labels = labels

        # Instantiate training dataset setup
        train_ds = XDataset(
            img_files=tuple(train_img_files),
            labels=train_labels,
            load_t=self.load_transform,  
            runtime_t=self.train_transform,  
            cache_rate=self.config.cache_rate[0],
            multiclass=self.multiclass,
        )
        self.logger.info("Training Dataset pipeline built successfully.")

        # Instantiate validation dataset setup
        valid_ds = XDataset(
            img_files=tuple(val_img_files),
            labels=val_labels,
            load_t=self.load_transform,  
            runtime_t=self.val_transforms,  
            cache_rate=self.config.cache_rate[1],
            multiclass=self.multiclass,
        )
        self.logger.info("Validation Dataset pipeline built successfully.")

        return train_ds, valid_ds