import os
from typing import List, Tuple, Union, Any, Literal
import numpy as np
import torch
import monai.transforms as T
from monai.data import CacheDataset  # pyright: ignore[reportPrivateImportUsage]
from torch.utils.data import Dataset

# NOTE: MEAN and STD are imported here as fixed constants. 
# To modify their values globally, update them directly in the project's constants file.
from src.constants import MEAN, STD
from src.entity.cls_entity import ClsTransformationConfig
from src.utils.logger import get_logger

class XDataset(Dataset):
    """Custom Dataset for medical image classification using MONAI.

    Supports both multiclass and multilabel classification, and leverages 
    MONAI's caching mechanism to optimize I/O performance.

    Attributes:
        img_files (Tuple[str, ...]): List of file paths to the images.
        labels_list (List[Any]): Ordered list of unique class keys.
        labels (Union[Dict, Any]): Data structure holding labels (Dict or Pandas DataFrame).
        kwargs (Dict[str, Any]): Additional configuration parameters (e.g., multiclass, cache_rate).
        logger (logging.Logger): Logger instance for execution tracking.
        cached (CacheDataset): MONAI dataset with pre-cached load-time transformations.
        runtime (T.Compose): Transformation pipeline applied on-the-fly during iteration.
    """

    def __init__(
        self, 
        img_files: tuple, 
        labels: Union[dict, Any], 
        load_t: T.Compose, 
        runtime_t: T.Compose,
        **kwargs: Any
    ) -> None:
        """Initializes the dataset and configures the MONAI data cache.

        Args:
            img_files (tuple): File paths to the target images.
            labels (Union[dict, Any]): Image labels (Dict or Pandas DataFrame).
            load_t (T.Compose): MONAI transforms executed once during caching.
            runtime_t (T.Compose): Runtime transformations (e.g., data augmentations).
            **kwargs: Extra settings (`multiclass`, `cache_rate`, `col_image`, etc.).
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
        """Returns the total number of samples in the dataset."""
        return len(self.img_files)

    def __getitem__(self, index: int) -> dict[Literal["image", "label"], torch.Tensor]:
        """Retrieves the transformed image and its corresponding label tensor at a given index.

        Falls back to a random index if a file reading or processing error occurs 
        to prevent halting the training loop.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            dict[Literal["image", "label"], torch.Tensor]: Formatted image tensor and label tensor.
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
            alt_index = np.random.randint(0, len(self.img_files))
            return self.__getitem__(alt_index)


class ClsDataTransformation:
    """Data transformation pipeline manager for image classification tasks.

    Handles building separate load-time (cached) and runtime (augmented) pipelines,
    and returns fully configured training and validation `XDataset` objects.

    Note:
        The `MEAN` and `STD` parameters used during intensity normalization are currently 
        fixed imports. To alter these values, edit them directly within your project's 
        `constants` configuration module.
    """

    def __init__(self, config: ClsTransformationConfig, multiclass: bool) -> None:
        """
        Args:
            config (ClsTransformationConfig): Configuration object storing user hyperparameters.
            multiclass (bool): Flag defining whether the problem is multiclass or multilabel.
        """
        self.logger = get_logger(__name__)
        self.config = config
        self.multiclass = multiclass
        
        # Placeholders for MONAI Compose objects
        self.load_transform: T.Compose
        self.train_transform:T.Compose
        self.val_transforms: T.Compose

    def _get_data_transformation(self) -> None:
        """Builds static (cached) and dynamic (runtime) MONAI pipelines.
        
        Isolates heavy operations (loading, resizing) meant for the persistent cache,
        from stochastic training augmentations (flips, affine transformations).
        """
        # Initial loading pipeline (Perfect candidate for caching)
        self.load_transform = T.Compose([  
            T.LoadImage(),  
            T.EnsureChannelFirst(),  
            T.NormalizeIntensity(subtrahend=MEAN, divisor=STD, nonzero=True),  # type: ignore
            T.Resize(spatial_size=self.config.image_size),  
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
                spatial_size=self.config.image_size, 
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
            T.EnsureType(data_type="tensor", dtype=torch.float32)
        ])  

    def transforms(self, img_files: List[list], labels: List[Any]) -> Tuple[XDataset, XDataset]:
        """Generates final operational XDataset instances for training and validation runs.

        Args:
            img_files (List[list]): Image paths split into format: [train_files, val_files].
            labels (List[Any]): Split labels matching format: [train_labels, val_labels].

        Returns:
            Tuple[XDataset, XDataset]: Configured Train and Validation datasets.
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