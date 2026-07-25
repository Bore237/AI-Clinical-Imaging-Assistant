"""
Description:
    Ce module orchestre le pipeline d'ingestion et de transformation de données pour la classification 
    d'images médicales (ex: radiographies de la colonne vertébrale). Il encapsule le chargement 
    haute performance de MONAI en séparant les opérations déterministes lourdes (mises en cache AOT) 
    des augmentations stochastiques appliquées à la volée.

Main components:

    * XDataset: Sous-classe PyTorch (`Dataset`) exploitant le système `CacheDataset` de MONAI 
      avec un mécanisme de repli récursif tolérant aux pannes.
    * ClsDataTransformation: Gestionnaire et assembleur des chaînes de transformation (Compose) 
      générant des instances prêtes pour les boucles d'entraînement et de validation.

Main features:

    * Accélération Ahead-Of-Time (AOT): Décodage, réalignement des canaux et redimensionnement 
      exécutés une seule fois au démarrage, éliminant les goulots d'étranglement E/S du disque.
    * Gestion Hybride des Cibles: Routage automatique des étiquettes selon le paradigme choisi 
      (LongScalar pour le multi-classe, FloatVector pour le multi-label).
    * Tolérance aux Pannes Robuste: En cas de fichier corrompu ou d'erreur de parsing au runtime, 
      le chargeur capture l'exception et sélectionne aléatoirement un autre échantillon pour éviter 
      l'interruption brutale du pipeline d'entraînement.

General architecture:
    La séparation stricte entre les transformations de chargement (`load_t`) et de runtime (`runtime_t`) 
    permet d'optimiser l'utilisation de la mémoire RAM/NVMe :
    [Chemins Disque] ➔ Ingestion MONAI ➔ Cache RAM (`CacheDataset`) ➔ Augmentations Stochastiques ➔ Batch Tensors

General data flow:

    .. code-block:: text

        [Boot Time]
        Fichiers sur Disque ➔ LoadImage ➔ EnsureChannelFirst ➔ Ingestion Cache RAM (AOT)
        
        [Training Epochs / Runtime Loop]
        Cache RAM ➔ [__getitem__(index)] ➔ RandFlip/RandAffine ➔ Normalisation Type ➔ Output Batch

Optimisations:

    * Multiprocessing AOT: La phase de mise en cache initiale exploite l'intégralité des cœurs 
      processeurs disponibles via `os.cpu_count()`.
    * Normalisation Ciblée: Utilisation des constantes globales `MEAN` et `STD` adaptées au domaine 
      médical via `NormalizeIntensity` avec filtrage des intensités non nulles (`nonzero=True`).

Example:

    .. code-block:: python

        from src.entity.cls_entity import ClsTransformationConfig
        from src.data.dataset import ClsDataTransformation

        config = ClsTransformationConfig(
            image_size=(512, 512),
            cache_rate=(1.0, 0.0),
            flip={"prob": 0.5, "axis": 0},
            affine={"prob": 0.7, "rotate": 4, "scale": (0.8, 1.2), "translate": (0.1, 0.1), "pad_mode": "reflection"},
            contrast={"prob": 0.3, "gamma": (0.5, 4.5)},
            gaussian_noise={"prob": 0.2, "std": 0.1}
        )
        
        orchestrator = ClsDataTransformation(config=config, multiclass=True)
        train_ds, val_ds = orchestrator.transforms(img_files=([path1, path2], [path3]), labels=(labels_train, labels_val))

References:
    * MONAI CacheDataset Architecture: https://docs.monai.io/
    * PyTorch Data Loading Utility: https://pytorch.org/docs/stable/data.html

Author:
    Goudjou Borel

Version:
    1.0.0
"""

import os
from typing import List, Tuple, Union, Any, Literal, Dict
import numpy as np
import torch
import monai.transforms as T
from monai.data import CacheDataset  # pyright: ignore[reportPrivateImportUsage]
from torch.utils.data import Dataset

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