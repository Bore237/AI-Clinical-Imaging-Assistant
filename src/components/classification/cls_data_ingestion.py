"""
Description:
    Ce module fournit le sous-système d'ingestion de données et de stratification des étiquettes 
    pour les tâches de classification d'images médicales. Il automatise la découverte des ressources 
    sur le disque, applique un sous-échantillonnage fractionnaire stochastique et calcule les coefficients 
    de pondération pour compenser le déséquilibre des classes (imbalance) dans les fonctions de perte.

Main components:

    * IngestionResult: Dataclass conteneur encapsulant les listes de chemins d'images, les dictionnaires 
      d'étiquettes formatés et les tenseurs/tableaux de pondération.
    * ClsDataIngestion: Classe pilote responsable du parsing des fichiers CSV, de la validation des assets 
      et de l'isolation des calculs statistiques sur le split d'entraînement.

Main features:

    * Stratégie Hybride d'Encodage: Support natif de la classification multi-classe (index numériques) 
      et multi-label (vecteurs multi-hot prenant en charge les chaînes de caractères séparées par des pipes `|`).
    * Isolation Anti-Fuite (Data Leakage Prevention): Calcul des poids de Cross Entropy et des `pos_weight` 
      de la BCE uniquement à partir du premier split (Train), garantissant l'intégrité de la validation.
    * Robustesse aux Divisions par Zéro: Intégration de fonctions planchers (`np.maximum` et `replace(0, 1)`) 
      pour sécuriser les calculs en présence de classes rares ou absentes.

General architecture:
    Le sous-système lie dynamiquement les chemins d'images découverts dans les répertoires physiques avec 
    les métadonnées extraites des fichiers d'annotations CSV. Il applique les échantillonnages configurés 
    puis aiguille le traitement selon le mode de classification choisi.

General data flow:

    .. code-block:: text

        ┌──────────────────────────────┐
        │ Configuration & Chemins CSV  │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Filtrage Extensions & Disque│ ➔ [Shuffle & Down-sampling]
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Est-ce du Multi-classe ?   │
        └──────┬────────────────┬──────┘
               │                │
               │ (Oui)          │ (Non / Multi-label)
               ▼                ▼
        [Map Class to Idx]   [Split string '|' & Multi-hot]
               │                │
               ▼                ▼
        [Calcul W_c (CE)]    [Calcul P_c (BCE pos_weight)]
               │                │
               └────────┬───────┘
                        │
                        ▼
         ┌────────────────────────────┐
         │ IngestionResult Container  │
         └────────────────────────────┘

Optimisations:

    * Nettoyage en amont: Le pipeline exécute un `drop_duplicates` sur les paires d'identifiants/étiquettes 
      pour éliminer les redondances dans les fichiers d'annotations volumineux avant toute indexation.
    * Vectorisation NumPy/Pandas: Les opérations de regroupement (`groupby`) et de calculs de ratios 
      positifs/négatifs sont entièrement vectorisées pour garantir des performances optimales sur de grands datasets.

Example:

    .. code-block:: python

        from src.entity.cls_entity import ClsDataIngestionConfig
        from src.data.ingestion import ClsDataIngestion

        config = ClsDataIngestionConfig(
            path_root="/data/spine_xr_project",
            paths_img=["train_images", "val_images"],
            paths_csv=["train.csv", "val.csv"],
            samples_rate=[1.0, 1.0],
            ext=".png",
            csv_col={"img": "image_id", "label": "finding"},
            multiclass=False
        )
        
        ingestor = ClsDataIngestion(config=config)
        result = ingestor.get_files()
        print(result.pos_weight)

References:

    * PyTorch Loss Functions (BCEWithLogitsLoss): https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    * Google Python Style Guide: https://google.github.io/styleguide/pyguide.html

Author:
    Goudjou Borel

Version:
    1.0.0
"""

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd

from src.entity.cls_entity import ClsDataIngestionConfig
from src.utils.logger import get_logger


@dataclass
class IngestionResult:
    r"""Container holding the outputs of the classification data ingestion stage.

    Attributes:
        images (List[List[str]]): Nested list of discovered absolute or relative 
            image paths for each dataset split. The first dimension corresponds 
            to dataset splits (e.g., index 0 for train, index 1 for validation).
        labels (List[Any]): Formatted target labels corresponding to each dataset split.

            * Multiclass: Mapped as a dictionary of ``{image_id: class_index_integer}``.
            * Multilabel: Mapped as a dictionary of ``{image_id: one_hot_encoded_numpy_array}``.

        weights (Dict[Any, float]): Class balancing weights computed exclusively from 
            the training split. Designed for standard Weighted Cross Entropy loss functions.
            Calculated using the following mathematical balanced ratio:

            .. math::

                W_c = \frac{N_{\text{total}}}{C \times N_c}
            
        pos_weight (Union[None, np.ndarray]): Positive class weighting factor array 
            computed exclusively from the training split. Designed to counteract severe 
            imbalances in multi-label BCE pipelines (``BCEWithLogitsLoss``).
            Calculated per class :math:`c` using the ratio:

            .. math::

                P_c = \frac{N_{\text{neg}, c}}{\max(N_{\text{pos}, c}, 1)}
    """
    images: List[List[str]]
    labels: List[Any]
    weights: Dict[Any, float]
    pos_weight: Union[None, np.ndarray]


class ClsDataIngestion:
    r"""Handles data ingestion, file discovery, sampling, and label balancing.

    This pipeline stage reads configuration paths, loads ground-truth CSV metadata records,
    validates asset existence against specified file extensions, performs dynamic fractional 
    sampling, and automatically computes loss-balancing tensors.

    Attributes:
        config (ClsDataIngestionConfig): Strongly-typed configuration container storing 
            system file paths, target column keys, extensions, and split sampling rates.
        pos_weight (Union[np.ndarray, None]): Calculated positive class weight ratios 
            isolated strictly from the primary training split to mitigate data leakage.
        weights (Dict[Any, float]): A dictionary mapping category keys to calculated 
            balancing coefficients for multiclass cross-entropy loss.
        logger (logging.Logger): Telemetry logging instance tracking ingestion progress 
            and structural parsing exceptions.
    """

    def __init__(self, config: ClsDataIngestionConfig) -> None:
        """Initializes the data ingestion stage with operational configurations.

        Args:
            config (ClsDataIngestionConfig): Target parameters containing file locations 
                and balancing settings.
        """
        self.config = config
        self.pos_weight: Union[np.ndarray, None] = None
        self.weights: Dict[Any, float] = {}
        self.logger = get_logger(__name__)

    def get_files(self) -> IngestionResult:
        r"""Discovers files, samples the datasets, and formats labels and balancing weights.

        Iterates over configured splits (e.g., train/validation tracks) to parse CSV annotations,
        match filesystem images using specified extension criteria, apply fractional down-sampling,
        and construct categorical data containers.

        Note:
            **Data Leakage Prevention**: Class weights (``self.weights``) and multi-label positive 
            balances (``self.pos_weight``) are calculated exclusively using the first split 
            (assumed to be Training). Validation and testing frequencies do not influence these metrics.

            *   **Multiclass Balancing Formula**:

                .. math::

                    W_c = \frac{N_{\text{total}}}{C \times N_c}

                Where :math:`N_{\text{total}}` is the total number of samples, :math:`C` is the number of unique 
                classes, and :math:`N_c` is the number of samples belonging to class :math:`c`.

            *   **Multilabel Positive Inbalance Scaling Factor**:

                .. math::

                    P_c = \frac{N_{\text{neg}, c}}{\max(N_{\text{pos}, c}, 1)}

                Where :math:`N_{\text{neg}, c}` is the number of negative samples for class :math:`c`, and :math:`N_{\text{pos}, c}`
                is the number of positive samples for class :math:`c`. A floor function of 1 prevents division by zero.

        Returns:
            IngestionResult: Encapsulated dataset split components ready for conversion into 
                downstream MONAI or PyTorch Dataset containers.

        Raises:
            RuntimeError: If an unexpected I/O failure occurs while reading a target annotation CSV.
            ValueError: If the asset discovery loop yields zero valid files matching the target extension.
        """
        img_files: List[List[str]] = []
        labels: List[Any] = []
        root = self.config.path_root

        # Destructure parallel configurations across available train/val datasets
        zipped_configs = zip(self.config.paths_img, self.config.paths_csv, self.config.samples_rate)
        for i, (path_img, path_csv, sample_rate) in enumerate(zipped_configs):
            csv_file_path = os.path.join(root, path_csv)
            
            try:
                df = pd.read_csv(csv_file_path)
            except Exception as e:
                self.logger.error(f"Failed to load CSV file: {csv_file_path}. Error: {str(e)}")
                raise RuntimeError(f"Impossible to load CSV file: {csv_file_path}") from e

            # Discover local files adhering strictly to configured extensions
            img_dir_path = os.path.join(root, path_img)
            raw_img_files = [
                os.path.join(img_dir_path, file)
                for file in os.listdir(img_dir_path)
                if file.endswith(self.config.ext.lower())
            ]

            processed_count = len(raw_img_files)
            if processed_count == 0:
                error_msg = f"No files discovered with extension '{self.config.ext}' in: {img_dir_path}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Execute localized random sampling slice based on config coefficients
            num_elements_to_keep = max(1, int(len(raw_img_files) * sample_rate))
            random.shuffle(raw_img_files)
            img_files.append(raw_img_files[:num_elements_to_keep])

            # Deduplicate targets on base column pairs to isolate clean classification mapping pairs
            img_col = self.config.csv_col["img"]
            label_col = self.config.csv_col["label"]
            df_unique = df.drop_duplicates([img_col, label_col]).reset_index(drop=True)

            disk_basenames = [os.path.splitext(os.path.basename(img_file))[0] for img_file in img_files[i]]
            df_unique = df_unique[df_unique[img_col].astype(str).isin(disk_basenames)]

            if df_unique.empty:
                error_msg = (
                    f"Data mapping resulted in 0 rows for split index {i}.\n"
                    f"Check if the image IDs in your CSV column '{img_col}' match your disk filenames (minus extension).\n"
                    f"Sample values from CSV: {list(df[img_col].head(3))}\n"
                    f"Sample identifiers from Disk: {disk_basenames[:3]}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Isolate distinct classes matching the current structural metadata
            if i == 0:
                unique_classes = sorted(df_unique[label_col].unique())
                class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

                # Compute balancing factors exclusively on the training split to avoid validation leakage
                class_counts = df_unique[label_col].value_counts().reindex(unique_classes).fillna(0)
                class_counts = class_counts.replace(0, 1)

                # Balance equation: total_samples / (num_classes * class_samples)
                calculated_weights = class_counts.sum() / (len(unique_classes) * class_counts)
                self.weights = calculated_weights.to_dict()

            # --- Label Strategy Formatting Selection ---
            if self.config.multiclass:
                # Multiclass Setup: Clean key-value lookup dictionaries (image_id -> label_index)
                df_unique["_label__digit_"] = df_unique[label_col].map(class_to_idx)
                labels.append(dict(zip(df_unique[img_col], df_unique["_label__digit_"])))
                self.logger.info("Configured pipeline for standard Multi-class classification.")
            
            else:
                # Multilabel Setup: One-hot array maps, supporting pipe-separated strings
                def encode_labels(value):
                    if isinstance(value, str):
                        current_labels = [x.strip() for x in value.split("|")]
                    else:
                        current_labels = value
                    return [1 if cls in current_labels else 0 for cls in unique_classes]
                
                df_unique["_label__one__hot_"] = df_unique[label_col].apply(encode_labels)
                image_labels = df_unique.groupby(img_col)["_label__one__hot_"].apply(
                    lambda x: np.array(x.tolist()).max(axis=0)
                )
                labels.append(image_labels.to_dict())

                # Compute positive/negative ratio balances exclusively on the training split
                if i == 0:
                    stacked_labels = np.stack(image_labels.values)  # type: ignore
                    positive_counts = stacked_labels.sum(axis=0)
                    negative_counts = stacked_labels.shape[0] - positive_counts
                    
                    # Prevent division by zero scenarios if a class contains zero occurrences
                    self.pos_weight = negative_counts / np.maximum(positive_counts, 1)

                self.logger.info("Configured pipeline for Multi-label / Binary classification.")

            # Validate extraction yield to safeguard training loaders
            processed_count = len(img_files[i])
            if processed_count == 0:
                error_msg = f"No files discovered with extension '{self.config.ext}' in: {img_dir_path}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            split_identifier = "train" if i == 0 else "val"
            self.logger.info(
                f"Successfully ingested {processed_count} files into the {split_identifier} pipeline."
            )

        return IngestionResult(img_files, labels, self.weights, self.pos_weight)