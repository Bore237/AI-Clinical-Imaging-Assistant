import os
import random
from typing import Any, Dict, List, Tuple, Union, Dict
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.entity.cls_entity import ClsDataIngestionConfig
from src.utils.logger import get_logger

@dataclass
class IngestionResult:
    """Container holding the outputs of the classification data ingestion stage.

    Attributes:
        images (List[List[str]]):
            Nested list of discovered image paths for each dataset split.
            The first dimension corresponds to dataset splits (e.g. train, validation),
            and the second dimension contains sampled image file paths.

        labels (List[Any]):
            Formatted labels for each dataset split.
            For multiclass classification, each element is typically a dictionary
            mapping image identifiers to integer class indices.
            For multilabel classification, each element is typically a dictionary
            mapping image identifiers to binary one-hot encoded vectors.

        weights (Dict[Any, float]):
            Class balancing weights computed from the training split only.
            Used for standard cross-entropy loss weighting in imbalanced datasets.

        pos_weight (Union[None, np.ndarray]):
            Positive class weighting factors computed from the training split only.
            Used for binary cross-entropy losses in multilabel classification.
            None when positive class weighting is not applicable.
    """
    images: List[List[str]]
    labels: List[Any]
    weights: Dict[Any, float]
    pos_weight: Union[None, np.ndarray]

class ClsDataIngestion:
    """Handles data ingestion, sampling, and label formatting for classification.

    This pipeline stage reads configuration paths, loads and processes ground truth CSV files,
    discovers matching local image files, draws a configured random sample subset, and 
    automatically computes class weights or positive sample balances to combat label imbalance.

    Attributes:
        config (ClsDataIngestionConfig): Configuration object containing folder paths,
            sampling rates, extensions, and target columns.
        pos_weight (Union[np.ndarray, None]): Calculated positive class weight tensor ratios 
            for BCE losses, isolated exclusively from the training data split.
        weights (Dict[Any, float]): A dictionary map of calculated dynamic class balances 
            for standard cross-entropy losses, isolated exclusively from the training split.
        logger (logging.Logger): Logger instance for status reporting and error tracking.
    """

    def __init__(self, config: ClsDataIngestionConfig) -> None:
        """Initializes the data ingestion component.

        Args:
            config (ClsDataIngestionConfig): Dataset ingestion and preprocessing configurations.
        """
        self.config = config
        self.pos_weight: Union[np.ndarray, None] = None
        self.weights: Dict[Any, float] = {}
        self.logger = get_logger(__name__)

    def get_files(self) -> IngestionResult:
        """Discovers files, samples the datasets, and formats labels and balancing weights.

        Iterates over configured splits (e.g., train/validation pairs) to read CSV annotations,
        match local image listings by extension, slice data dynamically based on requested 
        sampling rates, and format output structures depending on the classification setup.

        Note:
            Class weights (`self.weights`) and positive balances (`self.pos_weight`) are 
            calculated exclusively using the first split (assumed to be Training) to prevent 
            downstream validation data leakage.

        Returns:
            Tuple[List[List[str]], List[Any], Dict[Any, float], Union[None, np.ndarray]]:
                - Nested lists containing paths to discovered image files for each dataset split.
                - Structured labels matching classification layouts (Dict for multiclass, DataFrame for multilabel).
                - A dictionary map of training class balance configurations.
                - An array of training positive-to-negative imbalance factor coefficients (or None).

        Raises:
            RuntimeError: If a configured ground-truth annotation CSV cannot be loaded.
            ValueError: If zero valid image assets are found inside the target image folders.
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

            # Execute localized random sampling slice based on config coefficients
            num_elements_to_keep = max(1, int(len(raw_img_files) * sample_rate))
            random.shuffle(raw_img_files)
            img_files.append(raw_img_files[:num_elements_to_keep])

            # Deduplicate targets on base column pairs to isolate clean classification mapping pairs
            img_col = self.config.csv_col["img"]
            label_col = self.config.csv_col["label"]
            df_unique = df.drop_duplicates([img_col, label_col]).reset_index(drop=True)

            # Isolate distinct classes matching the current structural metadata
            unique_classes = sorted(df_unique[label_col].unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

            # Compute balancing factors exclusively on the training split to avoid validation leakage
            if i == 0:
                class_counts = df_unique[label_col].value_counts().reindex(unique_classes).fillna(0)
                class_counts = class_counts.replace(0,1)
                # Balance equation: total_samples / (num_classes * class_samples)
                calculated_weights = class_counts.sum() / (len(unique_classes) * class_counts)
                self.weights = calculated_weights.to_dict()

            # --- Label Strategy Formatting Selection ---
            # Multiclass Setup: Clean key-value lookup dictionaries (image_id -> label)
            if self.config.multiclass:
                df_unique["_label__digit_"] = df_unique[label_col].map(class_to_idx)
                
                labels.append(dict(zip(df_unique[img_col], df_unique["_label__digit_"])))
                self.logger.info("Configured pipeline for standard Multi-class classification.")
            
            else:
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