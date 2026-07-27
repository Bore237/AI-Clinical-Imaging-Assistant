r"""
Description:
    Industrial-grade medical imaging preprocessing and analytics engine designed to handle 
    large-scale datasets (DICOM, NumPy arrays) efficiently using asynchronous worker process pools.
    It specializes in memory-efficient dataset statistical calculation and multi-format batch 
    conversion pipelines.

Main components :

    * read_yaml: Configuration parser mapping YAML streams safely into attribute accessors.
    * create_directories: Safe idempotent directory tree structural creator.
    * process_dicom: Isolated worker job parsing DICOM data into compressed NPZ disk formats.
    * process_single_image: Local mathematical reducer normalising arrays and computing aggregations.
    * mean_std_fast: Globally distributed map-reduce statistical aggregator engine.

Main features :

    * High-throughput parallelized calculation of global dataset mean and standard deviation.
    * Fast batch conversion of raw DICOM images into lightweight compressed `.npz` files.
    * Advanced medical image bit-depth intensity normalization supporting heterogeneous inputs.

General architecture :
    Exposes stateless core functional blocks driven by a two-pronged structural Command Line 
    Interface (CLI) using sub-parsers. Parallel task synchronization relies on a centralized 
    multi-process worker scheduler pool to bypass Python's Global Interpreter Lock (GIL).

General data flow :

    .. code-block:: text

        [CLI - convert command] ---> Discover DICOMs ---> Distribute Tasks (ProcessPool)
                                                                    |
                                                                    v
                                              Read Array + BitsStored -> Compress -> Save .npz

        [CLI - stat command]    ---> Discover Images ---> Map Slices (ProcessPool)
                                                                    |
                                                                    v
                                              Normalize -> Compute Sum & Sum of Squares (Local)
                                                                    |
                                                                    v
                                              Global Reduction -> Compute Mean & Std -> Write Report

Optimisations :

    * Employs chunkless streaming maps across standalone CPU processes to eliminate memory leaks.
    * Avoids full dataset loading in memory by using local parallel reductions (Map-Reduce model).
    * Uses dynamic bit-depth computation per image slice to ensure maximum intensity scale safety.

Example:

    .. code-block:: python

        # Execute statistical extraction directly via Python API
        from src.utils.dataset_tool import mean_std_fast

        mean, std, report = mean_std_fast(
            img_dir="/data/train_radiographies",
            ext=".npz",
            nonzero=True
        )
        print(f"Dataset Mean: {mean}, Std: {std}")

Note:
    When combining calculations over large radiology datasets, always discard zero-valued pixels 
    (:math:`\text{nonzero}=\text{True}`) to avoid masking the true tissue distribution with 
    collimator masks or black padding spaces.

References:

    * Digital Imaging and Communications in Medicine (DICOM) Standard: https://www.dicomstandard.org/
    * NumPy High-Performance Storage Architecture: https://numpy.org/doc/stable/reference/routines.io.html

Author:
    Goudjou Borel

Version:
    1.0.0
"""

import io
import os
import yaml
import pydicom
import argparse
import zipfile
import numpy as np
from pathlib import Path
from box import ConfigBox 
from tqdm.auto import tqdm
from box.exceptions import BoxValueError
from typing import List, Tuple, Union, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.logger import get_logger

# Initialize the module-level logger channel
logger = get_logger(__name__)


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML configuration file and parses its contents into a ConfigBox.

    This utility reads an operational configuration file from disk using a safe parsing 
    algorithm to prevent arbitrary code execution vulnerabilities. The final dictionary 
    is encapsulated inside a ``ConfigBox`` object, allowing downstream components to access 
    nested parameters via clean dot notation rather than standard string keys.

    Args:
        path_to_yaml (Path): Structural filesystem path pointing directly to the target 
            YAML configuration file.

    Returns:
        ConfigBox: An optimized data dictionary wrapper providing attribute-style key 
            access boundaries.

    Raises:
        ValueError: If the targeted YAML structural layout resolves to an empty file 
            or fails to generate key-value configurations.
        Exception: Re-raises any underlying OS or hardware-level file stream access exceptions.

    Note:
        Wrapping dictionaries inside a ``ConfigBox`` minimizes code verbosity:
        Converting ``config['hyperparameters']['learning_rate']`` into 
        ``config.hyperparameters.learning_rate``.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("The provided YAML file is empty.")
    except Exception as e:
        raise e


def create_directories(path_to_directories: List[Union[str, Path]], verbose: bool = True) -> None:
    """Creates a collection of operational filesystem directories recursively if they do not exist.

    Iterates over a collection of sequence paths, verifying or manufacturing structural 
    folders down the file tree. This method is fully idempotent; if worker processes 
    attempt to generate identical directories concurrently, the engine bypasses creation 
    safely without crashing or altering existing data parameters.

    Args:
        path_to_directories (List[Union[str, Path]]): A list sequence containing directory 
            locations declared either as strings or explicit Path instances.
        verbose (bool, optional): Controls confirmation tracking telemetry. If True, 
            broadcasts an info-level verification log message for each validated target folder. 
            Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory verified or created at destination: {path}")

def process_dicom(
    dicom_path: str, 
    root_save: str, 
    zip_path: Optional[str] = None
) -> Tuple[str, bool, Optional[str]]:
    """Reads a DICOM file from disk or a ZIP archive, processes it, and exports it as an NPZ.

    This function operates as a standalone job designed for parallel execution. It 
    can parse standard filesystem files or extract files directly into memory from 
    a ZIP archive without prior disk decompression. It extracts the pixel array, 
    normalizes contrast for MONOCHROME1 images, and saves the payload with its 
    native bit depth into a compressed `.npz` file. Internal subdirectories 
    (e.g., 'train', 'test') found inside ZIP archives are automatically preserved.

    Args:
        dicom_path (str): Local filesystem path to the DICOM file, or the internal 
            archive path if processing from a ZIP (e.g., 'train/patient1/001.dcm').
        root_save (str): Target directory where the compressed `.npz` archive 
            will be saved.
        zip_path (str, optional): Path to the source local ZIP archive if the target 
            DICOM file is contained inside a zip. Defaults to None.

    Returns:
        Tuple[str, bool, Optional[str]]: A tuple containing execution metadata:
            * dicom_path (str): The processed filepath or internal archive path.
            * success (bool): Execution status. True if successful, False otherwise.
            * error_message (str or None): Error trace if an exception occurred, 
              otherwise None.
    """
    try:
        if zip_path:
            # Read DICOM bytes directly from the ZIP archive into RAM
            with zipfile.ZipFile(zip_path, 'r') as z:
                file_bytes = z.read(dicom_path)
            ds = pydicom.dcmread(io.BytesIO(file_bytes))
            
            # Retain the internal archive structure (e.g., 'train/patient1/001.npz')
            relative_npz = os.path.splitext(dicom_path)[0] + ".npz"
            save_path = os.path.join(root_save, relative_npz)
        else:
            # Read DICOM directly from the local filesystem
            ds = pydicom.dcmread(dicom_path)
            
            # Standard behavior: flatten filename to the root save directory
            base_name = os.path.splitext(os.path.basename(dicom_path))[0] + ".npz"
            save_path = os.path.join(root_save, base_name)

        # Extract and process pixel payload
        arr = ds.pixel_array.astype(np.float32)

        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = np.amax(arr) - arr

        imgs = {
            "img": arr,
            "bitsStored": ds.BitsStored
        }

        # Dynamically create subdirectories if they don't exist yet
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to a compressed NPZ archive format
        np.savez_compressed(save_path, **imgs)
        return dicom_path, True, None

    except Exception as e:
        return dicom_path, False, str(e)


def process_single_image(img_path: str, ext: str, nonzero: bool = True) -> Tuple[float, float, int, bool]:
    r"""Processes a single medical image slice to extract running summary statistics.

    This function runs within isolated worker processes. It loads the target image, 
    normalizes its pixel values based on the metadata storage bit depth, filters out 
    background/padding pixels if requested, and computes intermediate scalar aggregations.

    Mathematical Normalization:
        Each pixel intensity value :math:`I_{\text{raw}}` is normalized to a floating point value 
        range between :math:`0.0` and :math:`1.0` using the header's technical bit depth storage 
        specification (:math:`b = \text{BitsStored}`) via the following function:

        .. math::

           I_{\text{norm}} = \frac{I_{\text{raw}}}{2^b - 1}

    Args:
        img_path (str): Absolute or relative filesystem path to the target image file.
        ext (str): File extension of the target asset (e.g., '.dcm', '.dicom', '.npy', '.npz').
            Case-insensitive.
        nonzero (bool, optional): If True, discards background zero-valued pixels 
            (such as collimator masks or padding) before computing metrics. Defaults to True.

    Returns:
        Tuple[float, float, int, bool]: A tuple containing four elements:

            * **sum_pixels** (float): Sum of all valid normalized pixel intensities.
            * **sum_squared** (float): Sum of all squared valid normalized pixel intensities.
            * **num_pixels** (int): Total count of valid pixels processed in this slice.
            * **success** (bool): True if the file was processed without errors, False otherwise.

    Raises:
        KeyError: If a NumPy file structure is missing required 'img' or 'bitsStored' keys.
        TypeError: If the loaded NumPy file payload is not a dictionary-like structure.
        ValueError: If an unsupported file extension is supplied.
    """
    try:
        ext_lower = ext.lower()
        if ext_lower in ['.dcm', '.dicom']:
            dcm = pydicom.dcmread(img_path)
            img = dcm.pixel_array.astype(np.float32) / (2**dcm.BitsStored - 1)
        elif ext_lower in ['.npz', '.npy']:
            data = np.load(img_path, allow_pickle=True)
            
            if ext_lower == '.npy' and data.ndim == 0:
                data = data.item()
            
            if isinstance(data, dict) or hasattr(data, 'keys'):
                if "img" in data and "bitsStored" in data:
                    img = data["img"].astype(np.float32) / (2**int(data["bitsStored"]) - 1)
                else:
                    raise KeyError(f"Missing required keys 'img' or 'bitsStored'. Found: {list(data.keys())}")
            else:
                raise TypeError("Loaded numpy file does not contain a valid dictionary-like structure.")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        if nonzero:
            img = img[img != 0]
        
        if img.size == 0:
            return 0.0, 0.0, 0, True
            
        return float(img.sum()), float(np.square(img).sum()), int(img.size), True

    except Exception as e:
        logger.error(f"Error reading or parsing image asset {img_path}: {e}")
        return 0.0, 0.0, 0, False


def mean_std_fast(img_dir: str, ext: str, nonzero: bool = True) -> Tuple[float, float, Dict]:
    r"""Computes global dataset mean and standard deviation using parallelized processing.

    Scans the target directory for matching assets, distributes processing jobs across all
    available CPU cores using a process pool, and accumulates local reductions into global
    metrics. This prevents high peak memory consumption by avoiding full dataset loading.

    Mathematical Reduction Architecture:
        This engine implements a parallel Map-Reduce aggregation sequence. Local worker tasks 
        extract local sum statistics (:math:`S_k`) and local sum of squares (:math:`SS_k`) 
        for each image slice :math:`k`, alongside the structural pixel count :math:`N_k`. 
        The global parameters are subsequently assembled via the following reduction formulas:

        .. math::

           \mu = \frac{\sum_{k} S_k}{\sum_{k} N_k}

        .. math::

           \sigma = \sqrt{\max\left(0, \frac{\sum_{k} SS_k}{\sum_{k} N_k} - \mu^2\right)}

    Args:
        img_dir (str): Path to the directory containing the medical image collection.
        ext (str): Target file extension to filter. Supported values: '.dcm', '.dicom', 
            '.npz', '.npy'. Leading dots are automatically handled if omitted.
        nonzero (bool, optional): If True, isolates statistics calculations exclusively to 
            non-zero pixels. Defaults to True.

    Returns:
        Tuple[float, float, Dict]: A tuple containing three statistical payloads:

            * **mean** (float): The calculated global dataset mean intensity.
            * **std** (float): The calculated global dataset standard deviation.
            * **report_metrics** (Dict): A summary dictionary containing performance logs:
            
                * 'total_discovered' (int): Count of matching files found.
                * 'total_analyzed' (int): Count of successfully parsed files.
                * 'total_failed' (int): Count of failed/corrupted files.
                * 'failed_files' (List[str]): List of paths that raised processing errors.

    Raises:
        ValueError: If an unsupported extension is specified, no matching files are discovered,
            or the accumulated pixel count across the entire dataset is zero.
    """
    sum_pixels = 0.0
    sum_squared = 0.0
    num_pixels = 0
    
    if not ext.startswith('.'):
        ext = '.' + ext
        
    list_ext = ['.dcm', '.dicom', '.npz', '.npy']
    if ext.lower() in list_ext:
        img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(ext.lower())]
        img_paths = [os.path.join(img_dir, name) for name in img_names]
    else:
        raise ValueError("We only handle '.dcm', '.dicom', '.npz', and '.npy' extensions.")
    
    if not img_paths:
        raise ValueError(f"No files with extension '{ext}' found in directory: {img_dir}")
    
    successful_files = []
    failed_files = []
    
    logger.info(f"Starting parallel processing of {len(img_paths)} files...")

    with ProcessPoolExecutor(max_workers=None) as executor:
        results = executor.map(process_single_image, img_paths, [ext] * len(img_paths), [nonzero] * len(img_paths))
        
        for img_path, (p_sum, p_sq_sum, p_size, success) in zip(img_paths, tqdm(results, total=len(img_paths), desc="Processing Statistics", unit="img")):
            if success:
                sum_pixels += p_sum
                sum_squared += p_sq_sum
                num_pixels += p_size
                successful_files.append(img_path)
            else:
                failed_files.append(img_path)
                
    if num_pixels == 0:
        raise ValueError("No valid pixel data discovered within the target dataset to calculate statistics.")
        
    mean = sum_pixels / num_pixels
    variance = max(0.0, (sum_squared / num_pixels) - (mean ** 2))
    std = float(np.sqrt(variance))

    report_metrics = {
        "total_discovered": len(img_paths),
        "total_analyzed": len(successful_files),
        "total_failed": len(failed_files),
        "failed_files": failed_files
    }

    return mean, std, report_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset analysis tool: Compute statistics or convert DICOM images to NPZ format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

    # Command: stat
    stat_parser = subparsers.add_parser("stat", help="Compute global mean and standard deviation of the dataset.")
    stat_parser.add_argument("-i", "--input", type=str, required=True, help="Path to the training images directory.")
    stat_parser.add_argument("--ext", type=str, required=True, help="Extension of the training images (.dcm, .dicom, .npz, .npy).")
    stat_parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the execution text report (e.g., report.txt).")

    # Command: convert
    convert_parser = subparsers.add_parser("convert", help="Convert DICOM files to compressed .npz files.")
    convert_parser.add_argument("-i", "--input", required=True, help="Path to input directory containing DICOM files.")
    convert_parser.add_argument("-o", "--output", required=True, help="Path to output directory where results will be saved.")
    convert_parser.add_argument("-e", "--extension", required=False, default=".dicom", help="Target file extension filter.")

    args = parser.parse_args()

    try:
        if args.command == "convert":
            root_input = args.input
            root_save = args.output
            ext = args.extension

            if not ext.startswith('.'):
                ext = '.' + ext

            os.makedirs(root_save, exist_ok=True)
            is_zip = zipfile.is_zipfile(root_input)

            if is_zip:
                logger.info(f"Archive ZIP détectée : {root_input}")

                with zipfile.ZipFile(root_input, 'r') as z:
                    dicom_files = [
                        name for name in z.namelist()
                        if name.lower().endswith(ext.lower()) and not name.endswith('/')
                    ]
            else:
                dicom_files = [
                    os.path.join(root_input, file) 
                    for file in os.listdir(root_input) 
                    if file.lower().endswith(ext.lower())
                ]

            if not dicom_files:
                logger.info(f"No files matching extension '{ext}' were discovered inside: {root_input}")
                exit(0)

            max_workers = os.cpu_count() or 1
            logger.info(f"Beginning DICOM conversion for extension '{ext}' using {max_workers} CPU cores...")
            
            failed_jobs: List[Tuple[str, str]] = []

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                if is_zip:
                    futures = [
                            executor.submit(process_dicom, f, root_save, zip_path=root_input) 
                            for f in dicom_files
                    ]
                else:
                    futures = [executor.submit(process_dicom, f, root_save) for f in dicom_files]

                for future in tqdm(as_completed(futures), total=len(futures), desc="Converting DICOMs"):
                    dicom_path, success, error_msg = future.result()
                    if not success:
                        failed_jobs.append((dicom_path, str(error_msg)))
            
            successful_count = len(dicom_files) - len(failed_jobs)
            logger.info(f"Conversion complete: {successful_count}/{len(dicom_files)} files succeeded.")
            
            if failed_jobs:
                logger.warning(f"{len(failed_jobs)} files failed to process:")
                for dicom_path, error_msg in failed_jobs:
                    logger.warning(f" - {os.path.basename(dicom_path)}: {error_msg}")

        elif args.command == "stat":
            logger.info(f"Beginning dataset statistical analysis for extension '{args.ext}'...")

            global_mean, global_std, report = mean_std_fast(args.input, args.ext)
            
            logger.info(f"Calculated Global Mean: {global_mean:.4f}")
            logger.info(f"Calculated Global Std:  {global_std:.4f}")
            
            # Ensure path to the output report file exists
            output_dir = os.path.dirname(os.path.abspath(args.output))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("=== DATASET ANALYSIS REPORT ===\n")
                f.write(f"Directory analyzed: {args.input}\n")
                f.write(f"Target extension:   {args.ext}\n\n")
                f.write(f"Total files found:  {report['total_discovered']}\n")
                f.write(f"Successfully processed: {report['total_analyzed']}\n")
                f.write(f"Failed to process:      {report['total_failed']}\n\n")
                f.write("=== RESULTS ===\n")
                f.write(f"Global Mean: {global_mean:.6f}\n")
                f.write(f"Global Std:  {global_std:.6f}\n\n")
                
                if report['failed_files']:
                    f.write("=== FAILED FILES LOG ===\n")
                    for failed_path in report['failed_files']:
                        f.write(f"- {failed_path}\n")
                else:
                    f.write("All files processed successfully without errors.\n")
                    
            logger.info(f"Analysis complete. Report successfully saved to: {args.output}")

    except Exception as e:
        logger.critical(f"Dataset operation failed: {e}", exc_info=True)