import argparse
from pathlib import Path
import random
import numpy as np
import torch
from monai.utils import set_determinism

from src.constants import CONFIG_FILE_PATH
from src.pipelines.cls_pipeline import ClsPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def init_reproducibility(seed: int) -> None:
    """Configures global computing frameworks to guarantee strict deterministic execution.

    Locks random states across the core Python runtime, NumPy arrays, PyTorch CPU/GPU 
    backends, and specialized MONAI transform workers.

    Args:
        seed (int): The target seed value to bind to random number generators.
    """
    # 1. Standard Python Ecosystem & CPU Computations
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 2. Parallel GPU / CUDA Computations
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    
    # 3. Restrict CuDNN Convolution Algorithmic Non-Determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)


def main() -> None:
    """Main execution entry point (CLI) for the Medical Image Processing Framework.

    Orchestrates application execution paths using two distinct subcommands:
    1. `train`   : Initializes single/multi-GPU distributed model optimization loops driven by a structured YAML configuration profile.
    2. `predict` : Runs automated out-of-sample batch inference pipelines.

    Usage:
        # Launch training using a specific experimental configuration blueprint
        python main.py train --config config/exp_v1.yaml --device cuda:0

        # Execute batch inference processing on raw DICOM volumes
        python main.py predict --input "/data/raw" --output "/data/processed" --tolerance 1.5

    Raises:
        Exception: Catches and logs top-level execution crashes to prevent silent pipeline failures in production automation workers.
    """
    parser = argparse.ArgumentParser(
        description="Medical Image Analysis Framework - Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments accessible across all subcommands
    parser.add_argument(
        "--config", type=Path, default=CONFIG_FILE_PATH,help="Path leading to the system environment configuration YAML file."
    )

    # Subparser orchestration routing train and predict branches separately
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- Subparser Configuration: TRAIN ---
    train_parser = subparsers.add_parser("train", help="Launch the deep model training pipeline.")
    train_parser.add_argument(
        "--device",  type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target processing hardware context (e.g., 'cpu', 'cuda', 'cuda:1')."
    )
    train_parser.add_argument(
        "--seed", type=int, default=None, help="Target integer seed used to enforce exact transformation reproducibility."
    )

    # --- Subparser Configuration: PREDICT (Inference) ---
    predict_parser = subparsers.add_parser("predict", help="Execute batch inference workflows over a folder.")
    predict_parser.add_argument(
        "--input", type=str, required=True, help="Path pointing to the folder containing raw source images."
    )
    predict_parser.add_argument(
        "--output", type=str, required=True, help="Destination directory where output files will be saved."
    )
    predict_parser.add_argument(
        "--tolerance", type=float, default=0.5, help="Operational filtering processing tolerance threshold (Scale: 0.1 to 3.0)."
    )
    predict_parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target hardware engine running the inference forward passes."
    )

    args = parser.parse_args()

    # Route back to root help printout if no execution action subcommand is provided
    if args.command is None:
        parser.print_help()
        return

    try:
        # Initialize pipeline context with safely routed global configuration path
        pipeline = ClsPipeline(config_path=args.config)

        if args.command == "train":
            target_device = torch.device(args.device)
            logger.info(f"Initializing distributed train orchestration on target hardware: {target_device}")

            if args.seed is not None:
                logger.info(f"Enforcing strict operational determinism using user seed: {args.seed}")
                init_reproducibility(args.seed)
                
            pipeline.train()
        
        elif args.command == "predict":
            target_device = torch.device(args.device)
            logger.info(f"Initializing batch prediction workflow. Source Directory: {args.input}")
            
            # Forward execution paths directly onto targeted compute hardware context
            pipeline.predict(device=target_device)

    except Exception as e:
        logger.error(
            f"Fatal exception caught during framework execution under command [{args.command}]: {str(e)}", 
            exc_info=True
        )


if __name__ == "__main__":
    main()