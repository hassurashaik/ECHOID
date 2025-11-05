# Voice Speaker Recognition - Metrics Logging Utility
"""
This module provides functions to save evaluation and training metrics
into a versioned JSON file located within the model's artifact directory.

The metrics logged include:
    - Training & validation loss
    - Training & validation accuracy
    - Precision, recall, F1-score, AUC
    - Timestamp + model version metadata

This ensures reproducibility, experiment tracking, and version comparability.

Example:
--------
>>> from src.utils.metrics_utils import save_metrics
>>> save_metrics({"accuracy": 0.95, "loss": 0.12})

Output:
-------
models/cnn_model_<version>/metrics.json

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Metrics Export & Experiment Tracking
License: MIT
"""


import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from . import config_utils as cfg


# --------- Module Logger ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =============================================================
# Metrics Saving Function
# =============================================================

def save_metrics(metrics: dict) -> None:
    """
    Save model training/evaluation metrics to a JSON file inside the model's
    versioned artifact directory.

    Args:
        metrics (dict):
            Dictionary containing evaluation metrics such as:
            accuracy, loss, precision, recall, f1_score, roc_auc, etc.

    Raises:
        RuntimeError: If saving metrics fails.
    """
    try:
        # Load current configuration
        config = cfg.read_config()
        version = config.get("version", "v1")

        # Resolve project root directories
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        save_dir_rel = config["paths"].get(
            "save_dir", f"models/cnn_model_{version}")
        save_dir = (PROJECT_ROOT / save_dir_rel).resolve()

        # Ensure model directory exists
        save_dir.mkdir(parents=True, exist_ok=True)

        # Build structured file path
        file_path = save_dir / "metrics.json"

        # Build structured metrics log
        metrics_record = {
            "version": config.get("version", "v1"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": config["training"].get("epochs", 20),
            **metrics
        }

        # Write metrics to JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metrics_record, f, indent=4)

        logger.info(f"✅ Metrics saved successfully to {file_path}")

    except Exception as e:
        logger.error(f"❌ Failed to save metrics: {e}", exc_info=True)
        raise RuntimeError("Metrics saving failed") from e


# ---------------------------------------------------------
# Standalone Test (Safe Execution)
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
