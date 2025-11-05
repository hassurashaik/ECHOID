# Voice Speaker Recognition - Configuration Utilities
"""
This module provides helper functions for reading, writing, saving,
and version-managing the project's YAML configuration file (`config.yaml`).

Features:
---------
- Safe loading and dumping of YAML with preserved formatting (ruamel.yaml)
- Centralized configuration access across training, inference, and models
- Automatic version bumping (v1 → v2 → v3) with corresponding model directory updates
- Consistent global config path resolution

Behavior:
---------
✅ `read_config()`      → Load config as dictionary.
✅ `write_config()`     → Save provided config back to YAML.
✅ `save_config()`      → Store versioned config snapshot in /config directory.
✅ `bump_version()`     → Auto-increment version + update `paths.save_dir`.

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Project Configuration Management
License: MIT
"""


import os
import re
import logging
from typing import Any, Dict
from pathlib import Path
from ruamel.yaml import YAML


# ------------------ YAML Setup ------------------
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)


# ------------------ Paths ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ------------------ Logger Setup ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =============================================================
# Configuration Management Functions
# =============================================================

def read_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Read and return the configuration YAML file as a dictionary.

    Args:
        path (Path): File path to YAML configuration.

    Returns:
        dict: Parsed YAML contents.
    """
    try:
        if not path.exists():
            raise FileNotFoundError(f"❌ Config file not found at: {path}")

        with open(path, "r") as f:
            config = yaml.load(f)

        return config
    except Exception as e:
        logger.error(f"❌ Failed to read config: {e}", exc_info=True)
        raise


# ---------------------------Write Config----------------------------

def write_config(config: Dict[str, Any], path: str = CONFIG_PATH) -> None:
    """
    Write updated configuration dictionary back to config.yaml.

    Args:
        config (dict): Updated config data.
        path (Path): Save location.
    """
    try:
        with open(path, "w") as f:
            yaml.dump(config, f)

    except Exception as e:
        logger.error(f"❌ Failed to write config: {e}", exc_info=True)
        raise    


# ---------------------------Save Config-----------------------------

def save_config() -> None:
    """
    Save current config state into a versioned config snapshot.
    Example: config_v3.yaml stored inside /config.
    """
    try:
        config = read_config()
        version = config.get("version", "v1")
        path = (PROJECT_ROOT / "config" / f"config_{version}.yaml").resolve()   

        with open(path, "w") as f:
            yaml.dump(config, f)

    except Exception as e:
        logger.error(f"❌ Failed to save versioned config snapshot: {e}", exc_info=True)
        raise


# ---------------------------Bump Version----------------------------

def bump_version(path: str = CONFIG_PATH) -> str:
    """
    Auto-increment version (v1 → v2 → v3) and update model save directory accordingly.

    Returns:
        str: New version label.
    """
    try:
        config = read_config(path)
        current_version = config.get("version", "v1")
        
        match = re.match(r"v(\d+)", current_version)
        if not match:
            raise ValueError(f"Invalid version format: {current_version}")
        
        new_version = f"v{int(match.group(1)) + 1}"

        #Update model save directory
        old_save_dir = config["paths"].get("save_dir", f"models/cnn_model_{current_version}")
        new_save_dir = old_save_dir.replace(current_version, new_version)

        config["version"] = new_version
        config["paths"]["save_dir"] = new_save_dir

        write_config(config, path)

        logger.info(f"🔄 Version incremented: {current_version} → {new_version}")
        logger.info(f"📁 Model save directory updated: {old_save_dir} → {new_save_dir}")

        return new_version

    except Exception as e:
        logger.error(f"❌ Failed to bump version: {e}", exc_info=True)
        raise

# ---------------------------------------------------------
# Standalone Test (Safe Execution)
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
