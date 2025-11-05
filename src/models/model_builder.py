# Voice Speaker Recognition - CNN Model Builder
"""
This module dynamically constructs and compiles a CNN model
based on the configuration defined in `config.yaml`.

It supports automatic architecture creation from configuration
lists (filters, kernel sizes, pool sizes, dense units, etc.) and
handles compilation with chosen optimizer, loss function, and metrics.

The class is designed for flexibility and reproducibility across versions,
where each model version (v1, v2, etc.) is managed via the global config.

Notes
-----
- Logs important stages of model creation and compilation.
- Reads configurations directly from `config.yaml` through `config_utils`.

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: CNN Model Builder
License: MIT
"""


import os
import logging
from src.utils import config_utils as cfg


# ------------------ TensorFlow Environment Config ------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow C++ logs


# ------------------ Module Logger ------------------
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
# CNN Model Builder Class
# =============================================================

class ModelBuilder:
    """
    ModelBuilder
    ----------------
    A class that dynamically builds a CNN model using parameters
    defined in `config.yaml`.

    It automatically constructs convolutional, pooling, dropout, and dense
    layers based on provided lists in the configuration file.

    Attributes
    ----------
    model_cfg : dict
        Dictionary containing CNN model parameters.
    version : str
        Version identifier of the current model (e.g., "v1", "v2").

    Example
    -------
    >>> from src.models.model_builder import ModelBuilder
    >>> builder = ModelBuilder()
    >>> model = builder.build_model()
    >>> model.summary()
    """

    def __init__(self):
        """
        Initialize the CNN model builder by loading config and version.
        """
        try:
            config = cfg.read_config()
            self.model_cfg = config["model"]
            self.version = config["version"]
            logger.info(
                f"Initialized ModelBuilder (version={self.version}) successfully.")
        except Exception as e:
            logger.error(f"Error initializing ModelBuilder: {e}")
            raise

    # ---------------------------------------------------------
    def build_model(self):
        """
        Dynamically build and compile a CNN model based on the config file.

        Returns
        -------
        model : keras.Model
            A compiled Keras Sequential CNN model ready for training.

        Raises
        ------
        KeyError
            If required keys are missing in the model configuration.
        ValueError
            If invalid optimizer or parameter lengths are provided.
        """

        try:
            logger.info("Building CNN model dynamically from configuration...")

            from keras.models import Sequential
            from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
            from keras.optimizers import Adam

            model = Sequential()

            # Extract parameters from config safely
            filters = self.model_cfg["filters"]
            kernel_sizes = self.model_cfg["kernel_sizes"]
            max_pool_sizes = self.model_cfg["max_pool_sizes"]
            dense_units = self.model_cfg["dense_units"]
            dropout_rates = self.model_cfg["dropout_rates"]
            input_shape = tuple(self.model_cfg["input_shape"])

            logger.debug(f"Model Filters: {filters}")
            logger.debug(f"Input Shape: {input_shape}")

            # Validate list lengths
            if not (len(filters) == len(kernel_sizes) == len(max_pool_sizes)):
                raise ValueError(
                    "Length mismatch among 'filters', 'kernel_sizes', and 'max_pool_sizes' lists."
                )

            # ------------------ Convolutional Blocks ------------------
            for i, (filter, kernel, pool) in enumerate(zip(filters, kernel_sizes, max_pool_sizes)):
                logger.debug(
                    f"Adding Conv2D layer {i+1}: filters={filter}, kernel_size={kernel}")

                if i == 0:
                    model.add(
                        Input(shape=input_shape)
                    )
                    model.add(
                        Conv2D(
                            filter,
                            kernel,
                            activation="relu",
                        )
                    )
                else:
                    model.add(
                        Conv2D(
                            filters=filter,
                            kernel_size=kernel,
                            activation="relu"
                        )
                    )

                model.add(MaxPooling2D(pool_size=pool))

                if i < len(dropout_rates):
                    model.add(Dropout(dropout_rates[i]))

            # ------------------ Dense Layers ------------------
            model.add(Flatten())
            for i, units in enumerate(dense_units):
                logger.debug(f"Adding Dense layer {i+1}: units={units}")
                model.add(Dense(units=units, activation="relu"))
                if i + len(filters) < len(dropout_rates):
                    model.add(Dropout(dropout_rates[i + len(filters)]))

            # ------------------ Output Layer ------------------
            # Binary classification
            model.add(Dense(units=1, activation="sigmoid"))

            # ------------------ Compile Model ------------------
            if self.model_cfg.get("optimizer").lower() == "adam":
                optimizer = Adam()
            else:
                raise ValueError(
                    f"Unsupported optimizer specified in the configuration: {self.model_cfg.get('optimizer')}"
                )

            metrics = self.model_cfg.get("metrics", ["accuracy"])

            loss = self.model_cfg.get("loss", "binary_crossentropy")

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            logger.info(
                f"Model compiled successfully with optimizer={self.model_cfg.get('optimizer')}, loss={loss}, metrics={metrics}"
            )

            return model

        except KeyError as e:
            logger.error(f"Missing key in model configuration: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid configuration parameter: {e}")
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error occurred while building the model: {e}"
            )
            raise


# ---------------------------------------------------------
# Run module independently for testing
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
