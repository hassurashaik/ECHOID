# Voice Speaker Recognition - Model Evaluation & Metrics Module
"""
This module handles:
    1. Evaluating the trained CNN model on the test dataset.
    2. Computing detailed classification metrics including:
        - Accuracy, Precision, Recall, F1-score, ROC-AUC
        - Classification Report & Confusion Matrix
    3. Saving the computed metrics to a structured JSON file.

It is designed for CNN-based binary speaker recognition models
trained on Mel-spectrogram features.

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Evaluation & Metrics Saving
License: MIT
"""

from src.utils.metrics_utils import save_metrics
import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ------------------ Logger Setup ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def evaluate_model(model, history, x_test, y_test) -> dict:
    """
    Evaluate the trained model on the test dataset and compile evaluation metrics.

    Args:
        model: Trained Keras model.
        history: Training history object from model.fit().
        x_test: Test dataset features (batched).
        y_test: Test dataset labels (batched).

    Returns:
        dict: Dictionary containing all evaluation metrics.
    """

    try:
        logger.info("Evaluating model on test data...")

        # --------------------------------------------------
        # Flatten batched data (CRITICAL FIX)
        # --------------------------------------------------
        x_test = x_test.reshape(-1, 64, 188, 1)
        y_test = y_test.reshape(-1)               # ✅ MUST be 1D

        # --------------------------------------------------
        # Model evaluation (Keras)
        # --------------------------------------------------
        test_loss, test_accuracy, *_ = model.evaluate(
            x_test, y_test, verbose=0
        )

        logger.info(
            f"Test Metrics → Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------
        y_pred_probs = model.predict(x_test).reshape(-1)  # ✅ 1D probabilities
        y_pred = (y_pred_probs > 0.5).astype(int)         # ✅ Binary predictions

        logger.debug("Computing performance metrics...")

        # --------------------------------------------------
        # Metrics (Binary-safe)
        # --------------------------------------------------
        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),

            "train_loss": float(history.history.get("loss", [np.nan])[-1]),
            "train_accuracy": float(history.history.get("accuracy", [np.nan])[-1]),
            "val_loss": float(history.history.get("val_loss", [np.nan])[-1]),
            "val_accuracy": float(history.history.get("val_accuracy", [np.nan])[-1]),

            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_probs)),

            "classification_report": classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        # --------------------------------------------------
        # Float formatting
        # --------------------------------------------------
        for key in metrics:
            if isinstance(metrics[key], (float, np.floating)):
                metrics[key] = float(f"{metrics[key]:.5f}")

        logger.debug("Evaluation metrics computed successfully.")

        # --------------------------------------------------
        # Save metrics
        # --------------------------------------------------
        save_metrics(metrics)
        logger.info("✅ Evaluation metrics saved successfully.")

        return metrics

    except Exception as e:
        logger.error(
            f"Error occurred during model evaluation: {e}", exc_info=True
        )
        raise RuntimeError("Model evaluation failed.") from e


# ---------------------------------------------------------
# Standalone execution (optional)
# ---------------------------------------------------------
if __name__ == "__main__":
    pass
