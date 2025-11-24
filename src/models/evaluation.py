# Voice Speaker Recognition - Model Evaluation & Metrics Module
"""
This module handles:
    1. Evaluating the trained CNN model on the test dataset.
    2. Computing detailed classification metrics including:
        - Accuracy, Precision, Recall, F1-score, ROC-AUC
        - Classification Report & Confusion Matrix
    3. Saving the computed metrics to a structured JSON file.

It is designed for CNN-based speaker recognition models
trained on Mel-spectrogram features.

Example:
    >>> from src.training.evaluate_model import evaluate_model
    >>> metrics = evaluate_model(model, history, x_test, y_test)

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
    roc_auc_score, f1_score, recall_score, precision_score,
    accuracy_score, classification_report, confusion_matrix
)


# ------------------ Logger Setup ------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def evaluate_model(model, history, x_test, y_test) -> dict:
    """
    Evaluate the trained model on the test dataset and compile evaluation metrics.

    Args:
        model: Trained Keras model.
        history: Training history object from model.fit().
        x_test: Test dataset features.
        y_test: Test dataset labels.

    Returns:
        dict: Dictionary containing all evaluation metrics.
    """

    # ---------------------- Evaluation -------------------------

    try:
        logger.info("Evaluating model on test data...")
        x_test = x_test.reshape(-1, 64, 188, 1)
        y_test = y_test.reshape(-1, 1)

        test_loss, test_accuracy, * \
            _ = model.evaluate(x_test, y_test, verbose=0)

        logger.info(
            f"Test Metrics → Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

        y_pred_probs = model.predict(x_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        logger.debug("Computing performance metrics...")

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

        for key in metrics:
            metrics[key] = float(f"{metrics[key]:.5f}") if isinstance(
                metrics[key], (float, np.floating)) else metrics[key]

        logger.debug("Evaluation metrics computed successfully.")

        save_metrics(metrics)
        logger.info("✅ Evaluation metrics saved successfully.")

        return metrics

    except Exception as e:
        logger.error(
            f"Error occurred during model evaluation: {e}", exc_info=True)
        raise RuntimeError("Model evaluation failed.") from e


# ---------------------------------------------------------
# Run module independently for testing
# ---------------------------------------------------------
if __name__ == "__main__":
    ...
