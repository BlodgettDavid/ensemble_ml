# utils/evaluation.py
# Authors: David Blodgett and Microsoft Copilot
# Description: Evaluation utilities for classification and regression models,
#              including metrics computation and optional console printing.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_classification(y_true, y_pred, verbose=True):
    """
    Computes accuracy, confusion matrix, and classification report.

    Parameters:
    y_true (array-like): Ground truth labels
    y_pred (array-like): Predicted labels
    verbose (bool): If True, prints results to console

    Returns:
    tuple: (accuracy: float, confusion_matrix: ndarray, report: str)
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    if verbose:
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)

    return acc, cm, report
