# utils/visualization.py
# Authors: David Blodgett and Microsoft Copilot
# Description: Contains visualization utilities for model evaluation results,
#              including confusion matrix plots and accuracy comparison graphs.

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_plot_path(subdir, filename):
    """
    Constructs and ensures the subdirectory exists inside the 'plots' folder.

    Parameters:
        subdir (str): Subfolder name under 'plots'
        filename (str): File name for the plot

    Returns:
        str: Full path to save the plot
    """
    plot_dir = os.path.join("plots", subdir)
    os.makedirs(plot_dir, exist_ok=True)
    return os.path.join(plot_dir, filename)

def plot_strategy_accuracies(strategy_scores):
    """
    Plots a bar chart of accuracies per voting strategy.

    Parameters:
    strategy_scores (dict): Strategy name mapped to accuracy score.
    """
    strategies = list(strategy_scores.keys())
    accuracies = list(strategy_scores.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(strategies, accuracies, color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Voting Strategy")

    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", 
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    filename = f"{'_'.join(strategies)}_accuracy.png"
    path = get_plot_path("utils", filename)
    plt.savefig(path)
    plt.close()
    print(f" Saved strategy accuracy plot to: {path}")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Displays a heatmap of the confusion matrix.

    Parameters:
    y_true (array-like): Ground truth labels.
    y_pred (array-like): Predicted labels.
    title (str): Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    filename = f"{title.replace(' ', '_').lower()}.png"
    path = get_plot_path("utils", filename)
    plt.savefig(path)
    plt.close()
    print(f" Saved confusion matrix to: {path}")
