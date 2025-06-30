# eda/perform_eda.py
# Authors: David Blodgett and Microsoft Copilot
# Description: Performs exploratory data analysis (EDA) on a given dataset,
#              including pairplots and class distribution, and returns a flag
#              recommending whether dimensionality reduction (DR) should be considered.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

# Add root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

def perform_eda(df, dataset_name="dataset", save_dir="results/eda"):  
    """
    Performs exploratory data analysis on a DataFrame:
    - Displays basic info
    - Saves pairplot and class distribution plots
    - Analyzes feature count and correlation
    - Returns a recommendation for DR with reasoning
    """
    
    #os.makedirs(save_dir, exist_ok=True)
    # === Set plot directory ===
    plot_dir = os.path.join("plots", "eda")
    os.makedirs(plot_dir, exist_ok=True)
    
    print("\n Head of dataset:")
    print(df.head())

    print("\n Dataset summary:")
    print(df.describe())

    print("\n️ Data info:")
    print(df.info())

    # === Visuals ===
    label_col = df.columns[-1]
    features = df.iloc[:, :-1]

    # Pairplot
    pairplot_path = os.path.join(plot_dir, f"{dataset_name}_pairplot.png")
    sns.pairplot(df, hue=label_col)
    plt.suptitle(f"{dataset_name} Pairplot", y=1.02)
    plt.tight_layout()
    plt.savefig(pairplot_path)
    plt.close()

    # Class distribution
    class_counts = df[label_col].value_counts()
    classdist_path = os.path.join(plot_dir, f"{dataset_name}_class_distribution.png")
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title(f"{dataset_name} Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(classdist_path)
    plt.close()

    # === DR Recommendation Logic ===
    num_features = features.shape[1]
    MAX_FEATURES = 10

    # Check for high correlation
    corr_matrix = features.corr().abs()
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    high_corr = (corr_matrix.where(upper) > 0.95).sum().sum()

    # Decide
    if num_features > MAX_FEATURES:
        reason = f"{num_features} features exceeds threshold of {MAX_FEATURES}"
        recommend = True
    elif high_corr > 0:
        reason = f"{int(high_corr)} highly correlated feature pairs (ρ > 0.95)"
        recommend = True
    else:
        reason = "Feature count and correlation within acceptable limits"
        recommend = False

    return {
        "recommend_dr": recommend,
        "reason": reason,
        "num_features": num_features,
        "high_corr_pairs": int(high_corr)
    }

# Standalone test block (optional)
if __name__ == "__main__":
    from datasets.load_data import load_iris_data

    X_train, _, y_train, _ = load_iris_data()
    df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    df["target"] = y_train

    result = perform_eda(df, dataset_name="iris")

    print(f"\n Recommend DR? {'Yes' if result['recommend_dr'] else 'No'} — {result['reason']}")