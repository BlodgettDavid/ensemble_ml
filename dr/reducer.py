# dr/reducer.py
# Authors: David Blodgett and Microsoft Copilot
# Description: Applies unsupervised dimensionality reduction (PCA) and saves variance plot.
#              Returns a transformed DataFrame with labels preserved.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA

def reduce_dimensionality(df, label_col="target", n_components=2):
    """
    Applies PCA to reduce dataset dimensionality and saves variance plot.

    Parameters:
        df (pd.DataFrame): Full dataset with features + label
        label_col (str): Name of label column to preserve
        n_components (int): Number of PCA components to retain

    Returns:
        pd.DataFrame: PCA-reduced features + original label column
    """
    # Set plot directory for DR outputs
    plot_dir = os.path.join("plots", "dr")
    os.makedirs(plot_dir, exist_ok=True)

    features = df.drop(columns=[label_col])
    labels = df[label_col]

    pca = PCA(n_components=min(n_components, features.shape[1]))
    reduced = pca.fit_transform(features)

    pc_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(reduced, columns=pc_cols)
    df_pca[label_col] = labels.reset_index(drop=True)

    # Scree plot showing explained variance
    plt.figure(figsize=(6, 4))
    sns.barplot(x=pc_cols, y=pca.explained_variance_ratio_)
    plt.title("PCA Explained Variance Ratio")
    plt.ylabel("Ratio")
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{label_col}_pca_variance.png")
    plt.savefig(plot_path)
    plt.close()
    print(f" Saved variance plot to: {plot_path}")

    return df_pca

# Test block
if __name__ == "__main__":
    import sys
    import os

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(ROOT)

    from datasets.load_data import load_iris_data

    X_train, _, y_train, _ = load_iris_data()
    df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    df["target"] = y_train

    df_reduced = reduce_dimensionality(df, label_col="target", n_components=2)
    print(df_reduced.head())
