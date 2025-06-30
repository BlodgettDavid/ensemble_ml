# mainworkflow.py
# Authors: Dave Blodgett and Microsoft Copilot
# Description: Orchestrates full ML pipeline including EDA, DR, and ensemble experiments.

from eda.perform_eda import perform_eda
from dr.reducer import reduce_dimensionality
from datasets.load_data import load_iris_data
from experiments.classification.linear.bagging_linear_clf import BaggingLinearClassifierExperiment
import pandas as pd

def run_workflow():
    # === Step 1: Load data ===
    X_train, X_test, y_train, y_test = load_iris_data()
    label_col = "target"

    # === Step 2: Build training DataFrame for EDA/DR only ===
    df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    df_train[label_col] = y_train

    # === Step 3: Perform EDA on training data ===
    eda_result = perform_eda(df_train, dataset_name="iris")

    print(f"\n📌 DR Recommendation: {'Yes' if eda_result['recommend_dr'] else 'No'} — {eda_result['reason']}")

    # === Step 4: Dimensionality Reduction if recommended ===
    if eda_result["recommend_dr"]:
        # 🔹 Apply DR to training set
        df_train_reduced = reduce_dimensionality(df_train, label_col=label_col)

        # 🔹 Split into features and labels
        X_train = df_train_reduced.drop(columns=[label_col]).values
        y_train = df_train_reduced[label_col].values

        # 🔹 Fit PCA again on training set, apply it to test set using same transform
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        pca.fit(df_train.drop(columns=[label_col]))  # Fit on original training features

        df_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        X_test = pca.transform(df_test)

    # === Step 5: Run experiment ===
    experiment = BaggingLinearClassifierExperiment()
    experiment.run(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    run_workflow()