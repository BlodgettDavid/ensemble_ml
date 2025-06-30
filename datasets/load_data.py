# datasets/load_data.py
# Author: Microsoft Copilot
# Description: Provides utility functions for loading and preparing datasets.
#              Currently supports the Iris dataset and returns pre-split
#              training and testing sets for experimentation.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_iris_data(test_size=0.25, random_state=42):
    """
    Loads and splits the Iris dataset.

    Parameters:
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Seed for random number generator.

    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    iris = load_iris()
    return train_test_split(
        iris.data, iris.target, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=iris.target
    )