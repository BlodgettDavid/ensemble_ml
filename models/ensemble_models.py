# models/ensemble_model.py
# Author: Microsoft Copilot
# Description: Core logic for EnsembleModel class used in classification and regression tasks.
# This file defines an object-oriented interface to add, train, and predict using multiple models.

from typing import List, Union
import numpy as np

class EnsembleModel:
    def __init__(self, strategy: str = "hard_voting"):
        """
        Initialize the ensemble model.

        :param strategy: The strategy to use for combining predictions.
                         Options: 'hard_voting', 'soft_voting', 'averaging'
        """
        self.models: List = []
        self.strategy = strategy
        self.is_classifier = True  # Will adjust after first fit

    def add_model(self, model):
        """
        Add a model to the ensemble.

        :param model: Any sklearn-compatible estimator (must implement fit/predict)
        """
        self.models.append(model)

    def fit(self, X, y):
        """
        Fit all models on the same training data.

        :param X: Feature matrix
        :param y: Target labels
        """
        for model in self.models:
            model.fit(X, y)

        # Try to infer whether it's classification or regression based on first model
        sample_pred = self.models[0].predict(X[:5])
        self.is_classifier = len(np.unique(sample_pred)) <= len(np.unique(y))

    def predict(self, X):
        """
        Predict using the ensemble strategy selected at initialization.

        :param X: Feature matrix
        :return: Combined predictions from the ensemble
        """
        if not self.models:
            raise ValueError("No models in the ensemble. Add models before predicting.")

        predictions = [model.predict(X) for model in self.models]
        predictions = np.array(predictions)  # shape: (n_models, n_samples)

        if self.strategy == "hard_voting":
            # Majority vote for classification
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions.astype(int))
        elif self.strategy == "soft_voting":
            # Average predicted probabilities
            probas = [model.predict_proba(X) for model in self.models]
            avg_proba = np.mean(probas, axis=0)
            return np.argmax(avg_proba, axis=1)
        elif self.strategy == "averaging":
            # Mean for regression
            return np.mean(predictions, axis=0)
        else:
            raise NotImplementedError(f"Strategy '{self.strategy}' is not supported.")
