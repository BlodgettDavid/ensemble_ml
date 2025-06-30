# experiments/classification/linear/bagging_linear_clf.py
# Authors: David Blodgett and Microsoft Copilot
# Description: Executes a bagging ensemble of linear classifiers on the Iris dataset
#              using multiple voting strategies, with centralized logging and evaluation.

import sys
import os

# Add root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(ROOT_DIR)

from config import logger, RESULTS_DIR
from models.ensemble_models import EnsembleModel
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from utils.evaluation import evaluate_classification
from datasets.load_data import load_iris_data

from utils.visualization import plot_confusion_matrix
from utils.visualization import plot_strategy_accuracies

import csv
from datetime import datetime


class BaggingLinearClassifierExperiment:
    def __init__(self):
        logger.info("Initializing BaggingLinearClassifierExperiment...")
        self.models = self._define_models()
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

    def _define_models(self):
        logger.info("Defining base linear classifiers for ensemble...")
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RidgeClassifier": RidgeClassifier(),
            "SGDClassifier": SGDClassifier(max_iter=1000, tol=1e-3),
        }

    def run(self, X_train, X_test, y_train, y_test):
        logger.info("Starting ensemble strategy comparison...")

        strategy_scores = {}
        for strategy in ["hard_voting", "soft_voting"]:
            logger.info(f"Running strategy: {strategy}")
            print(f"\nStrategy: {strategy}")

            # 🎯 Filter models that support predict_proba for soft voting
            if strategy == "soft_voting":
                usable_models = {
                    name: model for name, model in self.models.items()
                    if hasattr(model, "predict_proba")
                }
            else:
                usable_models = self.models

            logger.info(f"Models used for strategy '{strategy}': {list(usable_models.keys())}")
            try:
                ensemble = EnsembleModel(models=usable_models, strategy=strategy)
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)

                acc, cm, report = evaluate_classification(y_test, y_pred)
                plot_confusion_matrix(y_test, y_pred, title=f"{strategy} Confusion Matrix")
                strategy_scores[strategy] = acc
               
                
                logger.info(f"Model accuracy: {acc:.4f}")
                logger.info(f"Confusion matrix:\n{cm}")
                logger.info("Classification report:\n" + report)

                results_path = os.path.join(self.results_dir, f"results_{strategy}.txt")
                with open(results_path, "w") as f:
                    f.write(f"Strategy: {strategy}\n")
                    f.write(f"Models Used: {list(usable_models.keys())}\n")
                    f.write(f"Accuracy: {acc:.4f}\n\n")
                    f.write("Confusion Matrix:\n")
                    f.write(str(cm) + "\n\n")
                    f.write("Classification Report:\n")
                    f.write(report)

                logger.info(f"Saved results to {results_path}")
                csv_path = os.path.join(self.results_dir, "summary_results.csv")
                write_header = not os.path.exists(csv_path)

                with open(csv_path, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if write_header:
                        writer.writerow(["Timestamp", "Strategy", "Accuracy", "Models Used"])
                    writer.writerow([
                        datetime.now().isoformat(timespec='seconds'),
                        strategy,
                        f"{acc:.4f}",
                        ", ".join(usable_models.keys())
                    ])
            except Exception as e:
                logger.error(f"Strategy {strategy} failed: {str(e)}")
                print(f" Strategy {strategy} encountered an error. See log for details.")
        plot_strategy_accuracies(strategy_scores)

if __name__ == "__main__":
    experiment = BaggingLinearClassifierExperiment()
    experiment.run()
