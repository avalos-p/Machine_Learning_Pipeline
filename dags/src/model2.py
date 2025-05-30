import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (precision_recall_fscore_support, 
                             roc_auc_score, average_precision_score,
                             confusion_matrix, RocCurveDisplay, 
                             PrecisionRecallDisplay)
from sklearn.pipeline import Pipeline as SklearnPipeline
import logging
import pickle
import matplotlib.pyplot as plt
import json
import os
from typing import Tuple, Dict, Any


class AdvancedModelTrainer:
    """
    Class for training and evaluating a model with hyperparameter optimization
    and using pipelines.
    """

    def __init__(self, config_path: str = 'config.json', logger: logging.Logger = None):
        """
        Initializes the trainer with configuration.

        Args:
            config_path (str): Path to the JSON configuration file.
            logger (logging.Logger): Logger for recording information.
        """
        self.config = self._load_config(config_path)
        self.logger = logger if logger else self._setup_logger()
        self.model = None # The complete pipeline will be the model
        self.metrics = None
        self.data = None
        self.X_test_data = None # To store X_test
        self.y_test_data = None # To store y_test

    def _setup_logger(self) -> logging.Logger:
        """Sets up a basic logger if one is not provided."""
        logger = logging.getLogger(__name__ + "Advanced")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Loads configuration from a JSON file."""
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found at {path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON configuration file at {path}")
            raise

    def load_data(self, filepath: str) -> None:
        """
        Loads data from CSV and checks for missing values.

        Args:
            filepath (str): Path to the CSV file.
        """
        try:
            self.data = pd.read_csv(filepath)
            if self.data.isnull().any().any():
                self.logger.warning("Warning: Data contains missing values. Consider imputing them.")
        except FileNotFoundError:
            self.logger.error(f"Data file not found at {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise

    def train(self, target_col: str) -> Tuple[Any, Dict[str, float]]:
        """
        Trains the model using a pipeline with GridSearchCV
        for hyperparameter optimization.

        Args:
            target_col (str): Name of the target column.

        Returns:
            Tuple: (trained model (pipeline), evaluation metrics)
        """
        if self.data is None:
            self.logger.error("Data has not been loaded. Call load_data() first.")
            raise ValueError("Data not loaded.")

        try:
            X = self.data.drop(columns=[target_col])
            y = self.data[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.get('test_size', 0.2),
                stratify=y, # Important for maintaining proportions in imbalanced datasets
                random_state=self.config.get('random_state', 42)
            )
            # Store X_test and y_test for later use in save_plots
            self.X_test_data = X_test
            self.y_test_data = y_test

            # Define the pipeline
            pipeline_steps = []

            # Base classifier (DecisionTree for now, could be made configurable)
            # Hyperparameters are set here
            fixed_hyperparams_key = f"fixed_hyperparameters_{self.config.get('model_type', 'default')}"
            fixed_hyperparams = self.config.get(fixed_hyperparams_key, {})
            # Remove 'classifier__' prefix if it exists in the config
            fixed_hyperparams_clean = {k.replace('classifier__',''): v for k,v in fixed_hyperparams.items()}

            classifier = DecisionTreeClassifier(random_state=self.config.get('random_state', 42), **fixed_hyperparams_clean)
            pipeline_steps.append(('classifier', classifier))

            current_pipeline = SklearnPipeline(steps=pipeline_steps) 

            # Define the parameter grid for GridSearchCV
            param_grid_key = f"param_grid_{self.config.get('model_type', 'default')}"
            param_grid = self.config.get(param_grid_key, {})

            if not param_grid:
                self.logger.warning(f"No param_grid found for {param_grid_key} in configuration. Using default classifier hyperparameters.")
                # Train directly if no grid
                current_pipeline.fit(X_train, y_train)
                self.model = current_pipeline
            else:
                self.logger.info(f"Starting GridSearchCV with param_grid: {param_grid}")
                search = GridSearchCV(
                    current_pipeline,
                    param_grid,
                    scoring='average_precision', # Optimize for PR AUC
                    cv=self.config.get('cv_splits', 5),
                    n_jobs=-1,
                    verbose=1 
                )
                search.fit(X_train, y_train)
                self.model = search.best_estimator_ 
                self.logger.info(f"Best parameters found by GridSearchCV: {search.best_params_}")
                self.logger.info(f"Best PR AUC (average_precision) in CV: {search.best_score_:.4f}")

            self.metrics = self.evaluate(X_test, y_test)

            self.save_model()
            self.save_metrics()
            self.save_plots() # No longer needs X_test, y_test as arguments

            return self.model, self.metrics

        except Exception as e:
            self.logger.error(f"Error in advanced training: {str(e)}", exc_info=True)
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluates the model (pipeline) with various metrics."""
        if self.model is None:
            self.logger.error("Model has not been trained. Call train() first.")
            raise ValueError("Model not trained.")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'precision_weighted': precision_recall_fscore_support(y_test, y_pred, average='weighted')[0],
            'recall_weighted': precision_recall_fscore_support(y_test, y_pred, average='weighted')[1],
            'f1_score_weighted': precision_recall_fscore_support(y_test, y_pred, average='weighted')[2],
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba), 
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        self.logger.info("\nAdvanced Model Evaluation Metrics:")
        for name, value in metrics.items():
            if name == 'confusion_matrix':
                self.logger.info(f"Confusion Matrix: {value}")
            else:
                self.logger.info(f"{name.replace('_', ' ').title()}: {value:.4f}")

        return metrics

    def save_model(self, path: str = None) -> None:
        """Saves the trained pipeline."""
        if self.model is None:
            self.logger.warning("No model to save.")
            return

        if path is None:
            path = self.config.get('output_model_path', 'advanced_model.pkl')

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        self.logger.info(f"Advanced model (pipeline) saved to {path} using pickle")

    def save_metrics(self, path: str = None) -> None:
        """Saves metrics to a JSON file."""
        if self.metrics is None:
            self.logger.warning("No metrics to save.")
            return

        if path is None:
            path = self.config.get('output_metrics_path', 'advanced_metrics.json')

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        serializable_metrics = {}
        for k, v in self.metrics.items():
            if isinstance(v, np.generic):
                serializable_metrics[k] = v.item()
            elif isinstance(v, list) and all(isinstance(i, list) for i in v): # For confusion matrix
                 serializable_metrics[k] = [[int(elem) for elem in row] for row in v]
            else:
                serializable_metrics[k] = v

        with open(path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        self.logger.info(f"Advanced metrics saved to {path}")

    def save_plots(self, plots_dir: str = None) -> None:
        """Generates and saves evaluation plots using the classifier from the pipeline."""
        if self.model is None:
            self.logger.warning("No model to generate plots.")
            return
        
        if self.X_test_data is None or self.y_test_data is None:
            self.logger.warning("Test data (X_test_data, y_test_data) not available. Run train() first or ensure test data is loaded.")
            return

        if plots_dir is None:
            plots_dir = self.config.get('output_plots_dir', 'plots_advanced')

        os.makedirs(plots_dir, exist_ok=True)

        # If the pipeline has a 'classifier' step, we use it.
        try:
            if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                classifier_for_plot = self.model.named_steps['classifier']
            elif hasattr(self.model, 'best_estimator_') and \
                 hasattr(self.model.best_estimator_, 'named_steps') and \
                 'classifier' in self.model.best_estimator_.named_steps: # If it's a GridSearchCV on a pipeline
                classifier_for_plot = self.model.best_estimator_.named_steps['classifier']
            else: 
                classifier_for_plot = self.model
                self.logger.warning("Could not find 'classifier' step in the pipeline for plotting, using the full pipeline/model. This might fail if the pipeline is not a simple estimator.")

            # ROC Curve
            try:
                roc_display = RocCurveDisplay.from_estimator(self.model, self.X_test_data, self.y_test_data)
                roc_display.figure_.savefig(os.path.join(plots_dir, 'roc_curve_advanced.png'))
                plt.close(roc_display.figure_)
                self.logger.info(f"Advanced ROC curve saved to {plots_dir}")
            except Exception as e:
                self.logger.error(f"Error generating ROC curve: {e}")


            # Precision-Recall Curve
            try:
                pr_display = PrecisionRecallDisplay.from_estimator(self.model, self.X_test_data, self.y_test_data)
                pr_display.figure_.savefig(os.path.join(plots_dir, 'precision_recall_advanced.png'))
                plt.close(pr_display.figure_)
                self.logger.info(f"Advanced Precision-Recall curve saved to {plots_dir}")
            except Exception as e:
                self.logger.error(f"Error generating PR curve: {e}")

        except Exception as e:
            self.logger.error(f"Error preparing classifier for plots: {e}")

    def get_feature_importance(self) -> pd.Series:
        """Returns feature importances from the classifier in the pipeline."""
        if self.model is None:
            self.logger.error("Model not trained.")
            raise ValueError("Model not trained.")

        try:
            # Access the classifier within the pipeline
            if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                final_estimator = self.model.named_steps['classifier']
            elif hasattr(self.model, 'best_estimator_') and \
                 hasattr(self.model.best_estimator_, 'named_steps') and \
                 'classifier' in self.model.best_estimator_.named_steps: # If it's a GridSearchCV on a pipeline
                final_estimator = self.model.best_estimator_.named_steps['classifier']
            else:
                final_estimator = self.model # Assume the model is the classifier if not a pipeline with 'classifier'
                self.logger.warning("Could not find 'classifier' step in the pipeline for feature importance, using the full model/pipeline.")


            if hasattr(final_estimator, 'feature_importances_'):
                # If the pipeline included resampling, features might change.
                # It's safer to get feature names from the original DataFrame, assuming the classifier uses them.
                # This part might need adjustment if feature transformers are used in the pipeline.
                if self.data is not None and 'is_fraud' in self.data.columns: # Assuming target_col is 'is_fraud'
                    feature_names = self.data.drop(columns=['is_fraud']).columns
                else: # Fallback if we cannot get names
                    num_features = len(final_estimator.feature_importances_)
                    feature_names = [f"feature_{i}" for i in range(num_features)]
                    self.logger.warning("Could not determine feature names; using generic names.")

                return pd.Series(
                    final_estimator.feature_importances_,
                    index=feature_names, # Use feature names from the original X that entered the pipeline
                    name='importance'
                ).sort_values(ascending=False)
            else:
                self.logger.warning("Final classifier does not have 'feature_importances_' attribute.")
                return pd.Series(dtype=float, name='importance')
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return pd.Series(dtype=float, name='importance')

# Example of how it could be used (for direct script testing)
if __name__ == '__main__':
    # Create a basic logger for testing
    logger = logging.getLogger("AdvancedModelTrainerTest")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    # Create an example configuration file for testing
    mock_config_content = {
        "test_size": 0.3,
        "random_state": 42,
        "cv_splits": 2, # Faster for testing
        "model_type": "DecisionTree",
        "param_grid_DecisionTree": {
            "classifier__max_depth": [3, 5],
            "classifier__min_samples_split": [10, 20],
            "classifier__min_samples_leaf": [5, 10],
            "classifier__class_weight": [None, "balanced"]
        },
        "fixed_hyperparameters_DecisionTree": {
            "classifier__criterion": "gini"
        },
        "output_model_path": "temp/advanced_model.pkl",
        "output_metrics_path": "temp/advanced_metrics.json",
        "output_plots_dir": "temp/plots_advanced"
    }

    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)

    mock_config_path = "temp/mock_config_advanced.json"
    with open(mock_config_path, 'w') as f:
        json.dump(mock_config_content, f, indent=4)

    # Create example data
    from sklearn.datasets import make_classification
    X_sample, y_sample = make_classification(n_samples=200, n_features=5, n_informative=3,
                                             weights=[0.9, 0.1], random_state=42, flip_y=0.1)
    sample_df = pd.DataFrame(X_sample, columns=[f'feature_{i}' for i in range(X_sample.shape[1])])
    sample_df['is_fraud'] = y_sample
    mock_data_path = "temp/mock_fraud_data.csv"
    sample_df.to_csv(mock_data_path, index=False)

    logger.info("--- Starting AdvancedModelTrainer test ---")
    try:
        trainer = AdvancedModelTrainer(config_path=mock_config_path, logger=logger)
        trainer.load_data(mock_data_path)

        if trainer.data is not None:
            trained_model, metrics = trainer.train(target_col='is_fraud')
            logger.info(f"Training completed. Model: {trained_model}")
            logger.info(f"Metrics: {metrics}")

            if metrics.get('pr_auc', 0) > 0: # A very basic check
                 logger.info("PR AUC is greater than 0, basic test seems to work.")
            else:
                 logger.warning("PR AUC is 0 or not present, review training.")

            importances = trainer.get_feature_importance()
            logger.info("Feature importances:")
            logger.info(importances)
        else:
            logger.error("Failed to load data, cannot continue with training.")

    except Exception as e:
        logger.error(f"Error during AdvancedModelTrainer test: {e}", exc_info=True)
    finally:
        logger.info("--- AdvancedModelTrainer test finished ---")
