"""Model training with MLflow tracking."""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from typing import Any, Dict

from ..utils import logger, settings


class FraudDetector:
    """Fraud detection model with MLflow tracking."""
    
    def __init__(self, model_type: str = "xgboost", use_smote: bool = True):
        """
        Initialize fraud detector.
        
        Args:
            model_type: Type of model ("xgboost" or "lightgbm")
            use_smote: Whether to use SMOTE for class balancing
        """
        self.model_type = model_type
        self.use_smote = use_smote
        self.model = None
        self.smote = SMOTE(random_state=42) if use_smote else None
        
    def _get_model(self, params: Dict[str, Any] | None = None) -> Any:
        """Get model instance with parameters."""
        default_params = {
            "random_state": 42,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
        }
        
        if params:
            default_params.update(params)
        
        if self.model_type == "xgboost":
            default_params["scale_pos_weight"] = 50  # Handle imbalance
            return XGBClassifier(**default_params)
        elif self.model_type == "lightgbm":
            default_params["is_unbalance"] = True  # Handle imbalance
            return LGBMClassifier(**default_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        params: Dict[str, Any] | None = None,
        experiment_name: str = "fraud-detection"
    ) -> Dict[str, float]:
        """
        Train model with MLflow tracking.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            params: Model hyperparameters
            experiment_name: MLflow experiment name
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model_type not in ["xgboost", "lightgbm"]:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            logger.info(f"Training {self.model_type} model...")
            
            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("use_smote", self.use_smote)
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)
            
            # Handle class imbalance with SMOTE
            if self.use_smote and self.smote is not None:
                logger.info("Applying SMOTE...")
                X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
                logger.info(f"After SMOTE: {len(X_train_resampled):,} samples")
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
            
            # Train model
            self.model = self._get_model(params)
            self.model.fit(X_train_resampled, y_train_resampled)
            
            # Evaluate on training set
            train_metrics = self.evaluate(X_train, y_train, prefix="train")
            
            # Evaluate on validation set if provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val, prefix="val")
            
            # Log all metrics
            all_metrics = {**train_metrics, **val_metrics}
            for key, value in all_metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            # Save model locally
            local_dir = Path("data/models")
            local_dir.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump(self.model, local_dir / f"{self.model_type}_model.joblib")
            logger.info(f"Model saved locally at data/models/{self.model_type}_model.joblib")
            
            logger.info(f"Training complete. Val F1: {val_metrics.get('val_f1', 0):.4f}")
            
            return all_metrics
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, prefix: str = "test") -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            prefix: Metric prefix (train/val/test)
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            f"{prefix}_precision": precision_score(y, y_pred),
            f"{prefix}_recall": recall_score(y, y_pred),
            f"{prefix}_f1": f1_score(y, y_pred),
            f"{prefix}_roc_auc": roc_auc_score(y, y_proba),
            f"{prefix}_pr_auc": average_precision_score(y, y_proba),
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics[f"{prefix}_true_negatives"] = tn
        metrics[f"{prefix}_false_positives"] = fp
        metrics[f"{prefix}_false_negatives"] = fn
        metrics[f"{prefix}_true_positives"] = tp
        
        logger.info(f"{prefix.capitalize()} Metrics:")
        logger.info(f"  Precision: {metrics[f'{prefix}_precision']:.4f}")
        logger.info(f"  Recall: {metrics[f'{prefix}_recall']:.4f}")
        logger.info(f"  F1-Score: {metrics[f'{prefix}_f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics[f'{prefix}_roc_auc']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        y_proba = self.model.predict_proba(X)[:, 1]
        return (y_proba >= threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features
            
        Returns:
            Probability of fraud
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)[:, 1]
