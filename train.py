"""Training script for FraudGuard fraud detection."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import load_credit_card_fraud_data, split_data, get_feature_target_split, FraudFeatureEngineer
from src.models import FraudDetector
from src.utils import logger, settings


def main():
    """Main training function."""
    logger.info("Starting FraudGuard training pipeline...")
    
    # Load data
    try:
        df = load_credit_card_fraud_data()
    except FileNotFoundError:
        logger.error("Dataset not found!")
        logger.info("Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        logger.info(f"Save to: {settings.data_path}/raw/creditcard.csv")
        return
    
    # Split data
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1)
    
    # Get features and target
    X_train, y_train = get_feature_target_split(train_df)
    X_val, y_val = get_feature_target_split(val_df)
    X_test, y_test = get_feature_target_split(test_df)
    
    # Feature engineering
    logger.info("Engineering features...")
    feature_engineer = FraudFeatureEngineer()
    X_train_transformed = feature_engineer.fit_transform(X_train)
    X_val_transformed = feature_engineer.transform(X_val)
    X_test_transformed = feature_engineer.transform(X_test)
    
    import joblib
    local_dir = Path("data/models")
    local_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_engineer, local_dir / "feature_engineer.joblib")
    logger.info("Saved fitted feature engineer to data/models/feature_engineer.joblib")
    
    logger.info(f"Features after engineering: {X_train_transformed.shape[1]}")
    
    # Train XGBoost model
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost Model")
    logger.info("="*50)
    
    xgb_detector = FraudDetector(model_type="xgboost", use_smote=True)
    xgb_metrics = xgb_detector.train(
        X_train_transformed,
        y_train,
        X_val_transformed,
        y_val,
        experiment_name="FraudGuard-Fraud-Detection"
    )
    
    # Final evaluation on test set
    logger.info("\n" + "="*50)
    logger.info("Final Test Set Evaluation")
    logger.info("="*50)
    test_metrics = xgb_detector.evaluate(X_test_transformed, y_test, prefix="test")
    
    logger.info("\n✅ Training complete!")
    logger.info(f"Test F1-Score: {test_metrics['test_f1']:.4f}")
    logger.info(f"Test Precision: {test_metrics['test_precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['test_recall']:.4f}")
    logger.info(f"Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")
    
    # Train LightGBM model for comparison
    logger.info("\n" + "="*50)
    logger.info("Training LightGBM Model (for comparison)")
    logger.info("="*50)
    
    lgb_detector = FraudDetector(model_type="lightgbm", use_smote=True)
    lgb_metrics = lgb_detector.train(
        X_train_transformed,
        y_train,
        X_val_transformed,
        y_val,
        experiment_name="FraudGuard-Fraud-Detection"
    )
    
    lgb_test_metrics = lgb_detector.evaluate(X_test_transformed, y_test, prefix="test")
    
    logger.info("\n📊 Model Comparison:")
    logger.info(f"XGBoost F1: {test_metrics['test_f1']:.4f} | LightGBM F1: {lgb_test_metrics['test_f1']:.4f}")
    logger.info(f"XGBoost Precision: {test_metrics['test_precision']:.4f} | LightGBM Precision: {lgb_test_metrics['test_precision']:.4f}")
    logger.info(f"XGBoost Recall: {test_metrics['test_recall']:.4f} | LightGBM Recall: {lgb_test_metrics['test_recall']:.4f}")
    
    logger.info("\n🎉 All training complete! Check MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    main()
