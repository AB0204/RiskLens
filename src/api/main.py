"""FastAPI application for FraudGuard fraud detection."""

import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
from pathlib import Path

from .schemas import TransactionRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from ..utils import settings, logger
from ..explainability import ModelExplainer
from ..monitoring.metrics import (
    track_prediction_metrics,
    get_metrics,
    update_model_status,
    predictions_total,
    CONTENT_TYPE_LATEST
)

# Global model instance
model = None
feature_engineer = None
explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, feature_engineer, explainer

    model_path = Path("data/models/xgboost_model.joblib")
    fe_path = Path("data/models/feature_engineer.joblib")

    try:
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        if fe_path.exists():
            feature_engineer = joblib.load(fe_path)
            logger.info(f"Feature engineer loaded from {fe_path}")

        if model is not None:
            feature_names = list(range(30))
            explainer = ModelExplainer(model, feature_names)
            logger.info("SHAP explainer initialized")
            update_model_status(True)
        else:
            logger.warning("No model file found — API will return 503 on /predict")
            update_model_status(False)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        update_model_status(False)

    yield


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Real-time fraud detection API with ML-powered risk assessment",
    lifespan=lifespan,
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to FraudGuard API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_type="xgboost" if model else None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
@track_prediction_metrics
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict if a transaction is fraudulent.
    
    Returns fraud probability and risk classification.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Apply feature engineering if available
        if feature_engineer:
            df = feature_engineer.transform(df)
        
        # Get prediction
        fraud_prob = model.predict_proba(df)[0, 1]
        is_fraud = fraud_prob >= settings.threshold
        
        # Determine risk score
        if fraud_prob < 0.3:
            risk_score = "low"
        elif fraud_prob < 0.7:
            risk_score = "medium"
        else:
            risk_score = "high"
        
        logger.info(f"Prediction: {is_fraud}, Probability: {fraud_prob:.4f}")
        
        return PredictionResponse(
            is_fraud=bool(is_fraud),
            fraud_probability=float(fraud_prob),
            risk_score=risk_score
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(transactions: list[TransactionRequest]):
    """
    Predict fraud for multiple transactions.
    
    Returns list of predictions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([t.dict() for t in transactions])
        
        # Apply feature engineering
        if feature_engineer:
            df = feature_engineer.transform(df)
        
        # Get predictions
        fraud_probs = model.predict_proba(df)[:, 1]

        results = []
        for prob in fraud_probs:
            is_fraud = prob >= settings.threshold
            risk_score = "low" if prob < 0.3 else ("medium" if prob < 0.7 else "high")
            
            results.append({
                "is_fraud": bool(is_fraud),
                "fraud_probability": float(prob),
                "risk_score": risk_score
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", tags=["Predictions"])
async def explain_prediction(transaction: TransactionRequest):
    """
    Explain a fraud prediction using SHAP values.
    
    Returns top contributing features and their impact.
    """
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model or explainer not loaded")
    
    try:
        # Convert to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Apply feature engineering
        if feature_engineer:
            df = feature_engineer.transform(df)
        
        # Get explanation
        explanation = explainer.explain_prediction(df, top_k=10)
        readable_explanation = explainer.explain_local(df)
        
        return {
            "prediction": explanation['prediction_value'],
            "base_value": explanation['base_value'],
            "top_features": explanation['top_features'],
            "explanation_text": readable_explanation
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type="xgboost",
        mlflow_run_id=None,  # Would be populated from MLflow
        features_count=30,  # Base features
        threshold=settings.threshold
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping.
    """
    from starlette.responses import Response
    return Response(content=get_metrics(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
