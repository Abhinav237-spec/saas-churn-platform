import os
import shutil
import pandas as pd
import pytest
from src.backend.engine import ChurnPredictionEngine, FEATURE_COLUMNS
from src.backend.utils.synthetic import generate_synthetic_data

TEMP_DATA_DIR = "temp_test_data"

@pytest.fixture(scope="module")
def setup_engine():
    # Setup temporary directory for testing
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
        
    engine = ChurnPredictionEngine(data_dir=TEMP_DATA_DIR)
    yield engine
    
    # Teardown
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)

def test_synthetic_data_generation():
    df = generate_synthetic_data(num_customers=100, seed=42)
    assert len(df) == 100
    assert "churn" in df.columns
    assert "customer_id" in df.columns
    assert "plan_type" in df.columns
    
    # Check that churn rate is between 5% and 95%
    churn_rate = df["churn"].mean()
    assert 0.05 < churn_rate < 0.95

def test_preprocessing(setup_engine):
    engine = setup_engine
    df = generate_synthetic_data(num_customers=10, seed=42)
    
    processed_df = engine.preprocess_df(df)
    
    # Check column match
    assert list(processed_df.columns) == FEATURE_COLUMNS
    assert len(processed_df) == 10
    
    # Check plan type encoding
    assert processed_df["plan_type_Basic"].sum() + processed_df["plan_type_Pro"].sum() + processed_df["plan_type_Enterprise"].sum() == 10
    assert ((processed_df["plan_type_Basic"] == 0) | (processed_df["plan_type_Basic"] == 1)).all()

def test_training_and_prediction(setup_engine):
    engine = setup_engine
    df = generate_synthetic_data(num_customers=200, seed=42)
    
    # Train
    metrics = engine.train(df)
    assert "accuracy" in metrics
    assert "auc_roc" in metrics
    assert metrics["accuracy"] >= 0.5
    
    # Predict
    predict_df = df.drop(columns=["churn"]).head(5)
    predictions = engine.predict(predict_df)
    
    assert len(predictions) == 5
    for pred in predictions:
        assert 0.0 <= pred["churn_probability"] <= 1.0
        assert pred["risk_category"] in ["Low", "Medium", "High"]
        assert "shap_contributions" in pred
        assert len(pred["shap_contributions"]) == len(FEATURE_COLUMNS)
        
def test_global_importance(setup_engine):
    engine = setup_engine
    df = generate_synthetic_data(num_customers=50, seed=42)
    
    # Ensure model is trained from previous test
    importance = engine.get_global_importance(df)
    assert len(importance) == len(FEATURE_COLUMNS)
    assert all(isinstance(val, float) for val in importance.values())
