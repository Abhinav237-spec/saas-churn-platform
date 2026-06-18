import os
import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_generate_synthetic():
    response = client.post("/api/generate-synthetic?count=100")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "metrics" in data
    assert "feature_importance" in data
    
    # Model should be loaded now
    health_r = client.get("/api/health")
    assert health_r.json()["model_loaded"] is True

def test_model_info():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "loaded"
    assert "metrics" in data
    assert "features" in data

def test_predict_endpoint():
    # Construct a valid customer payload
    payload = {
        "customers": [
            {
                "customer_id": "CUST-TEST",
                "name": "Test Company",
                "login_frequency": 12.0,
                "session_duration": 35.0,
                "feature_usage": 45.0,
                "support_tickets": 1.0,
                "subscription_age": 6.0,
                "plan_type": "Pro",
                "last_active_days": 3.0,
                "revenue": 129.0
            }
        ]
    }
    
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    
    pred_res = data["results"][0]
    assert pred_res["customer"]["customer_id"] == "CUST-TEST"
    assert "prediction" in pred_res
    assert "churn_probability" in pred_res["prediction"]
    assert "risk_category" in pred_res["prediction"]

def test_recommend_endpoint():
    payload = {
        "customer": {
            "customer_id": "CUST-TEST",
            "name": "Test Company",
            "login_frequency": 12.0,
            "session_duration": 35.0,
            "feature_usage": 45.0,
            "support_tickets": 1.0,
            "subscription_age": 6.0,
            "plan_type": "Pro",
            "last_active_days": 3.0,
            "revenue": 129.0
        },
        "prediction": {
            "churn_probability": 0.45,
            "churn_percentage": 45.0,
            "risk_category": "Medium",
            "top_churn_factors": ["last_active_days"],
            "top_retention_factors": ["login_frequency"]
        },
        "api_key": None
    }
    
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "playbook" in data
    assert "Executive Summary & Risk Assessment" in data["playbook"]

def test_train_endpoint():
    # Generate mock training csv
    df = pd.DataFrame({
        "login_frequency": [10, 20, 5, 25, 2],
        "session_duration": [30, 45, 10, 80, 5],
        "feature_usage": [40, 60, 20, 90, 10],
        "support_tickets": [1, 0, 4, 1, 8],
        "subscription_age": [12, 24, 3, 36, 1],
        "plan_type": ["Basic", "Pro", "Basic", "Enterprise", "Basic"],
        "last_active_days": [2, 1, 10, 0, 20],
        "revenue": [49.0, 149.0, 29.0, 799.0, 29.0],
        "churn": [0, 0, 1, 0, 1]
    })
    
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    
    files = {"file": ("test_train.csv", csv_bytes, "text/csv")}
    
    response = client.post("/api/train", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "metrics" in data
    assert "feature_importance" in data

def test_chat_endpoint():
    payload = {
        "customer": {
            "customer_id": "CUST-TEST",
            "name": "Test Company",
            "login_frequency": 12.0,
            "session_duration": 35.0,
            "feature_usage": 45.0,
            "support_tickets": 1.0,
            "subscription_age": 6.0,
            "plan_type": "Pro",
            "last_active_days": 3.0,
            "revenue": 129.0
        },
        "prediction": {
            "churn_probability": 0.45,
            "churn_percentage": 45.0,
            "risk_category": "Medium",
            "top_churn_factors": ["last_active_days"],
            "top_retention_factors": ["login_frequency"]
        },
        "messages": [
            {
                "role": "user",
                "content": "Why is this customer at risk?"
            }
        ],
        "api_key": None
    }
    
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "CUST-TEST" in data["response"] or "risk" in data["response"].lower() or "Test Company" in data["response"]

