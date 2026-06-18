import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.backend.engine import ChurnPredictionEngine
from src.backend.llm import generate_retention_playbook, chat_with_customer
from src.backend.utils.synthetic import generate_synthetic_data

app = FastAPI(
    title="SaaS Churn Intelligence Platform API",
    description="Backend API powering XGBoost predictions, SHAP explainability, and LLM-based retention recommendations.",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit runs on port 8501, allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predicting engine
engine = ChurnPredictionEngine(data_dir="data")

# Pydantic schemas
class CustomerData(BaseModel):
    customer_id: str = Field(..., example="CUST-1001")
    name: str = Field("Valued Customer", example="Acme Corp")
    login_frequency: float = Field(..., ge=1, le=30, example=15.5)
    session_duration: float = Field(..., ge=0, example=45.0)
    feature_usage: float = Field(..., ge=0, le=100, example=62.5)
    support_tickets: float = Field(..., ge=0, example=2.0)
    subscription_age: float = Field(..., ge=0, example=12.0)
    plan_type: str = Field(..., example="Pro")  # Basic, Pro, Enterprise
    last_active_days: float = Field(..., ge=0, le=30, example=4.0)
    revenue: float = Field(..., ge=0, example=150.0)

class BatchPredictRequest(BaseModel):
    customers: List[CustomerData]

class RecommendationRequest(BaseModel):
    customer: CustomerData
    prediction: Dict[str, Any]
    api_key: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    customer: CustomerData
    prediction: Dict[str, Any]
    messages: List[ChatMessage]
    api_key: Optional[str] = None

@app.get("/api/health")
def health_check():
    """Checks backend server health and model training status."""
    return {
        "status": "healthy",
        "model_loaded": engine.model is not None,
        "metrics_available": len(engine.get_metrics()) > 0
    }

@app.get("/api/model-info")
def model_info():
    """Returns the performance metrics of the currently loaded model."""
    if engine.model is None:
        return {"status": "untrained", "message": "No model loaded. Train the model first."}
    
    metrics = engine.get_metrics()
    return {
        "status": "loaded",
        "metrics": metrics,
        "features": engine.preprocess_df(pd.DataFrame([{"login_frequency": 1, "session_duration": 1, "feature_usage": 1, "support_tickets": 1, "subscription_age": 1, "plan_type": "Pro", "last_active_days": 1, "revenue": 100.0}])).columns.tolist()
    }

@app.post("/api/predict")
def predict_churn(request: BatchPredictRequest):
    """
    Predicts churn risk and calculates SHAP value contribution details for a batch of customers.
    """
    if engine.model is None:
        # Auto-train default model if none exists, to keep app functional out-of-the-box
        print("Model not loaded. Automatically generating synthetic data and training...")
        try:
            df_synth = generate_synthetic_data(num_customers=800, seed=42)
            engine.train(df_synth)
            df_synth.to_csv(os.path.join(engine.data_dir, "synthetic_customers.csv"), index=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded and auto-training failed: {str(e)}")

    # Convert list of Pydantic models to pandas DataFrame
    data_dict = [c.model_dump() for c in request.customers]
    df = pd.DataFrame(data_dict)
    
    try:
        predictions = engine.predict(df)
        
        # Merge predictions back with customer metadata
        results = []
        for i in range(len(data_dict)):
            results.append({
                "customer": data_dict[i],
                "prediction": predictions[i]
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/train")
async def train_model(file: UploadFile = File(...)):
    """
    Uploads a CSV of historical customer data (containing 'churn' label) 
    and trains a new XGBoost model on it.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        required_cols = [
            'login_frequency', 'session_duration', 'feature_usage', 
            'support_tickets', 'subscription_age', 'plan_type', 
            'last_active_days', 'revenue', 'churn'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}"
            )
            
        # Run training
        metrics = engine.train(df)
        
        # Calculate and return global importance
        importance = engine.get_global_importance(df)
        
        return {
            "message": "Model trained successfully",
            "metrics": metrics,
            "feature_importance": importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/api/recommend")
def recommend_retention(request: RecommendationRequest):
    """
    Generates a markdown Customer Success Playbook based on customer metrics and SHAP scores,
    calling the NVIDIA NIM API or local fallback.
    """
    try:
        playbook = generate_retention_playbook(
            customer=request.customer.model_dump(),
            prediction=request.prediction,
            api_key=request.api_key
        )
        return {"playbook": playbook}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/api/generate-synthetic")
def trigger_synthetic_generation(count: int = Query(1000, ge=100, le=5000)):
    """
    Generates a new synthetic customer dataset, trains the model on it, 
    and saves the dataset and trained model to the file system.
    """
    try:
        df = generate_synthetic_data(num_customers=count, seed=42)
        
        # Save dataset
        df.to_csv(os.path.join(engine.data_dir, "synthetic_customers.csv"), index=False)
        
        # Train model
        metrics = engine.train(df)
        importance = engine.get_global_importance(df)
        
        return {
            "message": f"Generated {count} synthetic customers and trained model.",
            "metrics": metrics,
            "feature_importance": importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate synthetic data: {str(e)}")

@app.get("/api/global-importance")
def global_importance():
    """
    Returns global feature importance by reading the default synthetic customer database.
    """
    if engine.model is None:
        return {"status": "untrained", "importance": {}}
        
    csv_path = os.path.join(engine.data_dir, "synthetic_customers.csv")
    
    # Try reading from current engine data dir, fallback to packaged data folder if not found
    if not os.path.exists(csv_path) and os.path.exists("data/synthetic_customers.csv"):
        csv_path = "data/synthetic_customers.csv"
        
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback to random data for mapping features if no file exists
        df = generate_synthetic_data(num_customers=100)
        
    importance = engine.get_global_importance(df)
    return {"importance": importance}

@app.post("/api/chat")
def chat_retention(request: ChatRequest):
    """
    Handles interactive chat dialogue about a customer's churn factors
    using OpenAI's ChatGPT.
    """
    try:
        response = chat_with_customer(
            customer=request.customer.model_dump(),
            prediction=request.prediction,
            messages=[msg.model_dump() for msg in request.messages],
            api_key=request.api_key
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

# Mount static files at root for local development and routing fallbacks
if os.path.exists("src/frontend"):
    app.mount("/", StaticFiles(directory="src/frontend", html=True), name="static")
