# ChurnIntel - SaaS Churn Intelligence Platform

ChurnIntel is a modern SaaS Churn Intelligence Platform designed to allow SaaS founders, indie hackers, and product managers to predict and manage customer churn. The platform utilizes advanced machine learning (XGBoost), Explainable AI (SHAP), and an interactive ChatGPT-powered retention advisor to provide actionable insights for keeping your MRR safe.

## Key Features

- **SaaS Health Dashboard**: An elegant, glassmorphic UI displaying global churn risks, MRR exposure, and a directory of your most at-risk customers.
- **Machine Learning Engine**: Powered by XGBoost, the engine trains on your historical data and accurately predicts customer churn probabilities.
- **Explainable AI**: Integrated SHAP values explicitly show *why* a customer is at risk—whether it's declining feature usage, low login frequency, or too many support tickets.
- **ChatGPT Retention Advisor**: Directly connect your ChatGPT account using your API key and interactively discuss tailored customer success playbooks for saving high-risk accounts.
- **Batch Processing & Retraining**: Upload CSV data files to generate batch predictions, or upload historical labels to retrain the local XGBoost model on your unique metrics.

## Quick Start (Local Development)

The platform has a backend written in FastAPI (Python) and a statically served HTML/JS frontend.

1. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `.\venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   uvicorn src.backend.main:app --reload
   ```

3. **Access the Dashboard**:
   Open [http://localhost:8000](http://localhost:8000) in your browser.

## Deployment
The static frontend is configured for deployment via services like Vercel (using the `vercel.json` file). The Python backend can be deployed onto platforms like Render, Railway, or Heroku.
