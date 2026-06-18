import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="ChurnIntel | SaaS Churn Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_URL = "http://localhost:8000"

# Custom style loader
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load CSS
load_css("src/frontend/styles.css")

# Helper: check backend health
def check_backend():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Helper: format risk badge
def get_risk_badge(category):
    if category == "Low":
        return '<span class="badge badge-low">Low Risk</span>'
    elif category == "Medium":
        return '<span class="badge badge-medium">Medium Risk</span>'
    else:
        return '<span class="badge badge-high">High Risk</span>'

# Header Branding
st.markdown("""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px;">
    <div style="background: linear-gradient(135deg, #6366F1, #8B5CF6); padding: 12px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);">
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-activity"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
    </div>
    <div>
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(to right, #F8FAFC, #C8CBD9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">ChurnIntel</h1>
        <p style="margin: 0; font-size: 0.95rem; color: #94A3B8;">SaaS Churn Prediction & Explainable AI Advisor</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar setup
st.sidebar.markdown("""
<h2 style="margin-top: 0; font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">Configuration</h2>
""", unsafe_allow_html=True)

# Check backend status
backend_info = check_backend()
if backend_info:
    st.sidebar.markdown(
        '<div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 10px 15px; margin-bottom: 20px;"><span style="color: #34D399; font-weight: 600;">● API Connected</span></div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 10px 15px; margin-bottom: 20px;"><span style="color: #F87171; font-weight: 600;">● API Offline</span><br><small style="color: #94A3B8;">FastAPI server on port 8000 is not running.</small></div>',
        unsafe_allow_html=True
    )

# NIM API configuration
st.sidebar.markdown("### LLM Retention Advisor")
nvidia_key = st.sidebar.text_input("NVIDIA NIM API Key", type="password", help="Enter your NVIDIA API key for Llama 3 retention recommendations. If empty, local rules fallback is used.")

st.sidebar.markdown("### Demo Tools")
if st.sidebar.button("⚙️ Generate & Train Synthetic Data", help="Creates 1,000 realistic customer rows and trains the XGBoost classifier.", use_container_width=True):
    if backend_info:
        with st.spinner("Generating customer profiles..."):
            try:
                r = requests.post(f"{API_URL}/generate-synthetic?count=1000")
                if r.status_code == 200:
                    st.sidebar.success("Synthetic data ready!")
                    st.rerun()
                else:
                    st.sidebar.error(f"Error: {r.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.sidebar.error(f"Error connecting to backend: {e}")
    else:
        st.sidebar.error("Start the FastAPI backend first!")

# Check if synthetic file exists in data directory to populate initial state
data_path = "data/synthetic_customers.csv"
model_trained = backend_info.get("model_loaded", False) if backend_info else False

# Main Navigation Tabs
tab_overview, tab_inspector, tab_predict, tab_train = st.tabs([
    "📊 SaaS Health Dashboard", 
    "🔍 Customer Inspector", 
    "📤 Batch Predict & Export",
    "🏋️ Model Trainer"
])

# Loading local synthetic customer DB helper
@st.cache_data(ttl=60)
def load_synthetic_db():
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df_customers = load_synthetic_db()

# ==================== TAB 1: SAAS HEALTH DASHBOARD ====================
with tab_overview:
    if df_customers is None:
        st.markdown("""
        <div class="metric-card" style="text-align: center; margin: 40px auto; max-width: 600px;">
            <h3 style="color: #F8FAFC;">Welcome to ChurnIntel!</h3>
            <p style="color: #94A3B8; margin-bottom: 25px;">To get started with predictions and metrics, generate a synthetic customer database or train a model in the Trainer tab.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Center button to trigger synthetic data
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Load Demo Customer Data", use_container_width=True):
                if backend_info:
                    with st.spinner("Initializing database..."):
                        requests.post(f"{API_URL}/generate-synthetic?count=1000")
                        st.rerun()
                else:
                    st.error("Please connect the FastAPI backend server first!")
    else:
        if not model_trained:
            st.warning("⚠️ Model is not trained. Active predictions cannot be shown. Please trigger 'Generate & Train Synthetic Data' in the sidebar.")
            st.dataframe(df_customers.head(10))
        else:
            # Prepare predictions for all customers (cached / on the fly)
            @st.cache_data(show_spinner=False)
            def get_all_predictions(df):
                customers_list = df.drop(columns=["churn"], errors="ignore").to_dict(orient="records")
                # Format to schema
                for c in customers_list:
                    # Parse numerical values
                    c["login_frequency"] = float(c["login_frequency"])
                    c["session_duration"] = float(c["session_duration"])
                    c["feature_usage"] = float(c["feature_usage"])
                    c["support_tickets"] = float(c["support_tickets"])
                    c["subscription_age"] = float(c["subscription_age"])
                    c["last_active_days"] = float(c["last_active_days"])
                    c["revenue"] = float(c["revenue"])
                
                try:
                    r = requests.post(f"{API_URL}/predict", json={"customers": customers_list})
                    if r.status_code == 200:
                        results = r.json()["results"]
                        probs = [res["prediction"]["churn_probability"] for res in results]
                        categories = [res["prediction"]["risk_category"] for res in results]
                        factors = [", ".join(res["prediction"]["top_churn_factors"]) for res in results]
                        return probs, categories, factors
                except Exception as e:
                    st.error(f"Connection failure: {e}")
                return None, None, None

            probs, categories, factors = get_all_predictions(df_customers)
            
            if probs is not None:
                df_pred = df_customers.copy()
                df_pred["churn_probability"] = probs
                df_pred["risk_category"] = categories
                df_pred["top_churn_factors"] = factors
                
                # Compute KPIs
                total_customers = len(df_pred)
                avg_risk = df_pred["churn_probability"].mean() * 100
                total_mrr = df_pred["revenue"].sum()
                
                # Revenue at Risk
                rev_at_risk = (df_pred["revenue"] * df_pred["churn_probability"]).sum()
                high_risk_count = len(df_pred[df_pred["risk_category"] == "High"])
                
                # Layout KPIs
                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                
                with kpi_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Active Customers</div>
                        <div class="metric-value">{total_customers:,}</div>
                        <div class="metric-delta delta-positive">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                            +3.2% vs last month
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Average Churn Risk</div>
                        <div class="metric-value">{avg_risk:.1f}%</div>
                        <div class="metric-delta delta-negative">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>
                            +1.5% vs last month
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Monthly Recurring Revenue</div>
                        <div class="metric-value">${total_mrr:,.2f}</div>
                        <div class="metric-delta delta-positive">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                            +$4,820.00 new subscriptions
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi_col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Expected MRR at Risk</div>
                        <div class="metric-value">${rev_at_risk:,.2f}</div>
                        <div class="metric-delta delta-negative">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>
                            {high_risk_count} customers in High Risk (>70%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visual Analytics Charts (Using Plotly)
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.markdown('<h3 style="font-size: 1.15rem; color: #F8FAFC;">Churn Risk Distribution</h3>', unsafe_allow_html=True)
                    risk_counts = df_pred["risk_category"].value_counts().reset_index()
                    risk_counts.columns = ["Risk Category", "Count"]
                    fig_risk = px.bar(
                        risk_counts, 
                        x="Risk Category", 
                        y="Count", 
                        color="Risk Category",
                        color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
                        template="plotly_dark"
                    )
                    fig_risk.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="",
                        yaxis_title="Number of Customers",
                        showlegend=False,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                with chart_col2:
                    st.markdown('<h3 style="font-size: 1.15rem; color: #F8FAFC;">Revenue Exposure by Plan Tier & Risk</h3>', unsafe_allow_html=True)
                    fig_rev = px.histogram(
                        df_pred,
                        x="plan_type",
                        y="revenue",
                        color="risk_category",
                        barmode="group",
                        color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
                        category_orders={"plan_type": ["Basic", "Pro", "Enterprise"]},
                        labels={"plan_type": "Plan Tier", "revenue": "Revenue ($)", "risk_category": "Risk Category"},
                        template="plotly_dark"
                    )
                    fig_rev.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="",
                        yaxis_title="Total Monthly Revenue ($)",
                        legend_title="Risk Level",
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_rev, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Customer Segment Plotly Bubble Chart
                st.markdown('<h3 style="font-size: 1.15rem; color: #F8FAFC;">Customer Health Mapping (Adoption vs Inactivity)</h3>', unsafe_allow_html=True)
                fig_bubble = px.scatter(
                    df_pred,
                    x="feature_usage",
                    y="last_active_days",
                    size="revenue",
                    color="churn_probability",
                    color_continuous_scale=px.colors.sequential.Bluered,
                    labels={
                        "feature_usage": "Feature Adoption (%)",
                        "last_active_days": "Days Since Last Active",
                        "churn_probability": "Churn Risk",
                        "revenue": "Revenue ($)"
                    },
                    hover_name="name",
                    template="plotly_dark"
                )
                fig_bubble.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Highest Risk Customers Table
                st.markdown('<h3 style="font-size: 1.15rem; color: #F8FAFC;">Highest-Risk Customers (Immediate CS Outreach)</h3>', unsafe_allow_html=True)
                high_risk_df = df_pred[df_pred["risk_category"] == "High"].sort_values(by="revenue", ascending=False).head(15)
                
                # Format tabular columns for display
                display_cols = ["customer_id", "name", "plan_type", "revenue", "last_active_days", "support_tickets", "churn_probability", "top_churn_factors"]
                table_df = high_risk_df[display_cols].copy()
                table_df["churn_probability"] = table_df["churn_probability"].apply(lambda x: f"{x*100:.1f}%")
                table_df["revenue"] = table_df["revenue"].apply(lambda x: f"${x:.2f}")
                table_df.columns = ["ID", "Name", "Plan Tier", "Monthly Revenue", "Days Inactive", "Tickets", "Churn Prob", "Top Churn Factors"]
                
                st.dataframe(table_df, use_container_width=True, hide_index=True)
                
            else:
                st.error("Failed to fetch churn predictions from the FastAPI API endpoint.")

# ==================== TAB 2: CUSTOMER INSPECTOR ====================
with tab_inspector:
    if df_customers is None or not model_trained:
        st.info("Please set up or train customer profiles in the Health Dashboard tab first.")
    else:
        # Search Box / Selector
        st.markdown('<h3 style="font-size: 1.15rem; color: #F8FAFC;">Select Customer to Profile</h3>', unsafe_allow_html=True)
        customer_options = [f"{row['customer_id']} - {row['name']}" for idx, row in df_customers.iterrows()]
        selected_option = st.selectbox("Search Customer Directory:", customer_options)
        
        selected_id = selected_option.split(" - ")[0]
        customer_row = df_customers[df_customers["customer_id"] == selected_id].iloc[0]
        
        # Prepare API payload for single customer
        customer_dict = customer_row.to_dict()
        customer_dict["login_frequency"] = float(customer_dict["login_frequency"])
        customer_dict["session_duration"] = float(customer_dict["session_duration"])
        customer_dict["feature_usage"] = float(customer_dict["feature_usage"])
        customer_dict["support_tickets"] = float(customer_dict["support_tickets"])
        customer_dict["subscription_age"] = float(customer_dict["subscription_age"])
        customer_dict["last_active_days"] = float(customer_dict["last_active_days"])
        customer_dict["revenue"] = float(customer_dict["revenue"])
        
        # Query API for single prediction
        try:
            r = requests.post(f"{API_URL}/predict", json={"customers": [customer_dict]})
            if r.status_code == 200:
                prediction_data = r.json()["results"][0]["prediction"]
                
                col_c1, col_c2 = st.columns([1, 2])
                
                with col_c1:
                    # Core Details Cards
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
                            <div>
                                <h3 style="margin: 0; font-size: 1.3rem; color: #F8FAFC;">{customer_dict['name']}</h3>
                                <code style="font-size: 0.8rem; color: #6366F1;">{customer_dict['customer_id']}</code>
                            </div>
                            {get_risk_badge(prediction_data['risk_category'])}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 15px;">
                            <div>
                                <small style="color: #94A3B8; display: block;">Plan Tier</small>
                                <strong style="color: #F8FAFC;">{customer_dict['plan_type']}</strong>
                            </div>
                            <div>
                                <small style="color: #94A3B8; display: block;">MRR Contribution</small>
                                <strong style="color: #F8FAFC;">${customer_dict['revenue']:.2f}</strong>
                            </div>
                            <div>
                                <small style="color: #94A3B8; display: block;">Relationship Age</small>
                                <strong style="color: #F8FAFC;">{int(customer_dict['subscription_age'])} months</strong>
                            </div>
                            <div>
                                <small style="color: #94A3B8; display: block;">Days Inactive</small>
                                <strong style="color: #F8FAFC;">{int(customer_dict['last_active_days'])} days</strong>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk Gauge Chart
                    prob_val = prediction_data['churn_percentage']
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob_val,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Probability (%)", 'font': {'size': 14, 'color': '#94A3B8'}},
                        number = {'font': {'color': '#F8FAFC', 'size': 32}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                            'bar': {'color': "#6366F1"},
                            'bgcolor': "rgba(30,41,59,0.3)",
                            'borderwidth': 1,
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#F8FAFC", 'family': "Inter"},
                        height=200,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with col_c2:
                    # Local SHAP Explainable AI chart
                    st.markdown('<h3 style="font-size: 1.15rem; color: #F8FAFC; margin-bottom: 15px;">Local Feature Contributions (Explainable AI SHAP)</h3>', unsafe_allow_html=True)
                    
                    shap_data = prediction_data["shap_contributions"]
                    # Map feature names to human-readable names
                    feature_mapping = {
                        "login_frequency": "Login Frequency",
                        "session_duration": "Session Duration",
                        "feature_usage": "Feature Adoption %",
                        "support_tickets": "Support Tickets",
                        "subscription_age": "Subscription Age",
                        "last_active_days": "Days Inactive",
                        "revenue": "MRR Value",
                        "plan_type_Basic": "Basic Plan Tier",
                        "plan_type_Pro": "Pro Plan Tier",
                        "plan_type_Enterprise": "Enterprise Plan Tier"
                    }
                    
                    # Sort SHAP contributions
                    shap_sorted = sorted(shap_data.items(), key=lambda item: abs(item[1]), reverse=True)
                    
                    features_list = [feature_mapping.get(k, k) for k, v in shap_sorted]
                    values_list = [v for k, v in shap_sorted]
                    colors_list = ["#EF4444" if v > 0 else "#10B981" for v in values_list] # Red for risk increase, Green for decrease
                    
                    fig_shap = go.Figure()
                    fig_shap.add_trace(go.Bar(
                        y=features_list,
                        x=values_list,
                        orientation='h',
                        marker_color=colors_list,
                        text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values_list],
                        textposition='auto',
                        hoverinfo='x'
                    ))
                    fig_shap.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                            title="SHAP Value (Impact on Log-Odds of Churn)",
                            gridcolor='rgba(255,255,255,0.06)',
                            zerolinecolor='rgba(255,255,255,0.2)'
                        ),
                        yaxis=dict(autorange="reversed"),
                        height=350,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    # Text insights based on drivers
                    pos_drivers = prediction_data.get("top_churn_factors", [])
                    neg_drivers = prediction_data.get("top_retention_factors", [])
                    
                    factor_bullets = []
                    for fd in pos_drivers:
                        factor_bullets.append(f"🔴 **{feature_mapping.get(fd, fd)}** is increasing churn risk.")
                    for rd in neg_drivers:
                        factor_bullets.append(f"🟢 **{feature_mapping.get(rd, rd)}** is buffering churn risk (retention anchor).")
                        
                    st.markdown("<br>".join(factor_bullets), unsafe_allow_html=True)
                
                # --- LLM Advisor Playbook Generator ---
                st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin-top: 30px;'>", unsafe_allow_html=True)
                st.markdown('<h3 style="font-size: 1.25rem; color: #F8FAFC;">⚡ AI Retention Advisor</h3>', unsafe_allow_html=True)
                
                playbook_state_key = f"playbook_{customer_dict['customer_id']}"
                
                if playbook_state_key not in st.session_state:
                    st.session_state[playbook_state_key] = None
                    
                if st.button("🔮 Generate Tailored Retention Playbook", type="primary", use_container_width=True):
                    with st.spinner("Generating Customer Success playbook..."):
                        try:
                            req_payload = {
                                "customer": customer_dict,
                                "prediction": prediction_data,
                                "api_key": nvidia_key if nvidia_key else None
                            }
                            rec_r = requests.post(f"{API_URL}/recommend", json=req_payload)
                            if rec_r.status_code == 200:
                                st.session_state[playbook_state_key] = rec_r.json()["playbook"]
                            else:
                                st.error(f"Failed to generate playbook: {rec_r.text}")
                        except Exception as e:
                            st.error(f"Error connecting to Advisor API: {e}")
                
                if st.session_state[playbook_state_key]:
                    st.markdown(f'<div class="playbook-container">', unsafe_allow_html=True)
                    st.markdown(st.session_state[playbook_state_key])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            else:
                st.error("Error retrieving prediction from backend API.")
        except Exception as e:
            st.error(f"Could not connect to FastAPI server to get predictions: {e}")

# ==================== TAB 3: BATCH PREDICT & EXPORT ====================
with tab_predict:
    st.markdown("""
    <h3 style="font-size: 1.15rem; color: #F8FAFC;">Upload Customer Activity File</h3>
    <p style="color: #94A3B8;">Upload a CSV of active customer usage data to perform batch churn predictions and risk category classification.</p>
    """, unsafe_allow_html=True)
    
    # Showcase required schema
    with st.expander("ℹ️ Show Required CSV Column Headers"):
        st.code("""
customer_id,name,login_frequency,session_duration,feature_usage,support_tickets,subscription_age,plan_type,last_active_days,revenue
CUST-9901,Vandelay Inc,14,45,60,3,15,Pro,2,149.00
CUST-9902,Initech,4,12,18,6,2,Basic,14,39.00
        """)
        
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Show preview
            st.markdown("##### File Preview (Top 5 rows)")
            st.dataframe(df_upload.head(5))
            
            # Pre-validation
            required_cols = [
                'login_frequency', 'session_duration', 'feature_usage', 
                'support_tickets', 'subscription_age', 'plan_type', 
                'last_active_days', 'revenue'
            ]
            missing = [c for c in required_cols if c not in df_upload.columns]
            
            if missing:
                st.error(f"The uploaded file is missing the following required columns: {', '.join(missing)}")
            else:
                if st.button("⚡ Run Batch Churn Prediction", type="primary", use_container_width=True):
                    with st.spinner("Processing records..."):
                        # Format to list of dicts
                        batch_records = df_upload.to_dict(orient="records")
                        # Clean ids and names if missing
                        for i, record in enumerate(batch_records):
                            if "customer_id" not in record or pd.isna(record["customer_id"]):
                                record["customer_id"] = f"BATCH-{1000+i}"
                            if "name" not in record or pd.isna(record["name"]):
                                record["name"] = f"Customer #{i}"
                            record["login_frequency"] = float(record["login_frequency"])
                            record["session_duration"] = float(record["session_duration"])
                            record["feature_usage"] = float(record["feature_usage"])
                            record["support_tickets"] = float(record["support_tickets"])
                            record["subscription_age"] = float(record["subscription_age"])
                            record["last_active_days"] = float(record["last_active_days"])
                            record["revenue"] = float(record["revenue"])
                        
                        try:
                            r = requests.post(f"{API_URL}/predict", json={"customers": batch_records})
                            if r.status_code == 200:
                                results = r.json()["results"]
                                
                                # Re-assemble predictions into DataFrame
                                out_probs = [res["prediction"]["churn_probability"] for res in results]
                                out_pcts = [res["prediction"]["churn_percentage"] for res in results]
                                out_categories = [res["prediction"]["risk_category"] for res in results]
                                out_factors = [", ".join(res["prediction"]["top_churn_factors"]) for res in results]
                                
                                df_result = df_upload.copy()
                                df_result["churn_probability"] = out_probs
                                df_result["churn_percentage"] = out_pcts
                                df_result["risk_category"] = out_categories
                                df_result["top_churn_factors"] = out_factors
                                
                                st.success(f"Batch prediction completed! Processed {len(df_result)} customer accounts.")
                                
                                # Layout stats summary
                                col_b1, col_b2, col_b3 = st.columns(3)
                                col_b1.metric("Batch Churn Rate (Avg)", f"{df_result['churn_probability'].mean() * 100:.1f}%")
                                col_b2.metric("High-Risk Accounts", len(df_result[df_result["risk_category"] == "High"]))
                                col_b3.metric("Revenue at Risk", f"${(df_result['revenue'] * df_result['churn_probability']).sum():,.2f}")
                                
                                st.markdown("##### Predictions Output")
                                st.dataframe(df_result, use_container_width=True)
                                
                                # Download Button
                                csv_buffer = df_result.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Churn Predictions CSV",
                                    data=csv_buffer,
                                    file_name="churn_predictions_export.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.error(f"API returned an error: {r.text}")
                        except Exception as e:
                            st.error(f"Error contacting FastAPI server: {e}")
        except Exception as e:
            st.error(f"Failed to read CSV file: {e}")

# ==================== TAB 4: MODEL TRAINER ====================
with tab_train:
    st.markdown("""
    <h3 style="font-size: 1.15rem; color: #F8FAFC;">Upload Historical Labeled Data to Re-Train</h3>
    <p style="color: #94A3B8;">To customize predictions for your exact business profile, upload a historical CSV dataset containing a <code>churn</code> column (with 1 representing churned, 0 representing retained).</p>
    """, unsafe_allow_html=True)
    
    # Showcase required schema
    with st.expander("ℹ️ Show Required Training CSV Format"):
        st.code("""
login_frequency,session_duration,feature_usage,support_tickets,subscription_age,plan_type,last_active_days,revenue,churn
24,65,72,1,22,Enterprise,1,750.00,0
2,5,10,8,4,Basic,24,29.00,1
        """)
        
    train_file = st.file_uploader("Choose training CSV", type=["csv"], key="train_uploader")
    
    if train_file is not None:
        try:
            df_train = pd.read_csv(train_file)
            st.success("Training file uploaded successfully!")
            st.markdown("##### Uploaded Training Data Preview")
            st.dataframe(df_train.head(5))
            
            # Pre-validation
            required_cols = [
                'login_frequency', 'session_duration', 'feature_usage', 
                'support_tickets', 'subscription_age', 'plan_type', 
                'last_active_days', 'revenue', 'churn'
            ]
            missing = [c for c in required_cols if c not in df_train.columns]
            
            if missing:
                st.error(f"The training file is missing required columns: {', '.join(missing)}")
            else:
                if st.button("🏋️ Train Custom XGBoost Model", type="primary", use_container_width=True):
                    with st.spinner("Training XGBoost Classifier... (Estimating SHAP explainer matrix)"):
                        # POST file to API
                        try:
                            # Send CSV file bytes to FastAPI
                            files = {"file": (train_file.name, train_file.getvalue(), "text/csv")}
                            r = requests.post(f"{API_URL}/train", files=files)
                            
                            if r.status_code == 200:
                                res_data = r.json()
                                st.success("Model trained and serialized successfully!")
                                
                                # Show metrics
                                metrics = res_data["metrics"]
                                col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
                                col_t1.metric("ROC-AUC Score", f"{metrics['auc_roc']:.3f}")
                                col_t2.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                                col_t3.metric("F1-Score", f"{metrics['f1']:.3f}")
                                col_t4.metric("Precision", f"{metrics['precision']*100:.1f}%")
                                col_t5.metric("Recall", f"{metrics['recall']*100:.1f}%")
                                
                                # Visual Importance
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown('<h4 style="font-size: 1.1rem; color: #F8FAFC;">Global SHAP Feature Importances</h4>', unsafe_allow_html=True)
                                
                                global_imp = res_data["feature_importance"]
                                feature_mapping = {
                                    "login_frequency": "Login Frequency",
                                    "session_duration": "Session Duration",
                                    "feature_usage": "Feature Adoption %",
                                    "support_tickets": "Support Tickets",
                                    "subscription_age": "Subscription Age",
                                    "last_active_days": "Days Inactive",
                                    "revenue": "MRR Value",
                                    "plan_type_Basic": "Basic Plan Tier",
                                    "plan_type_Pro": "Pro Plan Tier",
                                    "plan_type_Enterprise": "Enterprise Plan Tier"
                                }
                                
                                imp_df = pd.DataFrame([
                                    {"Feature": feature_mapping.get(k, k), "SHAP Importance": v} 
                                    for k, v in global_imp.items()
                                ])
                                
                                fig_imp = px.bar(
                                    imp_df,
                                    x="SHAP Importance",
                                    y="Feature",
                                    orientation='h',
                                    color="SHAP Importance",
                                    color_continuous_scale=px.colors.sequential.Indigo,
                                    template="plotly_dark"
                                )
                                fig_imp.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    yaxis=dict(autorange="reversed"),
                                    coloraxis_showscale=False,
                                    margin=dict(l=10, r=10, t=10, b=10)
                                )
                                st.plotly_chart(fig_imp, use_container_width=True)
                                
                                # Save training metrics
                                st.info("This model is now loaded as the active predictor across the dashboard.")
                                
                            else:
                                st.error(f"Training endpoint failed: {r.text}")
                        except Exception as e:
                            st.error(f"Error calling training endpoint: {e}")
        except Exception as e:
            st.error(f"Failed to read CSV file: {e}")
            
    else:
        # Show current model info if available
        if backend_info and backend_info.get("metrics_available", False):
            try:
                mi_r = requests.get(f"{API_URL}/model-info")
                if mi_r.status_code == 200:
                    mi = mi_r.json()
                    if mi["status"] == "loaded":
                        metrics = mi["metrics"]
                        st.markdown('<h4 style="font-size: 1.1rem; color: #F8FAFC;">Active Model Performance metrics</h4>', unsafe_allow_html=True)
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        col_m1.metric("ROC-AUC Score", f"{metrics['auc_roc']:.3f}")
                        col_m2.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                        col_m3.metric("F1-Score", f"{metrics['f1']:.3f}")
                        col_m4.metric("Dataset Size", f"{metrics['total_customers']:,} accounts")
                        
                        # Show active global importance
                        imp_r = requests.get(f"{API_URL}/global-importance")
                        if imp_r.status_code == 200 and imp_r.json().get("importance"):
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown('<h4 style="font-size: 1.1rem; color: #F8FAFC;">Active Global Feature Importance (SHAP)</h4>', unsafe_allow_html=True)
                            global_imp = imp_r.json()["importance"]
                            feature_mapping = {
                                "login_frequency": "Login Frequency",
                                "session_duration": "Session Duration",
                                "feature_usage": "Feature Adoption %",
                                "support_tickets": "Support Tickets",
                                "subscription_age": "Subscription Age",
                                "last_active_days": "Days Inactive",
                                "revenue": "MRR Value",
                                "plan_type_Basic": "Basic Plan Tier",
                                "plan_type_Pro": "Pro Plan Tier",
                                "plan_type_Enterprise": "Enterprise Plan Tier"
                            }
                            
                            imp_df = pd.DataFrame([
                                {"Feature": feature_mapping.get(k, k), "SHAP Importance": v} 
                                for k, v in global_imp.items()
                            ])
                            
                            fig_imp = px.bar(
                                imp_df,
                                x="SHAP Importance",
                                y="Feature",
                                orientation='h',
                                color="SHAP Importance",
                                color_continuous_scale=px.colors.sequential.Indigo,
                                template="plotly_dark"
                            )
                            fig_imp.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(autorange="reversed"),
                                coloraxis_showscale=False,
                                margin=dict(l=10, r=10, t=10, b=10)
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.write(f"Could not load active model info: {e}")
        else:
            st.info("No active model is trained yet. Select 'Generate & Train Synthetic Data' in the sidebar or upload a labeled CSV.")
