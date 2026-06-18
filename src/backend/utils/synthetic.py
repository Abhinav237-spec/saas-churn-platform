import numpy as np
import pandas as pd

def generate_synthetic_data(num_customers: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic customer dataset for SaaS churn prediction.
    Features:
    - customer_id: Unique identifier
    - name: Customer name
    - login_frequency: Average monthly logins (1-30)
    - session_duration: Average session duration in minutes (5-120)
    - feature_usage: Percentage of features utilized (10-100%)
    - support_tickets: Support tickets in the last 30 days (0-10)
    - subscription_age: Months subscribed (1-60)
    - plan_type: Basic, Pro, Enterprise
    - last_active_days: Days since last activity (0-30)
    - revenue: Monthly recurring revenue in USD
    - churn: Binary label (0 or 1) indicating if the customer churned
    """
    np.random.seed(seed)
    
    # Predefined company names list
    companies = [
        "Acme Corp", "Globex", "Initech", "Umbrella Corp", "Hooli", 
        "Vehement Capital", "Soylent Corp", "Reynholm Industries", "Massive Dynamic", 
        "Cyberdyne Systems", "Aperture Science", "Bluth Company", "Stark Industries", 
        "Wayne Enterprises", "Oscorp", "Tyrell Corp", "Dunder Mifflin", 
        "Prestige Worldwide", "Gekko & Co", "Vandelay Industries"
    ]
    
    # Generate features
    customer_ids = [f"CUST-{1000 + i}" for i in range(num_customers)]
    names = [f"{np.random.choice(companies)} #{100 + i}" for i in range(num_customers)]
    
    # Plan types with distribution: 50% Basic, 35% Pro, 15% Enterprise
    plan_types = np.random.choice(
        ["Basic", "Pro", "Enterprise"], 
        size=num_customers, 
        p=[0.50, 0.35, 0.15]
    )
    
    # Sub age: 1 to 60 months
    subscription_age = np.random.randint(1, 61, size=num_customers)
    
    # Last active days: 0 to 30
    last_active_days = np.random.randint(0, 31, size=num_customers)
    
    # Login frequency: related to last_active_days (more inactive means lower frequency)
    login_frequency = np.clip(
        np.random.normal(15 - (last_active_days * 0.3), 5, size=num_customers).astype(int),
        1, 30
    )
    
    # Session duration: minutes, correlates with features used
    session_duration = np.clip(
        np.random.normal(30 + (login_frequency * 2), 15, size=num_customers).astype(int),
        5, 120
    )
    
    # Feature usage: 10 to 100%
    feature_usage = np.clip(
        np.random.normal(40 + (session_duration * 0.5), 15, size=num_customers).astype(int),
        10, 100
    )
    
    # Support tickets: 0 to 10, slightly higher for basic plans and inactive/frustrated customers
    support_tickets = np.clip(
        np.random.poisson(lam=1.5 + (last_active_days * 0.1), size=num_customers),
        0, 10
    )
    
    # Revenue: based on plan type plus some small variance
    revenue = []
    for plan in plan_types:
        if plan == "Basic":
            revenue.append(round(np.random.uniform(29.0, 49.0), 2))
        elif plan == "Pro":
            revenue.append(round(np.random.uniform(99.0, 199.0), 2))
        else: # Enterprise
            revenue.append(round(np.random.uniform(499.0, 999.0), 2))
    revenue = np.array(revenue)
    
    # Calculate churn probability using a log-odds relationship
    # Base log-odds: starting at -1.0 (generally customers are likely to stay)
    log_odds = -1.0
    
    # Normalize features for scaling influence
    log_odds += 0.15 * last_active_days
    log_odds += 0.40 * support_tickets
    log_odds -= 0.08 * login_frequency
    log_odds -= 0.015 * session_duration
    log_odds -= 0.02 * feature_usage
    log_odds -= 0.03 * subscription_age
    
    # Plan type effects
    for i in range(num_customers):
        if plan_types[i] == "Basic":
            log_odds[i] += 0.5
        elif plan_types[i] == "Enterprise":
            log_odds[i] -= 0.6
            
    # Calculate churn probability via sigmoid
    probabilities = 1 / (1 + np.exp(-log_odds))
    
    # Generate binary labels
    churn = (np.random.rand(num_customers) < probabilities).astype(int)
    
    # Construct DataFrame
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "name": names,
        "login_frequency": login_frequency,
        "session_duration": session_duration,
        "feature_usage": feature_usage,
        "support_tickets": support_tickets,
        "subscription_age": subscription_age,
        "plan_type": plan_types,
        "last_active_days": last_active_days,
        "revenue": revenue,
        "churn": churn
    })
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.head())
    print("Churn rate:", df["churn"].mean())
