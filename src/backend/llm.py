import requests
import json
from typing import Dict, Any, List

def generate_retention_playbook(
    customer: Dict[str, Any], 
    prediction: Dict[str, Any], 
    api_key: str = None
) -> str:
    """
    Generates a tailored customer success retention playbook using OpenAI's ChatGPT.
    If no API key is provided, falls back to a high-quality, rules-based playbook generator.
    """
    # Extract data for clarity
    name = customer.get("name", "Valued Customer")
    customer_id = customer.get("customer_id", "N/A")
    plan_type = customer.get("plan_type", "Basic")
    revenue = customer.get("revenue", 0.0)
    login_freq = customer.get("login_frequency", 0)
    session_dur = customer.get("session_duration", 0)
    feat_usage = customer.get("feature_usage", 0)
    tickets = customer.get("support_tickets", 0)
    age = customer.get("subscription_age", 0)
    inactive = customer.get("last_active_days", 0)
    
    prob_pct = prediction.get("churn_percentage", 50.0)
    risk_cat = prediction.get("risk_category", "Medium")
    top_factors = prediction.get("top_churn_factors", [])
    top_strengths = prediction.get("top_retention_factors", [])
    
    # Check if OpenAI API Key is available
    if api_key and api_key.strip():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Construct a clear, structured prompt
            prompt = f"""You are a senior SaaS Customer Success Director and Retention Architect.
Design a highly tailored Customer Success Playbook to retain this customer:

### CUSTOMER profile
- Name: {name} (ID: {customer_id})
- Plan Tier: {plan_type}
- Monthly Revenue: ${revenue}/mo
- Relationship Age: {age} months

### USAGE & ENGAGEMENT METRICS
- Logins/Month: {login_freq}
- Avg Session Duration: {session_dur} mins
- Feature Usage: {feat_usage}%
- Days since last active: {inactive} days
- Support Tickets (last 30 days): {tickets}

### MACHINE LEARNING INSIGHTS
- Churn Risk: {prob_pct}% ({risk_cat} Risk)
- Top Churn Drivers (SHAP): {', '.join(top_factors) if top_factors else 'None'}
- Top Retention Strengths (SHAP): {', '.join(top_strengths) if top_strengths else 'None'}

Please construct a comprehensive retention playbook in valid Markdown. Use an encouraging, analytical, and professional tone.
Structure your playbook with the following exact headers:
1. ### Executive Summary & Risk Assessment
2. ### Immediate Action Plan (Next 48 Hours)
3. ### Engagement & Value Realization Strategy
4. ### Tailored Incentives & Plan Optimization
"""
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional SaaS customer success consulting agent. Speak directly to customer success managers. Keep advice tactical, specific, and metric-driven."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=1024
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Exception contacting OpenAI API: {e}")
            return f"### ⚠️ OpenAI API Error\nFailed to generate a custom playbook. Please check if your API key is valid and has sufficient quota. \n\n*Error details: {str(e)}*"
            
    # --- RULES-BASED FALLBACK PLAYBOOK ---
    return _generate_fallback_playbook(customer, prediction)

def chat_with_customer(
    customer: Dict[str, Any],
    prediction: Dict[str, Any],
    messages: List[Dict[str, str]],
    api_key: str = None
) -> str:
    """
    Interfaces with ChatGPT to conduct an interactive customer health chat.
    Uses a context-aware mock engine if no key is provided.
    """
    name = customer.get("name", "Valued Customer")
    customer_id = customer.get("customer_id", "N/A")
    plan_type = customer.get("plan_type", "Basic")
    revenue = customer.get("revenue", 0.0)
    login_freq = customer.get("login_frequency", 0)
    session_dur = customer.get("session_duration", 0)
    feat_usage = customer.get("feature_usage", 0)
    tickets = customer.get("support_tickets", 0)
    age = customer.get("subscription_age", 0)
    inactive = customer.get("last_active_days", 0)
    
    prob_pct = prediction.get("churn_percentage", 50.0)
    risk_cat = prediction.get("risk_category", "Medium")
    top_factors = prediction.get("top_churn_factors", [])
    top_strengths = prediction.get("top_retention_factors", [])
    
    system_prompt = f"""You are ChurnAI, an expert customer success retention assistant.
The user is a customer success manager asking you questions about this customer account.
Use this customer's profile to answer their questions, draft email outreach templates, outline action items, or suggest meeting scripts.

### CUSTOMER PROFILE
- Name: {name} (ID: {customer_id})
- Plan Tier: {plan_type}
- Monthly Revenue: ${revenue}/mo
- Relationship Age: {age} months

### METRICS
- Logins/Month: {login_freq}
- Session Length: {session_dur} mins
- Feature Usage: {feat_usage}%
- Days since last active: {inactive} days
- Support Tickets: {tickets}

### MACHINE LEARNING INSIGHTS
- Churn Risk: {prob_pct}% ({risk_cat} Risk)
- Top Churn Drivers: {', '.join(top_factors) if top_factors else 'None'}
- Top Retention Anchors: {', '.join(top_strengths) if top_strengths else 'None'}

Be friendly, professional, analytical, and highly tactical. Always refer to the customer by their name '{name}'. Use markdown styling for responses."""

    if api_key and api_key.strip():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Combine system prompt with message history
            openai_messages = [{"role": "system", "content": system_prompt}]
            # Only send clean user/assistant/system messages from request
            for msg in messages:
                openai_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
                
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=openai_messages,
                temperature=0.5,
                max_tokens=800
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Exception calling OpenAI Chat: {e}")
            return f"⚠️ **OpenAI API Error**: Failed to generate a response. Please check if your API key is valid and has sufficient quota. \n\n*Error details: {str(e)}*"
            
    # --- CONTEXT-AWARE MOCK CHATBOT FALLBACK ---
    return _generate_fallback_chat_reply(customer, prediction, messages)

def _generate_fallback_chat_reply(customer: Dict[str, Any], prediction: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    """Generates highly realistic replies matching the customer metrics and the user's questions."""
    name = customer.get("name", "Valued Customer")
    plan_type = customer.get("plan_type", "Basic")
    revenue = customer.get("revenue", 0.0)
    tickets = int(customer.get("support_tickets", 0))
    inactive = int(customer.get("last_active_days", 0))
    
    last_user_message = messages[-1].get("content", "").lower() if messages else ""
    
    # Analyze prompt intent
    if "email" in last_user_message or "outreach" in last_user_message or "write" in last_user_message or "template" in last_user_message:
        subject = f"Improving your experience with ChurnIntel - Quick Sync"
        body = f"Hi {name.split(' #')[0]},\n\nI hope you're doing well.\n\nI noticed you haven't logged in for the last {inactive} days, and I wanted to check in and see if you ran into any roadblocks. "
        if tickets > 0:
            body += f"I see you recently opened {tickets} support requests. I want to make sure those issues were fully resolved to your satisfaction.\n\n"
        else:
            body += "I'd love to make sure you're getting the full value out of your subscription.\n\n"
            
        body += "Do you have 10 minutes for a quick feedback call this week?\n\nBest regards,\nCustomer Success Team"
        
        reply = f"""Here is a tailored email outreach template you can send to the team at **{name}**:

```text
Subject: {subject}

{body}
```

*Tip: For **{plan_type}** tier accounts, this should be followed up with an automated nudging campaign if they do not reply within 3 days.*"""

    elif "why" in last_user_message or "cause" in last_user_message or "reason" in last_user_message or "churn" in last_user_message:
        top_factors = prediction.get("top_churn_factors", [])
        factor_desc = []
        for factor in top_factors:
            if factor == 'last_active_days':
                factor_desc.append(f"- **Silent Churn/Inactivity**: They haven't logged in for **{inactive} days**. This is the highest risk signal.")
            elif factor == 'support_tickets':
                factor_desc.append(f"- **Support Friction**: They've filed **{tickets} support tickets** in the last month, showing system frustration.")
            elif factor == 'feature_usage':
                factor_desc.append(f"- **Low Adoption**: They are using less than 30% of key features, indicating they haven't integrated the tool into their workflow.")
        factors_str = "\n".join(factor_desc) if factor_desc else "- General drop in login habits and monthly active time."

        reply = f"""**{name}** is showing elevated risk due to the following primary factors:
{factors_str}

**Retention Strategy Recommendation**:
Since this is a **{plan_type}** tier account generating **${revenue}/mo**, I recommend starting with direct support verification. Check their open tickets and resolve them immediately, then follow up with a personalized re-engagement sync."""

    elif "discount" in last_user_message or "offer" in last_user_message or "deal" in last_user_message:
        if plan_type == "Enterprise":
            reply = f"""For **Enterprise** clients like **{name}** contributing **${revenue}/mo**, **do not offer a direct pricing discount first**. 

Instead, propose:
1. **Premium Support SLA Upgrade**: Offer dedicated Slack integration and 2-hour SLA response times.
2. **Dedicated Solutions Engineer Review**: Arrange a 1-on-1 workflow optimization session to help them integrate features.
3. **Billing Extension**: If they are struggling with budget, offer a 2-month extension at 50% off in exchange for a 12-month contract renewal."""
        else:
            reply = f"""For **{name}** (on the **{plan_type}** plan), you can offer a **25% discount for 3 billing cycles** as a retention incentive. 

Alternatively, offer to upgrade them to the next tier for 30 days for free so they can experience our advanced collaboration features, which could re-engage their active users."""
    else:
        reply = f"""Hello! I am **ChurnAI**, your customer success companion. 

I'm analyzing **{name}** (a **{plan_type}** account contributing **${revenue}/mo**). They currently have a **{prediction.get('churn_percentage', 50.0)}% churn risk**.

I can help you with:
- ✉️ Drafting custom email outreach templates.
- 🔍 Explaining the core reasons why they are flagged.
- 💡 Providing pricing options, incentives, or contract negotiation strategies.

*Note: Connect your OpenAI ChatGPT API key in the sidebar configuration to unlock full, open-ended conversational capabilities with gpt-4o-mini!*"""
        
    return reply

def _generate_fallback_playbook(customer: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    """Generates a detailed, context-aware retention playbook locally (Fallback)."""
    name = customer.get("name", "Valued Customer")
    customer_id = customer.get("customer_id", "N/A")
    plan_type = customer.get("plan_type", "Basic")
    revenue = customer.get("revenue", 0.0)
    login_freq = customer.get("login_frequency", 0)
    feat_usage = customer.get("feature_usage", 0)
    tickets = customer.get("support_tickets", 0)
    age = customer.get("subscription_age", 0)
    inactive = customer.get("last_active_days", 0)
    
    prob_pct = prediction.get("churn_percentage", 50.0)
    risk_cat = prediction.get("risk_category", "Medium")
    top_factors = prediction.get("top_churn_factors", [])
    
    primary_issue = "Under-utilization"
    if inactive > 10:
        primary_issue = "Inactivity (Silent Churn)"
    elif tickets >= 4:
        primary_issue = "Support Overload / Product Frustration"
    elif login_freq < 5:
        primary_issue = "Low Login Frequency"
    elif feat_usage < 30:
        primary_issue = "Feature Adoption Bottleneck"

    plan_text = ""
    incentive_text = ""
    immediate_steps = []
    engagement_steps = []
    
    if plan_type == "Enterprise":
        plan_text = "Enterprise customer representing high MRR. Requires white-glove, human-in-the-loop customer success."
        immediate_steps = [
            f"**Assign Dedicated CSM**: Have their Account Manager immediately email and schedule an urgent 15-minute Strategy Alignment Call.",
            f"**Direct Executive Outreach**: Send a note from the CEO/VP of Product to check if our platform is meeting their expectations.",
            f"**Audit Support Queue**: Review all `{tickets}` open support tickets and pull in senior engineering to resolve any lingering issues before the call."
        ]
        engagement_steps = [
            "**Custom Training Session**: Offer a free live onboarding workshop for their entire team to drive adoption.",
            "**Product Roadmap Review**: Shares upcoming feature releases that address their specific business goals.",
            "**API / Integrations Review**: Set up a developer sync to check if we can deeper integrate our API to lock in the customer."
        ]
        incentive_text = (
            f"As an **Enterprise** customer contributing **${revenue}/mo**, do NOT offer generic discounts. "
            f"Instead, offer:\n"
            f"- **Free custom integration assistance** (valued at $1,500).\n"
            f"- **2 months of a premium add-on feature** for free to demonstrate new value.\n"
            f"- **A dedicated Slack channel** with support SLA guarantees."
        )
    elif plan_type == "Pro":
        plan_text = "Pro plan user. Requires focused engagement, combining personalized support with automated workflows."
        immediate_steps = [
            f"**Personalized Video Outreach**: Create a 60-second Loom video pointing out one feature they haven't used that can save them time.",
            f"**High-Priority Ticket Resolution**: Escalate any unresolved support issues. Ensure a senior support engineer responds within 2 hours.",
            f"**Risk Flagging**: Mark account status as 'At Risk' in internal systems to prevent auto-cancellations."
        ]
        engagement_steps = [
            "**In-App Guided Tour**: Trigger a customized walk-through guide targeting feature adoption.",
            "**Email Case Study**: Send a short case study on how similar companies on the Pro plan scaled by 4x using our advanced analytics features."
        ]
        incentive_text = (
            f"For this **Pro** account (**${revenue}/mo**), we can provide:\n"
            f"- **A 20% discount for the next 3 billing cycles** if they agree to a brief feedback call.\n"
            f"- **A complimentary 'Account Health Checkup'** session with a solutions engineer."
        )
    else: # Basic
        plan_text = "Basic account. Requires automated, scalable retention workflows."
        immediate_steps = [
            f"**Automated Re-engagement Sequence**: Enroll customer in the 'Feature Value Highlight' drip campaign.",
            f"**Feedback Survey Email**: Send a one-click survey asking, 'How can we make our platform more helpful for you?'",
            f"**In-App Notification**: Show a custom dark-mode banner highlighting a quick-win dashboard layout."
        ]
        engagement_steps = [
            "**Self-serve documentation push**: Send a targeted newsletter containing 'Top 5 Quick Tips for New Users'.",
            "**Interactive Webinar Invitation**: Invite them to our weekly live Q&A session."
        ]
        incentive_text = (
            f"For the **Basic** account (**${revenue}/mo**):\n"
            f"- Offer an upgrade to **Pro** for 30 days at no extra charge to showcase advanced features.\n"
            f"- Or offer a **30% discount on the next billing month** upon completion of a feedback form."
        )

    shap_advice = []
    for factor in top_factors:
        if factor == 'last_active_days':
            shap_advice.append(f"**Address Inactivity ({inactive} days inactive)**: The customer has not logged in recently. Immediate login re-engagement is required.")
        elif factor == 'support_tickets':
            shap_advice.append(f"**Resolve Support Friction ({tickets} tickets)**: High ticket volume indicates technical or UX roadblocks. Ensure their tickets are closed successfully.")
        elif factor == 'login_frequency':
            shap_advice.append(f"**Boost Low Logins ({login_freq} times/mo)**: Regular usage habits have not formed. Introduce a 'Daily Digest' email option.")
        elif factor == 'feature_usage':
            shap_advice.append(f"**Drive Feature Adoption ({feat_usage}%)**: They use a fraction of features. Target them with product tours.")
        elif factor == 'subscription_age':
            shap_advice.append(f"**Overcome Early-Stage Churn ({age} mo)**: Customer is in the vulnerable early adoption phase. High-touch check-ins are crucial.")

    immediate_list = "\n".join([f"- {step}" for step in immediate_steps])
    engagement_list = "\n".join([f"- {step}" for step in engagement_steps])
    shap_list = "\n".join([f"- {advice}" for advice in shap_advice]) if shap_advice else "- Review usage patterns to identify custom adoption strategies."

    playbook = f"""> [!NOTE]
> *This playbook was generated locally by the platform's rules-based Retention Engine.*

### Executive Summary & Risk Assessment
- **Customer Name**: {name}
- **Account Status**: `{risk_cat}` Churn Risk ({prob_pct}% probability)
- **Revenue Exposure**: **${revenue}/mo** ({plan_type} Tier)
- **Primary Churn Vector**: `{primary_issue}`

**Risk Profile Details**:
{plan_text} The model indicates that their highest risk stems from {', '.join(top_factors) if top_factors else 'general usage decline'}.

### Immediate Action Plan (Next 48 Hours)
{immediate_list}

### Engagement & Value Realization Strategy
{engagement_list}

**SHAP-Based Diagnostic Interventions**:
{shap_list}

### Tailored Incentives & Plan Optimization
{incentive_text}
"""
    return playbook
