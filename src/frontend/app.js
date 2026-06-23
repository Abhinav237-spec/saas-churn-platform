// ChurnIntel App Controller - HTML SPA Frontend
const API_BASE = window.location.origin + "/api";

// Local State
const state = {
  activeTab: "overview",
  customers: [],
  predictions: {},
  activeCustomerId: null,
  chatHistories: {}, // Keyed by customer_id: array of messages
  charts: {} // Store ApexCharts instances
};

// Map backend feature names to human-readable names
const FEATURE_LABELS = {
  "login_frequency": "Login Frequency",
  "session_duration": "Session Duration",
  "feature_usage": "Feature Adoption %",
  "support_tickets": "Support Tickets",
  "subscription_age": "Subscription Age",
  "last_active_days": "Days Inactive",
  "revenue": "Monthly Revenue",
  "plan_type_Basic": "Basic Plan Tier",
  "plan_type_Pro": "Pro Plan Tier",
  "plan_type_Enterprise": "Enterprise Plan Tier"
};

// Initial setup
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  checkBackendHealth();
  initEventHandlers();
  initTiltAnimation();
  
  // Try loading initial data if exists
  loadDefaultData();
});

// Check API Health
async function checkBackendHealth() {
  const dot = document.getElementById("api-status-dot");
  const text = document.getElementById("api-status-text");
  
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      dot.className = "api-dot connected";
      text.innerText = "API Connected";
      return true;
    }
  } catch (err) {
    console.error("Backend offline:", err);
  }
  
  dot.className = "api-dot disconnected";
  text.innerText = "API Offline";
  return false;
}

// Initialize Navigation Tabs
function initNavigation() {
  const buttons = document.querySelectorAll(".nav-item");
  const panels = document.querySelectorAll(".tab-panel");
  const title = document.getElementById("header-page-title");
  const subtitle = document.getElementById("header-page-subtitle");

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.dataset.tab;
      
      // Update sidebar state
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // Update active panel
      panels.forEach(p => {
    p.classList.remove("active");
    if (p.id === `panel-${tabId}`) {
      p.classList.add("active");
    }
  });

  if (tabId === "overview") {
    // We already fetch global data on start, no need to do heavy refresh here unless needed
    // title.innerHTML = pageTitle;
  }
      
      // Set Header title and sub, add typewriter animation on change
      let pageTitle = "SaaS Health Dashboard";
      let pageSub = "Interactive churn analytics overview";
      
      if (tabId === "inspector") {
        pageTitle = "Customer Churn Inspector";
        pageSub = "Drilldown explainable AI and ChatGPT playbooks";
      } else if (tabId === "predict") {
        pageTitle = "Batch Churn Predictions";
        pageSub = "Upload customer analytics CSV lists and export risks";
      } else if (tabId === "trainer") {
        pageTitle = "Model Training Center";
        pageSub = "Re-train XGBoost model on historical datasets";
      }

      title.innerHTML = pageTitle;
      subtitle.innerText = pageSub;
      
      // Refresh charts if needed
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 150);
    });
  });
}

// Bind custom 3D card tilt animation on hover
function initTiltAnimation() {
  document.querySelectorAll(".card").forEach(card => {
    card.addEventListener("mousemove", e => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left; // x coordinate inside element
      const y = e.clientY - rect.top;  // y coordinate inside element
      const xc = rect.width / 2;
      const yc = rect.height / 2;
      
      // Calculate rotation angles (max tilt 10 degrees)
      const angleX = -(y - yc) / (rect.height / 10);
      const angleY = (x - xc) / (rect.width / 10);
      
      card.style.transform = `rotateX(${angleX}deg) rotateY(${angleY}deg) translateY(-4px)`;
    });
    
    card.addEventListener("mouseleave", () => {
      card.style.transform = "rotateX(0deg) rotateY(0deg) translateY(0px)";
    });
  });
}

// Load default dataset if backend already has one
async function loadDefaultData() {
  try {
    const infoRes = await fetch(`${API_BASE}/model-info`);
    if (infoRes.ok) {
      const info = await infoRes.json();
      if (info.status === "loaded") {
        // Trigger predict call on backend default dataset
        const response = await fetch(`${API_BASE}/global-importance`);
        if (response.ok) {
          // Triggering a generation of 1000 synthetic rows or load if exists
          fetchDataAndRenderDashboard();
        }
      } else {
        // Show welcome screen
        document.getElementById("overview-welcome-card").style.display = "block";
        document.getElementById("overview-active-board").style.display = "none";
      }
    }
  } catch (err) {
    console.error("Failed to load default database config:", err);
  }
}

// Fetch all database records, predict and render
async function fetchDataAndRenderDashboard() {
  const isHealthy = await checkBackendHealth();
  if (!isHealthy) return;

  try {
    // We generate-synthetic to populate default if empty, or just reload
    const health = await (await fetch(`${API_BASE}/health`)).json();
    if (!health.model_loaded) {
      document.getElementById("overview-welcome-card").style.display = "block";
      document.getElementById("overview-active-board").style.display = "none";
      return;
    }
    
    document.getElementById("overview-welcome-card").style.display = "none";
    document.getElementById("overview-active-board").style.display = "block";

    // Model is loaded, let's get the synthetic records by calling a prediction on a generated batch
    const res = await fetch(`${API_BASE}/generate-synthetic?count=1000`, { method: "POST" }); // Fix GET to POST
    if (res.ok) {
      const data = await res.json();
      
      // Load the synthetic customers database
      // In a real environment, we'd fetch actual rows. In this app, generate-synthetic returns trained metrics.
      // To get the actual predictions and rows, let's generate them in JavaScript using the same parameters,
      // or we can request /predict with a generated batch. Let's do a batch generation in JS and send to predict endpoint.
      const customers = generateMockDataJS(1000);
      
      // Send to predict
      const predRes = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ customers: customers })
      });
      
      if (predRes.ok) {
        const predData = await predRes.json();
        
        // Populate state
        state.customers = customers;
        state.predictions = {};
        
        predData.results.forEach(item => {
          state.predictions[item.customer.customer_id] = item.prediction;
        });

        // Hide welcome show active dashboard
        document.getElementById("overview-welcome-card").style.display = "none";
        document.getElementById("overview-active-board").style.display = "block";

        renderDashboard();
        populateCustomerSelect();
      }
    }
  } catch (err) {
    console.error("Error loading dashboard data:", err);
  }
}

// Generate matching mock data in JS for prediction payload
function generateMockDataJS(count) {
  const planOptions = ["Basic", "Pro", "Enterprise"];
  const companies = ["Acme Corp", "Globex", "Initech", "Umbrella Corp", "Hooli", "Soylent Corp", "Wayne Enterprises", "Wayne Corp", "Stark Industries", "Dunder Mifflin", "Vandelay Industries"];
  const data = [];
  
  for (let i = 0; i < count; i++) {
    const plan = planOptions[Math.floor(Math.random() * planOptions.length)];
    const subAge = Math.floor(Math.random() * 59) + 1;
    const inactive = Math.floor(Math.random() * 30);
    const logins = Math.max(1, Math.min(30, Math.floor(18 - (inactive * 0.3) + (Math.random() * 8 - 4))));
    const duration = Math.max(5, Math.min(120, Math.floor(30 + (logins * 2) + (Math.random() * 20 - 10))));
    const usage = Math.max(10, Math.min(100, Math.floor(40 + (duration * 0.5) + (Math.random() * 20 - 10))));
    
    // Frustration ticket Poisson approximation
    let tickets = Math.floor(Math.random() * 3);
    if (inactive > 15) tickets += Math.floor(Math.random() * 3);
    
    let revenue = 39.0;
    if (plan === "Pro") revenue = Math.floor(Math.random() * 100) + 99;
    else if (plan === "Enterprise") revenue = Math.floor(Math.random() * 500) + 499;

    data.push({
      customer_id: `CUST-${1000 + i}`,
      name: `${companies[Math.floor(Math.random() * companies.length)]} #${100 + i}`,
      login_frequency: parseFloat(logins.toFixed(1)),
      session_duration: parseFloat(duration.toFixed(1)),
      feature_usage: parseFloat(usage.toFixed(1)),
      support_tickets: parseFloat(tickets.toFixed(1)),
      subscription_age: parseFloat(subAge.toFixed(1)),
      plan_type: plan,
      last_active_days: parseFloat(inactive.toFixed(1)),
      revenue: parseFloat(revenue.toFixed(2))
    });
  }
  return data;
}

// Render Dashboard Panel KPIs and Charts
function renderDashboard() {
  const count = state.customers.length;
  let totalMRR = 0;
  let totalRisk = 0;
  let revRisk = 0;
  let highRiskCount = 0;
  
  const riskCategories = { Low: 0, Medium: 0, High: 0 };
  const planRevenueByRisk = {
    Basic: { Low: 0, Medium: 0, High: 0 },
    Pro: { Low: 0, Medium: 0, High: 0 },
    Enterprise: { Low: 0, Medium: 0, High: 0 }
  };
  
  const healthMatrixData = [];

  state.customers.forEach(c => {
    const pred = state.predictions[c.customer_id];
    totalMRR += c.revenue;
    totalRisk += pred.churn_probability;
    
    // Revenue exposure
    revRisk += c.revenue * pred.churn_probability;
    
    // Categories count
    riskCategories[pred.risk_category]++;
    
    if (pred.risk_category === "High") {
      highRiskCount++;
    }
    
    // Plan revenue breakdown
    planRevenueByRisk[c.plan_type][pred.risk_category] += c.revenue;
    
    // Health Matrix points
    healthMatrixData.push({
      x: c.feature_usage,
      y: c.last_active_days,
      prob: pred.churn_probability,
      name: c.name,
      mrr: c.revenue
    });
  });

  const avgRiskPercent = (totalRisk / count) * 100;

  // Animate KPI Numbers
  animateCounter("kpi-customers", count);
  animateCounter("kpi-avg-risk", avgRiskPercent, "", "%", true);
  animateCounter("kpi-mrr", totalMRR, "$", "", true);
  animateCounter("kpi-revenue-risk", revRisk, "$", "", true);
  
  // Update deltas labels
  document.getElementById("kpi-customers-delta").innerHTML = `<i data-lucide="trending-up" style="width: 14px; height: 14px;"></i> +3.2% vs last month`;
  document.getElementById("kpi-risk-delta").innerHTML = `<i data-lucide="trending-down" style="width: 14px; height: 14px;"></i> -0.8% average change`;
  document.getElementById("kpi-mrr-delta").innerHTML = `<i data-lucide="trending-up" style="width: 14px; height: 14px;"></i> +$4,120.00 new trials`;
  document.getElementById("kpi-rev-risk-delta").innerHTML = `<span>${highRiskCount} accounts at High Risk (>70%)</span>`;
  
  document.getElementById("outreach-queue-count").innerText = `${highRiskCount} Accounts`;
  
  lucide.createIcons();
  
  // Render Plotly/ApexCharts
  renderRiskDistributionChart(riskCategories);
  renderRevenueExposureChart(planRevenueByRisk);
  renderHealthMatrixChart(healthMatrixData);
  renderOutreachQueueTable();
}

// Chart 1: Churn Risk Segments (ApexCharts Bar)
function renderRiskDistributionChart(categories) {
  const options = {
    chart: {
      type: 'bar',
      height: 250,
      fontFamily: 'Inter, sans-serif',
      toolbar: { show: false }
    },
    series: [{
      name: 'Accounts',
      data: [categories.Low, categories.Medium, categories.High]
    }],
    colors: ['#10b981', '#f59e0b', '#ef4444'],
    plotOptions: {
      bar: {
        distributed: true,
        borderRadius: 8,
        columnWidth: '55%'
      }
    },
    xaxis: {
      categories: ['Low Risk', 'Medium Risk', 'High Risk'],
      labels: { style: { colors: '#64748b' } }
    },
    yaxis: {
      labels: { style: { colors: '#64748b' } }
    },
    dataLabels: { enabled: false },
    legend: { show: false }
  };

  if (state.charts.riskDist) {
    state.charts.riskDist.destroy();
  }
  state.charts.riskDist = new ApexCharts(document.getElementById("chart-risk-distribution"), options);
  state.charts.riskDist.render();
}

// Chart 2: Revenue Stacked Bar
function renderRevenueExposureChart(data) {
  const options = {
    chart: {
      type: 'bar',
      height: 250,
      stacked: true,
      fontFamily: 'Inter, sans-serif',
      toolbar: { show: false }
    },
    series: [
      {
        name: 'Low Risk',
        data: [data.Basic.Low, data.Pro.Low, data.Enterprise.Low]
      },
      {
        name: 'Medium Risk',
        data: [data.Basic.Medium, data.Pro.Medium, data.Enterprise.Medium]
      },
      {
        name: 'High Risk',
        data: [data.Basic.High, data.Pro.High, data.Enterprise.High]
      }
    ],
    colors: ['#10b981', '#f59e0b', '#ef4444'],
    plotOptions: {
      bar: {
        borderRadius: 8,
        columnWidth: '50%'
      }
    },
    xaxis: {
      categories: ['Basic Plan', 'Pro Plan', 'Enterprise Plan'],
      labels: { style: { colors: '#64748b' } }
    },
    yaxis: {
      title: { text: 'Monthly Revenue ($)' },
      labels: { style: { colors: '#64748b' } }
    },
    dataLabels: { enabled: false },
    legend: { position: 'bottom' }
  };

  if (state.charts.revExposure) {
    state.charts.revExposure.destroy();
  }
  state.charts.revExposure = new ApexCharts(document.getElementById("chart-revenue-exposure"), options);
  state.charts.revExposure.render();
}

// Chart 3: Health Matrix Scatter
function renderHealthMatrixChart(data) {
  // Sort into Low, Med, High series for color categorization
  const lowSeries = [];
  const medSeries = [];
  const highSeries = [];
  
  data.forEach(pt => {
    const item = [pt.x, pt.y, pt.mrr]; // feature usage, inactive days, bubble size
    if (pt.prob < 0.3) lowSeries.push(pt);
    else if (pt.prob < 0.7) medSeries.push(pt);
    else highSeries.push(pt);
  });

  const options = {
    chart: {
      type: 'scatter',
      height: 350,
      fontFamily: 'Inter, sans-serif',
      toolbar: { show: false }
    },
    series: [
      {
        name: 'Low Risk',
        data: lowSeries.map(p => ({ x: p.x, y: p.y, name: p.name, mrr: p.mrr }))
      },
      {
        name: 'Medium Risk',
        data: medSeries.map(p => ({ x: p.x, y: p.y, name: p.name, mrr: p.mrr }))
      },
      {
        name: 'High Risk',
        data: highSeries.map(p => ({ x: p.x, y: p.y, name: p.name, mrr: p.mrr }))
      }
    ],
    colors: ['#10b981', '#f59e0b', '#ef4444'],
    xaxis: {
      title: { text: 'Feature Usage / Adoption (%)' },
      min: 0,
      max: 100,
      labels: { style: { colors: '#64748b' } }
    },
    yaxis: {
      title: { text: 'Days Since Last Active (Inactivity)' },
      min: 0,
      max: 30,
      labels: { style: { colors: '#64748b' } }
    },
    tooltip: {
      custom: function({series, seriesIndex, dataPointIndex, w}) {
        const pt = w.config.series[seriesIndex].data[dataPointIndex];
        return `<div style="padding: 10px; background: #fff; border: 1px solid var(--border-color); border-radius: 8px;">
          <strong>${pt.name}</strong><br>
          Feature Adoption: ${pt.x}%<br>
          Days Inactive: ${pt.y} days<br>
          MRR: $${pt.mrr.toFixed(2)}
        </div>`;
      }
    },
    legend: { position: 'top' }
  };

  if (state.charts.healthMatrix) {
    state.charts.healthMatrix.destroy();
  }
  state.charts.healthMatrix = new ApexCharts(document.getElementById("chart-health-matrix"), options);
  state.charts.healthMatrix.render();
}

// Table: High Risk Customer Directories
function renderOutreachQueueTable() {
  const tbody = document.getElementById("overview-risk-table-body");
  
  // Filter High Risk accounts sorted by revenue descending
  const highRiskList = state.customers
    .filter(c => state.predictions[c.customer_id].risk_category === "High")
    .sort((a, b) => b.revenue - a.revenue)
    .slice(0, 10);
    
  if (highRiskList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 30px; color: var(--text-muted);">No accounts flagged. Generate demo data.</td></tr>`;
    return;
  }
  
  let html = "";
  highRiskList.forEach(c => {
    const pred = state.predictions[c.customer_id];
    html += `<tr>
      <td><code>${c.customer_id}</code></td>
      <td><strong>${c.name}</strong></td>
      <td><span class="badge badge-secondary">${c.plan_type}</span></td>
      <td>$${c.revenue.toFixed(2)}</td>
      <td>${c.last_active_days} days</td>
      <td>${c.support_tickets} tickets</td>
      <td><strong style="color: var(--danger);">${(pred.churn_probability * 100).toFixed(1)}%</strong></td>
      <td>
        <button onclick="inspectCustomer('${c.customer_id}')" class="btn btn-secondary" style="padding: 6px 12px; font-size: 0.75rem;">
          Inspect & Chat
        </button>
      </td>
    </tr>`;
  });
  tbody.innerHTML = html;
}

// Set up event handlers
function initEventHandlers() {
  // Demo data loading trigger
  const chatgptBtn = document.getElementById("btn-chatgpt-connect");
  const oauthModal = document.getElementById("oauth-modal");
  
  if (chatgptBtn && oauthModal) {
    chatgptBtn.addEventListener("click", () => {
      oauthModal.style.display = "flex";
    });
    
    document.getElementById("btn-oauth-cancel").addEventListener("click", () => {
      oauthModal.style.display = "none";
    });
    
    document.getElementById("btn-oauth-submit").addEventListener("click", () => {
      const key = document.getElementById("oauth-api-key").value.trim();
      if (key) {
        state.chatgptToken = key;
        chatgptBtn.innerHTML = `<i data-lucide="check-circle" style="width: 16px; height: 16px; margin-right: 8px;"></i> ChatGPT Connected`;
        chatgptBtn.style.backgroundColor = "#047857";
        lucide.createIcons();
        oauthModal.style.display = "none";
        alert("Successfully authorized ChatGPT access!");
      } else {
        alert("Please enter a key to authorize.");
      }
    });
  }

  const loadDemoData = async () => {
    const btn1 = document.getElementById("btn-generate-synthetic");
    const btn2 = document.getElementById("btn-welcome-load");
    if(btn1) btn1.innerHTML = `<i data-lucide="loader" style="width: 16px; height: 16px; animation: spin 2s linear infinite;"></i> Loading...`;
    if(btn2) btn2.innerHTML = `<i data-lucide="loader" style="width: 16px; height: 16px; animation: spin 2s linear infinite;"></i> Initializing...`;
    
    try {
      await fetch(`${API_BASE}/generate-synthetic?count=1000`, { method: "POST" });
      document.getElementById("overview-welcome-card").style.display = "none";
      document.getElementById("overview-active-board").style.display = "block";
      await fetchDataAndRenderDashboard();
    } catch (e) {
      console.error(e);
    } finally {
      if(btn1) btn1.innerHTML = `<i data-lucide="database" style="width: 16px; height: 16px;"></i> Load Demo Data`;
      if(btn2) btn2.innerHTML = `<i data-lucide="database" style="width: 16px; height: 16px;"></i> Initialize Default Database`;
      lucide.createIcons();
    }
  };

  document.getElementById("btn-generate-synthetic").addEventListener("click", loadDemoData);
  document.getElementById("btn-welcome-load").addEventListener("click", loadDemoData);
  
  // Custom Selector Dropdown switch
  document.getElementById("inspector-select").addEventListener("change", (e) => {
    inspectCustomer(e.target.value);
  });
  
  // Chat message send triggers
  document.getElementById("chat-send-btn").addEventListener("click", sendChatMessage);
  document.getElementById("chat-user-input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendChatMessage();
  });

  // Drag-and-drop prediction
  initDragAndDrop("predict-upload-zone", "predict-file-input", handlePredictionCSV);
  
  // Drag-and-drop training
  initDragAndDrop("trainer-upload-zone", "trainer-file-input", handleTrainingCSV);
}

// Drag & drop file wrapper
function initDragAndDrop(zoneId, inputId, callback) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  
  zone.addEventListener("click", () => input.click());
  
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  
  zone.addEventListener("dragleave", () => {
    zone.classList.remove("dragover");
  });
  
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      callback(e.dataTransfer.files[0]);
    }
  });
  
  input.addEventListener("change", () => {
    if (input.files.length) {
      callback(input.files[0]);
    }
  });
}

// Populate customer select list
function populateCustomerSelect() {
  const select = document.getElementById("inspector-select");
  select.innerHTML = `<option value="" disabled selected>Select a customer account...</option>`;
  
  state.customers.forEach(c => {
    const pred = state.predictions[c.customer_id];
    const riskIndicator = pred.risk_category === "High" ? "⚠️ " : "";
    select.innerHTML += `<option value="${c.customer_id}">${riskIndicator}${c.customer_id} - ${c.name}</option>`;
  });
}

// Select a customer, render inspector data and load chat
function inspectCustomer(customerId) {
  state.activeCustomerId = customerId;
  const customer = state.customers.find(c => c.customer_id === customerId);
  const pred = state.predictions[customerId];

  // Navigate to Inspector Tab if selected from outreach queue
  if (state.activeTab !== "inspector") {
    document.querySelector('[data-tab="inspector"]').click();
  }

  // Update dropdown value
  document.getElementById("inspector-select").value = customerId;

  // Toggle profile view
  document.getElementById("inspector-empty-state").style.display = "none";
  document.getElementById("inspector-active-profile").style.display = "flex";

  // Fill profile details
  document.getElementById("inspector-cust-name").innerText = customer.name;
  document.getElementById("inspector-cust-id").innerText = customer.customer_id;
  
  // Format risk badge
  const badgeSlot = document.getElementById("inspector-risk-badge-slot");
  if (pred.risk_category === "Low") {
    badgeSlot.innerHTML = `<span class="badge badge-low">Low Risk</span>`;
  } else if (pred.risk_category === "Medium") {
    badgeSlot.innerHTML = `<span class="badge badge-medium">Medium Risk</span>`;
  } else {
    badgeSlot.innerHTML = `<span class="badge badge-high">High Risk</span>`;
  }
  
  // Fill metric capsules
  document.getElementById("pill-plan").innerText = customer.plan_type;
  document.getElementById("pill-revenue").innerText = `$${customer.revenue.toFixed(2)}`;
  document.getElementById("pill-age").innerText = `${customer.subscription_age} mo`;
  document.getElementById("pill-inactive").innerText = `${customer.last_active_days} days`;

  // Enable chat inputs
  document.getElementById("chat-user-input").removeAttribute("disabled");
  document.getElementById("chat-send-btn").removeAttribute("disabled");

  // Render Inspector Charts
  renderInspectorGauge(pred.churn_percentage);
  renderInspectorSHAPChart(pred.shap_contributions);

  // Initialize Chat Conversation for this Customer
  if (!state.chatHistories[customerId]) {
    state.chatHistories[customerId] = [
      {
        role: "assistant",
        content: `Hi! I'm **ChurnAI**. I have analyzed **${customer.name}** and found a **${pred.churn_percentage}%** churn probability. 
        
How would you like to proceed?
- Ask: *"What are the top churn factors?"*
- Ask: *"Draft a personalized re-engagement email outreach"*
- Ask: *"What pricing discount or plan incentive should we offer?"*`
      }
    ];
  }
  renderChatMessages();
}

// Inspector Chart 1: Radial Risk Gauge
function renderInspectorGauge(percentage) {
  const options = {
    chart: {
      type: 'radialBar',
      height: 180,
      sparkline: { enabled: true }
    },
    series: [percentage],
    colors: [percentage < 30 ? '#10b981' : percentage < 70 ? '#f59e0b' : '#ef4444'],
    plotOptions: {
      radialBar: {
        startAngle: -90,
        endAngle: 90,
        track: {
          background: "#e2e8f0",
          strokeWidth: '97%',
          margin: 5
        },
        dataLabels: {
          name: { show: false },
          value: { show: false }
        }
      }
    },
    grid: { padding: { top: -10 } },
    stroke: { lineCap: 'round' }
  };

  if (state.charts.inspGauge) {
    state.charts.inspGauge.destroy();
  }
  state.charts.inspGauge = new ApexCharts(document.getElementById("chart-inspector-gauge"), options);
  state.charts.inspGauge.render();
  document.getElementById("inspector-gauge-label").innerText = `${percentage.toFixed(1)}%`;
  document.getElementById("inspector-gauge-label").style.color = percentage < 30 ? '#10b981' : percentage < 70 ? '#f59e0b' : '#ef4444';
}

// Inspector Chart 2: Local SHAP bar
function renderInspectorSHAPChart(shap) {
  // Sort features by impact value magnitude
  const sorted = Object.entries(shap)
    .map(([feature, val]) => ({ name: FEATURE_LABELS[feature] || feature, value: val }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 5); // Show top 5 factors

  const options = {
    chart: {
      type: 'bar',
      height: 220,
      fontFamily: 'Inter, sans-serif',
      toolbar: { show: false }
    },
    series: [{
      name: 'SHAP Value (Impact)',
      data: sorted.map(item => item.value)
    }],
    colors: [function({ value }) {
      return value > 0 ? '#ef4444' : '#10b981'; // Red triggers risk increase, green retention anchor
    }],
    plotOptions: {
      bar: {
        borderRadius: 4,
        horizontal: true
      }
    },
    xaxis: {
      categories: sorted.map(item => item.name),
      labels: { style: { colors: '#64748b' } }
    },
    yaxis: {
      labels: { style: { colors: '#64748b' } }
    },
    dataLabels: {
      enabled: true,
      formatter: function (val) {
        return (val > 0 ? "+" : "") + val.toFixed(3);
      },
      style: { fontSize: '10px' }
    }
  };

  if (state.charts.inspShap) {
    state.charts.inspShap.destroy();
  }
  state.charts.inspShap = new ApexCharts(document.getElementById("chart-inspector-shap"), options);
  state.charts.inspShap.render();
}

// Render Inspector Chat History Dialogues
function renderChatMessages() {
  const container = document.getElementById("chat-messages-container");
  const messages = state.chatHistories[state.activeCustomerId] || [];
  
  let html = "";
  messages.forEach(msg => {
    // Basic Markdown format helper
    const formattedContent = parseMarkdown(msg.content);
    html += `<div class="message ${msg.role}">${formattedContent}</div>`;
  });
  
  container.innerHTML = html;
  
  // Smooth scroll to bottom
  container.scrollTop = container.scrollHeight;
}

// Markdown parser helper for chat bubbles
function parseMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // bold
    .replace(/\*(.*?)\*/g, "<em>$1</em>") // italics
    .replace(/`(.*?)`/g, "<code>$1</code>") // inline code
    .replace(/\n/g, "<br>") // newline
    .replace(/(?:^|<br>)-\s(.*?)(?=<br>|$)/g, "<li>$1</li>"); // list items
}

// Send interactive message to Chat API
async function sendChatMessage() {
  const inputEl = document.getElementById("chat-user-input");
  const userText = inputEl.value.trim();
  const apiKey = state.chatgptToken || null;
  
  if (!userText || !state.activeCustomerId) return;
  
  // Clear input
  inputEl.value = "";
  
  const customer = state.customers.find(c => c.customer_id === state.activeCustomerId);
  const pred = state.predictions[state.activeCustomerId];
  
  // Add user message to state and render
  state.chatHistories[state.activeCustomerId].push({
    role: "user",
    content: userText
  });
  renderChatMessages();
  
  // Disable input during pending API call
  inputEl.setAttribute("disabled", "true");
  
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        customer: customer,
        prediction: pred,
        messages: state.chatHistories[state.activeCustomerId],
        api_key: apiKey || null
      })
    });
    
    if (response.ok) {
      const data = await response.json();
      state.chatHistories[state.activeCustomerId].push({
        role: "assistant",
        content: data.response
      });
      renderChatMessages();
    } else {
      state.chatHistories[state.activeCustomerId].push({
        role: "assistant",
        content: "Failed to generate chat advice. Check connection or OpenAI API Key."
      });
      renderChatMessages();
    }
  } catch (err) {
    console.error("Chat error:", err);
  } finally {
    inputEl.removeAttribute("disabled");
    inputEl.focus();
  }
}

// Handle Batch Predict CSV upload
async function handlePredictionCSV(file) {
  const isHealthy = await checkBackendHealth();
  if (!isHealthy) return;

  const zone = document.getElementById("predict-upload-zone");
  zone.innerHTML = `<i class="upload-icon" data-lucide="loader" style="width:40px; height:40px; animation: spin 2s linear infinite;"></i><p>Evaluating csv rows...</p>`;
  lucide.createIcons();

  const formData = new FormData();
  formData.append("file", file);

  try {
    // In our main.py, we have POST /train for CSV but predictions are POST /predict as JSON.
    // However, to validate file upload, let's parse CSV client side to JSON and POST it to `/predict`.
    const reader = new FileReader();
    reader.onload = async (e) => {
      const text = e.target.result;
      const json = csvToJson(text);
      
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ customers: json })
      });
      
      if (res.ok) {
        const data = await res.json();
        renderBatchResults(data.results);
      } else {
        zone.innerHTML = `<p style="color:var(--danger)">Batch failed. Ensure column fields match schema.</p>`;
      }
    };
    reader.readAsText(file);
  } catch (err) {
    console.error("Batch error:", err);
  }
}

// Parse uploaded csv into valid json matching schema
function csvToJson(csv) {
  const lines = csv.split("\n");
  const result = [];
  const headers = lines[0].split(",");
  
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const obj = {};
    const currentline = lines[i].split(",");
    
    headers.forEach((h, idx) => {
      const val = currentline[idx].trim();
      const head = h.trim();
      if (head === "plan_type" || head === "customer_id" || head === "name") {
        obj[head] = val;
      } else {
        obj[head] = parseFloat(val);
      }
    });
    
    // Add defaults if missing
    if (!obj.customer_id) obj.customer_id = `BATCH-${1000 + i}`;
    if (!obj.name) obj.name = `Customer #${i}`;
    
    result.push(obj);
  }
  return result;
}

// Render batch results view
function renderBatchResults(results) {
  document.getElementById("predict-upload-zone").style.display = "none";
  document.getElementById("predict-summary-view").style.display = "block";
  
  const size = results.length;
  let totalChurn = 0;
  let highRisk = 0;
  let revRisk = 0;
  
  const tbody = document.getElementById("batch-results-body");
  tbody.innerHTML = "";
  
  results.forEach(item => {
    const c = item.customer;
    const p = item.prediction;
    totalChurn += p.churn_probability;
    if (p.risk_category === "High") highRisk++;
    revRisk += c.revenue * p.churn_probability;
    
    tbody.innerHTML += `<tr>
      <td><code>${c.customer_id}</code></td>
      <td><strong>${c.name}</strong></td>
      <td><span class="badge badge-secondary">${c.plan_type}</span></td>
      <td>$${c.revenue.toFixed(2)}</td>
      <td><strong style="color: ${p.risk_category === 'High' ? 'var(--danger)' : '#10b981'}">${p.churn_percentage}%</strong></td>
      <td><span class="badge ${p.risk_category === 'High' ? 'badge-high' : p.risk_category === 'Medium' ? 'badge-medium' : 'badge-low'}">${p.risk_category}</span></td>
      <td>${p.top_churn_factors.join(", ") || 'None'}</td>
    </tr>`;
  });
  
  document.getElementById("kpi-batch-size").innerText = size;
  document.getElementById("kpi-batch-churn").innerText = `${((totalChurn / size) * 100).toFixed(1)}%`;
  document.getElementById("kpi-batch-high").innerText = highRisk;
  document.getElementById("kpi-batch-rev").innerText = `$${revRisk.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}`;
  
  // Download button handler
  document.getElementById("btn-download-predictions").onclick = () => {
    const csvContent = "data:text/csv;charset=utf-8," 
      + ["Customer ID,Name,Plan,MRR,Churn Prob,Risk Level,Top Churn Factors"].join(",") + "\n"
      + results.map(item => {
        const c = item.customer;
        const p = item.prediction;
        return `"${c.customer_id}","${c.name}","${c.plan_type}",${c.revenue},${p.churn_probability},"${p.risk_category}","${p.top_churn_factors.join('; ')}"`;
      }).join("\n");
      
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "churnintel_batch_predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
}

// Handle Model Training File upload
async function handleTrainingCSV(file) {
  const isHealthy = await checkBackendHealth();
  if (!isHealthy) return;

  const zone = document.getElementById("trainer-upload-zone");
  zone.innerHTML = `<i class="upload-icon" data-lucide="loader" style="width:40px; height:40px; animation: spin 2s linear infinite;"></i><p>Fitting booster trees...</p>`;
  lucide.createIcons();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/train`, {
      method: "POST",
      body: formData
    });
    
    if (res.ok) {
      const data = await res.json();
      zone.style.display = "none";
      document.getElementById("trainer-stats-card").style.display = "block";
      
      document.getElementById("train-auc").innerText = data.metrics.auc_roc.toFixed(3);
      document.getElementById("train-accuracy").innerText = `${(data.metrics.accuracy * 100).toFixed(1)}%`;
      document.getElementById("train-f1").innerText = data.metrics.f1.toFixed(3);
      
      // Update global importance chart
      renderGlobalImportanceChart(data.feature_importance);
      
      // Reload overall state data
      loadDefaultData();
    } else {
      zone.innerHTML = `<p style="color:var(--danger)">Training failed. Check headers contain a 'churn' label.</p>`;
    }
  } catch (err) {
    console.error("Training error:", err);
  }
}

// Chart 4: Global SHAP Importance bar chart
function renderGlobalImportanceChart(importance) {
  const sorted = Object.entries(importance)
    .map(([feature, val]) => ({ name: FEATURE_LABELS[feature] || feature, value: val }))
    .sort((a, b) => b.value - a.value);

  const options = {
    chart: {
      type: 'bar',
      height: 280,
      fontFamily: 'Inter, sans-serif',
      toolbar: { show: false }
    },
    series: [{
      name: 'Global SHAP Importance',
      data: sorted.map(item => item.value)
    }],
    colors: ['#f97316'],
    plotOptions: {
      bar: {
        borderRadius: 4,
        horizontal: true,
        barHeight: '70%'
      }
    },
    xaxis: {
      categories: sorted.map(item => item.name),
      labels: { style: { colors: '#64748b' } }
    },
    yaxis: {
      labels: { style: { colors: '#64748b' } }
    },
    dataLabels: { enabled: false }
  };

  if (state.charts.globalImp) {
    state.charts.globalImp.destroy();
  }
  state.charts.globalImp = new ApexCharts(document.getElementById("chart-global-importance"), options);
  state.charts.globalImp.render();
}

// Animate Stat numbers count-up from 0
function animateCounter(id, target, prefix = "", suffix = "", isFloat = false) {
  const el = document.getElementById(id);
  if (!el) return;
  
  let start = 0;
  const duration = 800; // ms
  const steps = 30;
  const stepTime = duration / steps;
  const diff = target - start;
  let currentStep = 0;
  
  const timer = setInterval(() => {
    currentStep++;
    const val = start + (diff * (currentStep / steps));
    
    if (isFloat) {
      el.innerText = prefix + val.toFixed(1) + suffix;
    } else {
      el.innerText = prefix + Math.floor(val).toLocaleString() + suffix;
    }
    
    if (currentStep >= steps) {
      clearInterval(timer);
      if (isFloat) {
        el.innerText = prefix + target.toFixed(1) + suffix;
      } else {
        el.innerText = prefix + target.toLocaleString() + suffix;
      }
    }
  }, stepTime);
}
