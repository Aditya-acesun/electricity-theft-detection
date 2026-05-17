import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import time

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: white; }
    .metric-card {
        background: #1e293b;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #334155;
    }
    .theft-alert {
        background: #7f1d1d;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #ef4444;
    }
    .normal-alert {
        background: #14532d;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #22c55e;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://api:8000"

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=80)
    st.title("⚡ ETD System")
    st.markdown("---")
    
    # API Status
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Error")
    except:
        st.error("🔴 API Offline")
    
    st.markdown("---")
    st.markdown("### Model Info")
    try:
        info = requests.get(f"{API_URL}/info", timeout=2).json()
        st.metric("ROC-AUC", "0.7662")
        st.metric("Features", info['n_features'])
        st.metric("Threshold", info['threshold'])
    except:
        st.warning("Could not load model info")
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "AI system detecting electricity theft "
        "using XGBoost ML model trained on "
        "42,372 real customers."
    )

# ── Main Page ─────────────────────────────────────────────────────
st.title("⚡ Electricity Theft Detection System")
st.markdown("*AI-powered fraud detection using smart meter data*")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Customer",
    "📊 Batch Analysis", 
    "📈 Live Simulation"
])

# ── Tab 1: Single Customer ────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Input")
        customer_id = st.text_input("Customer ID", "CUST_001")
        
        pattern = st.selectbox("Select Pattern", [
            "Normal Customer",
            "Theft Pattern (Bypass)",
            "Theft Pattern (Tampering)",
            "Suspicious Pattern"
        ])
        
        # Generate readings based on pattern
        if pattern == "Normal Customer":
            readings = list(
                8 + 3*np.sin(np.linspace(0, 6*np.pi, 365)) +
                np.random.normal(0, 0.5, 365)
            )
            readings = [max(0, r) for r in readings]
            
        elif pattern == "Theft Pattern (Bypass)":
            readings = list(np.random.uniform(25, 45, 365))
            readings[100:150] = [0] * 50
            
        elif pattern == "Theft Pattern (Tampering)":
            readings = list(np.random.uniform(0.5, 2, 365))
            spike_idx = np.random.choice(365, 20)
            for i in spike_idx:
                readings[i] = np.random.uniform(30, 50)
                
        else:
            readings = list(np.random.uniform(1, 15, 365))
            readings[50:80] = [0] * 30
        
        readings = [round(r, 2) for r in readings]
        
        analyze_btn = st.button(
            "🔍 Analyze Customer",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.subheader("Consumption Pattern")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=readings,
            mode='lines',
            name='Daily Reading',
            line=dict(color='#3b82f6', width=1)
        ))
        fig.update_layout(
            plot_bgcolor='#1e293b',
            paper_bgcolor='#1e293b',
            font=dict(color='white'),
            xaxis=dict(title='Days', gridcolor='#334155'),
            yaxis=dict(title='kWh', gridcolor='#334155'),
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Result
    if analyze_btn:
        with st.spinner("Analyzing customer data..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "customer_id": customer_id,
                        "readings": readings
                    }
                )
                result = response.json()
                
                st.markdown("---")
                st.subheader("Analysis Result")
                
                # Result display
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    if "THEFT" in result['prediction']:
                        st.error(f"🚨 {result['prediction']}")
                    else:
                        st.success(f"✅ {result['prediction']}")
                
                with col4:
                    st.metric(
                        "Risk Score",
                        f"{result['risk_score']}%"
                    )
                
                with col5:
                    st.metric(
                        "Risk Level",
                        result['risk_level']
                    )
                
                # Risk gauge
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['risk_score'],
                    title={'text': "Theft Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ef4444"},
                        'steps': [
                            {'range': [0,40],  'color': "#14532d"},
                            {'range': [40,60], 'color': "#854d0e"},
                            {'range': [60,80], 'color': "#7c2d12"},
                            {'range': [80,100],'color': "#450a0a"},
                        ],
                    }
                ))
                fig2.update_layout(
                    paper_bgcolor='#1e293b',
                    font=dict(color='white'),
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.error(f"API Error: {e}")

# ── Tab 2: Batch Analysis ─────────────────────────────────────────
with tab2:
    st.subheader("Batch Customer Analysis")
    st.info("Analyze multiple customers at once")
    
    n_customers = st.slider("Number of customers to analyze", 5, 20, 10)
    
    if st.button("🔄 Run Batch Analysis", use_container_width=True):
        results = []
        progress = st.progress(0)
        
        for i in range(n_customers):
            # Mix of normal and theft patterns
            is_theft = np.random.random() < 0.3
            
            if is_theft:
                r = list(np.random.uniform(20, 45, 365))
                r[100:150] = [0]*50
            else:
                r = list(
                    8 + 3*np.sin(np.linspace(0,6*np.pi,365)) +
                    np.random.normal(0, 0.5, 365)
                )
            r = [max(0, round(x, 2)) for x in r]
            
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"customer_id": f"CUST_{i+1:03d}", "readings": r}
                ).json()
                results.append(resp)
            except:
                pass
            
            progress.progress((i+1)/n_customers)
        
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            
            # Summary
            theft_count  = (df['prediction'] == 'THEFT SUSPECTED').sum()
            normal_count = (df['prediction'] == 'NORMAL').sum()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Analyzed", len(results))
            c2.metric("Theft Suspected", theft_count)
            c3.metric("Normal", normal_count)
            
            # Table
            st.dataframe(
                df[['customer_id','prediction','risk_score','risk_level']],
                use_container_width=True
            )

# ── Tab 3: Live Simulation ────────────────────────────────────────
with tab3:
    st.subheader("Real-time Consumption Simulation")
    st.info("Watch live electricity consumption patterns")
    
    sim_type = st.radio(
        "Simulation Type",
        ["Normal Customer", "Theft Customer"],
        horizontal=True
    )
    
    if st.button("▶ Start Simulation", use_container_width=True):
        placeholder = st.empty()
        chart_data  = []
        
        for i in range(50):
            if sim_type == "Normal Customer":
                new_val = 8 + 3*np.sin(i/5) + np.random.normal(0, 0.5)
                new_val = max(0, new_val)
                color   = '#22c55e'
            else:
                if i < 20:
                    new_val = np.random.uniform(25, 45)
                else:
                    new_val = np.random.uniform(0, 1)
                color = '#ef4444'
            
            chart_data.append(round(new_val, 2))
            
            with placeholder.container():
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    y=chart_data,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=4)
                ))
                fig3.update_layout(
                    title=f"Live Reading: {chart_data[-1]} kWh",
                    plot_bgcolor='#1e293b',
                    paper_bgcolor='#1e293b',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#334155'),
                    yaxis=dict(
                        title='kWh',
                        gridcolor='#334155'
                    ),
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.metric("Current Reading", f"{chart_data[-1]} kWh")
            
            time.sleep(0.15)
        
        st.success("✅ Simulation complete!")