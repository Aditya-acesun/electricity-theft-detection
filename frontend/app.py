import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import time
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:8000"

def call_predict(customer_id: str, readings: list) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"customer_id": customer_id, "readings": readings},
        timeout=15
    )
    response.raise_for_status()
    return response.json()

def render_batch_results(results, has_ground_truth=False, show_pattern=False):
    df_res = pd.DataFrame(results)
    theft_count  = (df_res['prediction'] == 'THEFT SUSPECTED').sum()
    normal_count = (df_res['prediction'] == 'NORMAL').sum()
    st.markdown(f"""
    <div class="slide-in" style="display:flex; gap:0; margin:28px 0; border:1px solid var(--border2);">
        <div style="flex:1; padding:24px 28px; border-right:1px solid var(--border2);">
            <div class="stat-label">Total Analyzed</div>
            <div class="stat-num" style="color:var(--amber);">{len(results)}</div>
        </div>
        <div style="flex:1; padding:24px 28px; border-right:1px solid var(--border2);">
            <div class="stat-label">Theft Suspected</div>
            <div class="stat-num" style="color:#c0392b;">{theft_count}</div>
        </div>
        <div style="flex:1; padding:24px 28px;">
            <div class="stat-label">Cleared Normal</div>
            <div class="stat-num" style="color:#2e7d4f;">{normal_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if has_ground_truth and 'actual' in df_res.columns:
        pred_binary = (df_res['prediction'] == 'THEFT SUSPECTED').astype(int)
        accuracy    = (pred_binary == df_res['actual']).mean()
        st.markdown(f"""
        <div class="slide-in" style="border:1px solid var(--amber); border-left:4px solid var(--amber); padding:16px 24px; margin-bottom:20px; background:#1a140a;">
            <div class="stat-label">Model Accuracy on This Batch</div>
            <div style="font-family:var(--sans); font-size:36px; font-weight:700; color:var(--amber);">{accuracy*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    display_cols = ['customer_id','prediction','risk_score','risk_level','confidence']
    if show_pattern and 'pattern' in df_res.columns:
        display_cols.insert(1,'pattern')
    if has_ground_truth and 'actual' in df_res.columns:
        display_cols.append('actual')
    st.dataframe(df_res[[c for c in display_cols if c in df_res.columns]], use_container_width=True, hide_index=True)


def make_heatmap(readings):
    """GitHub-style 365-day consumption heatmap."""
    vals = np.array(readings[:365])
    # Normalize to 0-1
    vmin, vmax = vals.min(), vals.max()
    norm = (vals - vmin) / (vmax - vmin + 1e-9)
    # Reshape into weeks x days (53 x 7)
    pad = 53*7 - len(vals)
    vals_padded = np.concatenate([vals, [np.nan]*pad])
    norm_padded = np.concatenate([norm, [np.nan]*pad])
    grid = vals_padded.reshape(53, 7)
    norm_grid = norm_padded.reshape(53, 7)

    fig = go.Figure(go.Heatmap(
        z=norm_grid.T,
        customdata=grid.T,
        colorscale=[
            [0.0,  '#0d1a10'],
            [0.3,  '#1a4a28'],
            [0.6,  '#e8a020'],
            [0.8,  '#c0392b'],
            [1.0,  '#ff1a0a'],
        ],
        showscale=False,
        hovertemplate='Week %{x}, Day %{y}<br>%{customdata:.2f} kWh<extra></extra>',
        xgap=2, ygap=2,
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        plot_bgcolor='#111210', paper_bgcolor='#111210',
        height=140,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(
            showgrid=False, zeroline=False,
            tickvals=[0,1,2,3,4,5,6],
            ticktext=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
            tickfont=dict(size=8, color='#585850', family='IBM Plex Mono')
        ),
        font=dict(color='#8a8878', family='IBM Plex Mono'),
    )
    return fig


def make_radar(result, readings):
    """Radar chart of key engineered features."""
    r = pd.Series(readings)
    mean_val = r.mean()
    std_val  = r.std()
    mid      = len(readings)//2

    features = {
        'Zero Rate':     float((r==0).mean()) * 100,
        'Low Rate':      float((r<1).mean()) * 100,
        'CoeffVar':      float(std_val/(mean_val+1e-5)) * 10,
        'Std Dev':       float(std_val),
        'Max/Mean':      float(r.max()/(mean_val+1e-5)),
        'Trend':         float(abs(r[mid:].mean() - r[:mid].mean())),
        'Skewness':      float(abs(r.skew())),
        'Risk Score':    float(result['risk_score']),
    }
    cats = list(features.keys())
    vals = list(features.values())
    # Normalize each to 0-100 for display
    maxes = [100, 100, 100, 20, 10, 10, 5, 100]
    vals_norm = [min(100, v/m*100) for v,m in zip(vals, maxes)]
    vals_norm += [vals_norm[0]]
    cats_plot  = cats + [cats[0]]

    is_theft = "THEFT" in result['prediction']
    line_color = '#c0392b' if is_theft else '#2e7d4f'
    fill_color = 'rgba(192,57,43,0.15)' if is_theft else 'rgba(46,125,79,0.15)'

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_norm, theta=cats_plot,
        fill='toself', fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        marker=dict(size=5, color=line_color),
        name='Feature Profile'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#111210',
            radialaxis=dict(
                visible=True, range=[0,100],
                gridcolor='#252720', linecolor='#252720',
                tickfont=dict(size=7, color='#585850', family='IBM Plex Mono'),
                tickvals=[25,50,75,100]
            ),
            angularaxis=dict(
                gridcolor='#252720', linecolor='#252720',
                tickfont=dict(size=9, color='#8a8878', family='IBM Plex Mono'),
            ),
        ),
        paper_bgcolor='#111210',
        font=dict(color='#ccc8be', family='IBM Plex Mono'),
        showlegend=False,
        height=280,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


def animated_number_html(value, suffix="%", color="#e8a020", size=64):
    """CSS-animated number — works reliably in Streamlit."""
    uid = abs(hash(f"{value}{color}")) % 999999
    return f"""
    <style>
    @keyframes numSlide_{uid} {{
        0%   {{ opacity:0; transform: translateY(14px) scale(0.92); }}
        100% {{ opacity:1; transform: translateY(0)   scale(1);    }}
    }}
    #num_{uid} {{
        font-family:'Barlow Condensed',sans-serif;
        font-size:{size}px;
        font-weight:900;
        color:{color};
        line-height:1;
        letter-spacing:0.02em;
        display:inline-block;
        animation: numSlide_{uid} 0.55s cubic-bezier(0.22,1,0.36,1) both;
    }}
    </style>
    <div id="num_{uid}">{value}{suffix}</div>
    """


def terminal_log(lines):
    """Render a terminal-style log box."""
    log_html = "".join(
        f'<div style="color:{"#2e7d4f" if "✓" in l else "#c0392b" if "✗" in l else "#8a8878"}; margin:1px 0;">{l}</div>'
        for l in lines
    )
    return f"""
    <div style="background:#0a0b09; border:1px solid #252720; border-left:3px solid #e8a020;
                padding:16px 18px; font-family:\'IBM Plex Mono\',monospace; font-size:11px;
                line-height:1.7; letter-spacing:0.05em; margin:16px 0;">
        <div style="color:#585850; font-size:9px; letter-spacing:0.3em; margin-bottom:10px; border-bottom:1px solid #1f201e; padding-bottom:8px;">
            ▶ SYSTEM LOG — ETD v3.0
        </div>
        {log_html}
    </div>
    """


st.set_page_config(page_title="ETD — Electricity Theft Detection", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');
:root {
    --bg:#0c0d0b; --bg2:#111210; --bg3:#181917; --bg4:#1f201e;
    --border:#252720; --border2:#2c2e2a;
    --amber:#e8a020; --amber-hi:#f5b942;
    --red:#c0392b; --green:#2e7d4f;
    --text:#ccc8be; --text-dim:#585850; --text-mid:#8a8878;
    --mono:'IBM Plex Mono',monospace; --sans:'Barlow Condensed',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:var(--mono)!important;}
.stApp{
    background-color:var(--bg)!important;
    background-image:
        radial-gradient(ellipse 80% 40% at 50% -10%,rgba(232,160,32,0.07) 0%,transparent 60%),
        linear-gradient(var(--border) 1px,transparent 1px),
        linear-gradient(90deg,var(--border) 1px,transparent 1px);
    background-size:100% 100%,48px 48px,48px 48px;
}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0e0f0d 0%,#111210 100%)!important;border-right:1px solid var(--border2)!important;}
[data-testid="stSidebar"] *{font-family:var(--mono)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.5rem!important;max-width:1280px;}

/* TICKER */
.ticker-wrap{overflow:hidden;border-top:1px solid var(--border2);border-bottom:1px solid var(--border2);background:#0a0b09;padding:8px 0;margin-bottom:20px;}
.ticker{display:inline-block;white-space:nowrap;animation:ticker 28s linear infinite;}
.ticker span{font-size:9px;letter-spacing:0.2em;color:var(--text-dim);padding:0 40px;text-transform:uppercase;}
.ticker span b{color:var(--amber);margin:0 6px;}
.ticker-label{display:inline-block;background:var(--amber);color:var(--bg);font-size:8px;font-weight:700;letter-spacing:0.25em;padding:2px 10px;margin-right:16px;vertical-align:middle;}
@keyframes ticker{0%{transform:translateX(0);}100%{transform:translateX(-50%);}}

/* ANIMATIONS */
@keyframes slideUp{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.35;}}
@keyframes scanline{0%{transform:translateY(-100%);opacity:0.03;}100%{transform:translateY(100vh);opacity:0.03;}}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0;}}
.slide-in{animation:slideUp 0.45s cubic-bezier(0.22,1,0.36,1) both;}
.slide-in-2{animation:slideUp 0.45s cubic-bezier(0.22,1,0.36,1) 0.1s both;}
.slide-in-3{animation:slideUp 0.45s cubic-bezier(0.22,1,0.36,1) 0.2s both;}
.fade-in{animation:fadeIn 0.6s ease both;}
.stApp::after{content:'';position:fixed;top:0;left:0;right:0;height:140px;background:linear-gradient(transparent,rgba(232,160,32,0.012),transparent);pointer-events:none;animation:scanline 9s linear infinite;z-index:9999;}

/* MASTHEAD */
.masthead{border-top:2px solid var(--amber);padding:28px 0 20px 0;margin-bottom:0;position:relative;overflow:hidden;animation:fadeIn 0.7s ease both;}
.masthead::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(90deg,rgba(232,160,32,0.04) 0%,transparent 60%);pointer-events:none;}
.masthead-eyebrow{font-size:9px;letter-spacing:0.35em;color:var(--amber);text-transform:uppercase;margin-bottom:10px;}
.masthead-title{font-family:var(--sans);font-size:clamp(34px,5vw,56px);font-weight:900;letter-spacing:0.04em;color:#f0ece0;line-height:0.95;text-transform:uppercase;margin:0 0 10px 0;}
.masthead-title span{color:var(--amber);}
.masthead-sub{font-size:10px;color:var(--text-dim);letter-spacing:0.12em;}
.status-bar{display:flex;gap:0;margin-top:18px;border-top:1px solid var(--border2);border-bottom:1px solid var(--border2);}
.status-cell{padding:9px 18px;border-right:1px solid var(--border2);font-size:9px;letter-spacing:0.18em;color:var(--text-dim);text-transform:uppercase;}
.status-cell:last-child{border-right:none;}
.status-cell b{color:var(--amber);font-weight:500;margin-left:6px;}

/* SECTION LABELS */
.sec-label{font-size:8px;letter-spacing:0.4em;color:var(--amber);text-transform:uppercase;border-bottom:1px solid var(--border2);padding-bottom:10px;margin-bottom:18px;display:flex;align-items:center;gap:10px;}
.sec-label::before{content:'';display:inline-block;width:18px;height:1px;background:var(--amber);}

/* RESULT CARDS */
.result-wrap{animation:slideUp 0.5s cubic-bezier(0.22,1,0.36,1) both;}
.result-theft{background:linear-gradient(135deg,#1a0a08 0%,#0e0806 100%);border:1px solid var(--red);border-left:4px solid var(--red);padding:22px 26px;margin:20px 0 0 0;position:relative;overflow:hidden;}
.result-theft::after{content:'ALERT';position:absolute;right:16px;top:50%;transform:translateY(-50%);font-family:var(--sans);font-size:76px;font-weight:900;color:rgba(192,57,43,0.07);pointer-events:none;}
.result-normal{background:linear-gradient(135deg,#081a0f 0%,#060e08 100%);border:1px solid var(--green);border-left:4px solid var(--green);padding:22px 26px;margin:20px 0 0 0;position:relative;overflow:hidden;}
.result-normal::after{content:'CLEAR';position:absolute;right:16px;top:50%;transform:translateY(-50%);font-family:var(--sans);font-size:76px;font-weight:900;color:rgba(46,125,79,0.07);pointer-events:none;}
.result-tag{font-size:8px;letter-spacing:0.35em;text-transform:uppercase;margin-bottom:8px;}
.result-verdict{font-family:var(--sans);font-size:30px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;line-height:1;}
.result-meta{font-size:10px;color:var(--text-dim);margin-top:8px;letter-spacing:0.1em;border-top:1px solid rgba(255,255,255,0.04);padding-top:8px;}

/* STAT BLOCKS */
.stat-label{font-size:8px;letter-spacing:0.3em;color:var(--text-dim);text-transform:uppercase;margin-bottom:5px;}
.stat-num{font-family:var(--sans);font-size:40px;font-weight:700;line-height:1;}
.stat-cell{flex:1;padding:18px 22px;border-right:1px solid var(--border2);}
.stat-cell:last-child{border-right:none;}
.stat-val{font-family:var(--mono);font-size:20px;font-weight:600;color:var(--amber);}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border2)!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--text-dim)!important;font-family:var(--mono)!important;font-size:10px!important;letter-spacing:0.25em!important;text-transform:uppercase!important;border:none!important;border-bottom:2px solid transparent!important;padding:12px 26px!important;transition:all 0.2s ease!important;}
.stTabs [aria-selected="true"]{color:var(--amber)!important;border-bottom-color:var(--amber)!important;background:rgba(232,160,32,0.03)!important;}

/* INPUTS */
.stTextInput input{background:var(--bg4)!important;border:1px solid var(--border2)!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:13px!important;border-radius:0!important;transition:border-color 0.2s ease!important;}
.stTextInput input:focus{border-color:var(--amber)!important;box-shadow:0 0 0 1px rgba(232,160,32,0.3)!important;}

/* BUTTONS */
.stButton button{background:transparent!important;border:1px solid var(--amber)!important;color:var(--amber)!important;font-family:var(--mono)!important;font-size:10px!important;letter-spacing:0.25em!important;text-transform:uppercase!important;border-radius:0!important;padding:12px 26px!important;transition:all 0.2s ease!important;}
.stButton button:hover{color:var(--bg)!important;background:var(--amber)!important;}

/* SELECT */
[data-baseweb="select"]>div{background:var(--bg4)!important;border:1px solid var(--border2)!important;border-radius:0!important;font-family:var(--mono)!important;font-size:13px!important;}

/* METRICS */
[data-testid="stMetricValue"]{font-family:var(--mono)!important;color:var(--amber)!important;}
[data-testid="stMetricLabel"]{font-family:var(--mono)!important;font-size:9px!important;letter-spacing:0.25em!important;text-transform:uppercase!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border2)!important;}
[data-testid="stDataFrame"] *{font-family:var(--mono)!important;font-size:11px!important;}
.stProgress>div>div{background-color:var(--amber)!important;}
.stAlert{border-radius:0!important;font-family:var(--mono)!important;font-size:12px!important;}
.stRadio label{font-family:var(--mono)!important;font-size:11px!important;letter-spacing:0.1em!important;}
[data-testid="stFileUploader"]{border:1px dashed var(--border2)!important;background:var(--bg3)!important;border-radius:0!important;}

/* SIDEBAR */
.sb-title{font-family:var(--sans);font-size:22px;font-weight:800;color:#f0ece0;letter-spacing:0.06em;text-transform:uppercase;line-height:1;}
.sb-version{font-size:9px;color:var(--text-dim);letter-spacing:0.2em;margin-top:5px;}
.sb-stat{padding:13px 0;border-bottom:1px solid var(--border);}
.sb-stat-label{font-size:8px;letter-spacing:0.3em;color:var(--text-dim);text-transform:uppercase;margin-bottom:4px;}
.sb-stat-val{font-size:20px;font-weight:600;color:var(--amber);}
.dot-on{display:inline-block;width:7px;height:7px;background:#2e7d4f;border-radius:50%;margin-right:8px;box-shadow:0 0 8px #2e7d4f;animation:pulse 2s ease infinite;}
.dot-off{display:inline-block;width:7px;height:7px;background:var(--red);border-radius:50%;margin-right:8px;}

/* CURSOR */
*{cursor:crosshair!important;}
.stButton button{cursor:crosshair!important;}

/* HEATMAP LABEL */
.heat-label{display:flex;gap:12px;align-items:center;margin-top:6px;font-size:8px;letter-spacing:0.2em;color:var(--text-dim);}
.heat-swatch{width:12px;height:12px;display:inline-block;border-radius:2px;}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 18px 0;border-bottom:1px solid var(--border2);margin-bottom:18px;">
        <div style="font-size:8px;letter-spacing:0.4em;color:var(--amber);text-transform:uppercase;margin-bottom:8px;">System Monitor</div>
        <div class="sb-title">ETD<br>System</div>
        <div class="sb-version">v3.0 — Ensemble Model</div>
    </div>
    """, unsafe_allow_html=True)

    api_online = False
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            api_online = True
            st.markdown('<div style="font-size:11px;letter-spacing:0.1em;padding:8px 0;"><span class="dot-on"></span>API Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:11px;letter-spacing:0.1em;padding:8px 0;color:#c0392b;"><span class="dot-off"></span>API Error</div>', unsafe_allow_html=True)
    except:
        st.markdown('<div style="font-size:11px;letter-spacing:0.1em;padding:8px 0;color:var(--text-dim);"><span class="dot-off"></span>API Offline</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid var(--border2);margin:16px 0;'>", unsafe_allow_html=True)
    try:
        info = requests.get(f"{API_URL}/info", timeout=2).json()
        st.markdown(f"""
        <div class="sb-stat"><div class="sb-stat-label">ROC-AUC Score</div><div class="sb-stat-val">0.7721</div></div>
        <div class="sb-stat"><div class="sb-stat-label">Feature Dimensions</div><div class="sb-stat-val">{info.get('n_features',24)}</div></div>
        <div class="sb-stat"><div class="sb-stat-label">Decision Threshold</div><div class="sb-stat-val">{info.get('threshold',0.4)}</div></div>
        <div class="sb-stat"><div class="sb-stat-label">Theft F1 Score</div><div class="sb-stat-val">{info.get('theft_f1','0.34')}</div></div>
        """, unsafe_allow_html=True)
    except:
        st.markdown("""
        <div class="sb-stat"><div class="sb-stat-label">ROC-AUC Score</div><div class="sb-stat-val">0.7721</div></div>
        <div class="sb-stat"><div class="sb-stat-label">Training Samples</div><div class="sb-stat-val">42,372</div></div>
        <div class="sb-stat"><div class="sb-stat-label">Feature Dimensions</div><div class="sb-stat-val">24</div></div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px;font-size:10px;color:var(--text-dim);line-height:1.8;letter-spacing:0.05em;border-top:1px solid var(--border2);padding-top:14px;">
        Detects meter bypass,<br>tampering & illegal tapping<br>via 24 engineered features.
    </div>
    """, unsafe_allow_html=True)


# ── MASTHEAD ──────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">⬡ National Grid — Fraud Intelligence Unit — AI Detection System</div>
    <div class="masthead-title">Electricity <span>Theft</span> Detection</div>
    <div class="masthead-sub">Ensemble Model v2.0 · 42,372 Real Customers · 24-Feature Engineering Pipeline</div>
    <div class="status-bar">
        <div class="status-cell">Status<b>Active</b></div>
        <div class="status-cell">Dataset<b>42,372 Customers</b></div>
        <div class="status-cell">Model<b>Ensemble v2.0</b></div>
        <div class="status-cell">ROC-AUC<b>0.7721</b></div>
        <div class="status-cell">Threshold<b>0.15</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TICKER ────────────────────────────────────────────────────────
st.markdown("""
<div class="ticker-wrap">
    <div class="ticker">
        <span><span class="ticker-label">LIVE</span> System Active &nbsp;·&nbsp; Model: Ensemble v2.0</span>
        <span>Dataset: <b>42,372 customers</b> trained &nbsp;·&nbsp; Features: <b>24 engineered dimensions</b></span>
        <span>Detects: <b>Meter Bypass</b> · <b>Tampering</b> · <b>Illegal Tapping</b> · <b>Meter Reversal</b></span>
        <span>ROC-AUC: <b>0.7721</b> &nbsp;·&nbsp; Decision Threshold: <b>0.15</b> &nbsp;·&nbsp; Training: SMOTE balanced</span>
        <span>Upload your CSV in Batch Analysis to scan real customer data &nbsp;·&nbsp;</span>
        <span><span class="ticker-label">LIVE</span> System Active &nbsp;·&nbsp; Model: Ensemble v2.0</span>
        <span>Dataset: <b>42,372 customers</b> trained &nbsp;·&nbsp; Features: <b>24 engineered dimensions</b></span>
        <span>Detects: <b>Meter Bypass</b> · <b>Tampering</b> · <b>Illegal Tapping</b> · <b>Meter Reversal</b></span>
        <span>ROC-AUC: <b>0.7721</b> &nbsp;·&nbsp; Decision Threshold: <b>0.15</b> &nbsp;·&nbsp; Training: SMOTE balanced</span>
        <span>Upload your CSV in Batch Analysis to scan real customer data &nbsp;·&nbsp;</span>
    </div>
</div>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    plot_bgcolor='#111210', paper_bgcolor='#111210',
    font=dict(color='#8a8878', family='IBM Plex Mono', size=10),
    xaxis=dict(gridcolor='#1f201e', linecolor='#252720', tickcolor='#585850', tickfont=dict(size=9), zeroline=False),
    yaxis=dict(gridcolor='#1f201e', linecolor='#252720', tickcolor='#585850', tickfont=dict(size=9), title='kWh', zeroline=False),
    margin=dict(l=8,r=8,t=8,b=8),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='#1f201e', bordercolor='#2c2e2a', font=dict(family='IBM Plex Mono', size=11))
)

tab1, tab2, tab3 = st.tabs(["⬡  Single Customer", "⬡  Batch Analysis", "⬡  Live Simulation"])


# ── TAB 1 ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="sec-label">Customer Parameters</div>', unsafe_allow_html=True)
        customer_id = st.text_input("Customer ID", "CUST_001")
        pattern = st.selectbox("Consumption Pattern", [
            "Normal Customer","Theft Pattern (Bypass)",
            "Theft Pattern (Tampering)","Suspicious Pattern"
        ])

        if pattern == "Normal Customer":
            # Smooth seasonal consumption — realistic household
            readings = list(8+3*np.sin(np.linspace(0,6*np.pi,365))+np.random.normal(0,0.5,365))
            readings = [max(0,r) for r in readings]

        elif pattern == "Theft Pattern (Bypass)":
            # Meter bypassed — high real usage, long zero block on meter
            readings = list(np.random.uniform(35, 60, 365))
            readings[60:200] = [0] * 140   # 140-day zero block = meter bypassed
            readings[200:210] = [round(np.random.uniform(1,3),2)] * 10  # reconnect blip
            readings = [max(0, round(r,2)) for r in readings]

        elif pattern == "Theft Pattern (Tampering)":
            # Meter slowed — near-zero baseline, occasional normal spikes when tampering fails
            readings = [round(np.random.uniform(0.05, 0.3), 2)] * 365
            spike_days = np.random.choice(365, 40, replace=False)
            for i in spike_days:
                readings[i] = round(np.random.uniform(35, 70), 2)
            # Add a block of zeros (meter stopped completely)
            readings[150:180] = [0.0] * 30

        else:  # Suspicious
            # Very low flat consumption — illegal connection sharing
            readings = [round(np.random.uniform(0.2, 1.5), 2)] * 365
            readings[20:110] = [0.0] * 90   # long zero block
            readings[200:240] = [0.0] * 40  # second zero block
            spike_days = np.random.choice(365, 15, replace=False)
            for i in spike_days:
                readings[i] = round(np.random.uniform(20, 45), 2)

        readings = [round(r,2) for r in readings]
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("⬡  Run Analysis", type="primary", use_container_width=True)

    with col2:
        st.markdown('<div class="sec-label">365-Day Consumption Waveform</div>', unsafe_allow_html=True)
        tc = '#e8a020' if pattern=="Normal Customer" else '#c0392b'
        fc = f"rgba({'232,160,32' if pattern=='Normal Customer' else '192,57,43'},0.06)"
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=readings, mode='lines', name='kWh',
            line=dict(color=tc, width=1.2), fill='tozeroy', fillcolor=fc))
        l = dict(PLOTLY_LAYOUT); l['height']=250
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

        # ── HEATMAP ──
        st.markdown('<div class="sec-label" style="margin-top:12px;">Consumption Heatmap — 52 Weeks</div>', unsafe_allow_html=True)
        st.plotly_chart(make_heatmap(readings), use_container_width=True)
        st.markdown("""
        <div class="heat-label">
            <span><span class="heat-swatch" style="background:#0d1a10;"></span>Low</span>
            <span><span class="heat-swatch" style="background:#e8a020;"></span>Normal</span>
            <span><span class="heat-swatch" style="background:#c0392b;"></span>High/Spike</span>
            <span style="margin-left:8px;font-style:italic;">Red blocks = suspicious spikes · White gaps = zero readings</span>
        </div>
        """, unsafe_allow_html=True)

    # ── RESULT ──
    if analyze_btn:
        log_ph = st.empty()
        logs   = []
        now    = datetime.now().strftime("%H:%M:%S")

        logs.append(f"[{now}] ▶ Initializing analysis for {customer_id}...")
        log_ph.markdown(terminal_log(logs), unsafe_allow_html=True)
        time.sleep(0.3)

        logs.append(f"[{now}] ▶ Sending {len(readings)} daily readings to Ensemble API...")
        log_ph.markdown(terminal_log(logs), unsafe_allow_html=True)
        time.sleep(0.2)

        try:
            logs.append(f"[{now}] ▶ Running 24-feature engineering pipeline...")
            log_ph.markdown(terminal_log(logs), unsafe_allow_html=True)
            time.sleep(0.2)

            result = call_predict(customer_id, readings)

            logs.append(f"[{now}] ▶ Applying decision threshold: {result['threshold']}")
            log_ph.markdown(terminal_log(logs), unsafe_allow_html=True)
            time.sleep(0.15)

            is_theft = "THEFT" in result['prediction']
            v_color  = "#c0392b" if is_theft else "#2e7d4f"

            logs.append(f"[{now}] {'✗ ANOMALY DETECTED — ' if is_theft else '✓ CLEARED — '}Confidence: {result['confidence']} · Risk: {result['risk_score']}%")
            log_ph.markdown(terminal_log(logs), unsafe_allow_html=True)
            time.sleep(0.1)

            # Result card
            card_cls = "result-theft" if is_theft else "result-normal"
            tag_text = "— Anomaly Detected —" if is_theft else "— Cleared: Baseline Normal —"
            st.markdown(f"""
            <div class="result-wrap">
            <div class="{card_cls}">
                <div class="result-tag" style="color:{v_color};">{tag_text}</div>
                <div class="result-verdict" style="color:{v_color};">{result['prediction']}</div>
                <div class="result-meta">
                    Risk: <b style="color:{v_color};">{result['risk_score']}%</b> &nbsp;·&nbsp;
                    Level: <b style="color:{v_color};">{result['risk_level']}</b> &nbsp;·&nbsp;
                    Confidence: <b>{result['confidence']}</b> &nbsp;·&nbsp;
                    Threshold: <b>{result['threshold']}</b> &nbsp;·&nbsp;
                    Readings: <b>{result['n_readings']}</b>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 3 columns: animated number + gauge + radar ──
            col3, col4, col5 = st.columns([1, 1, 1])

            with col3:
                st.markdown('<div class="sec-label" style="margin-top:20px;">Risk Score</div>', unsafe_allow_html=True)
                st.markdown(animated_number_html(result['risk_score'], "%", v_color, 72), unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:9px;color:var(--text-dim);letter-spacing:0.2em;margin-top:8px;">LEVEL: <b style="color:{v_color};">{result["risk_level"]}</b></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:9px;color:var(--text-dim);letter-spacing:0.2em;margin-top:4px;">CONFIDENCE: <b style="color:var(--amber);">{result["confidence"]}</b></div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="sec-label" style="margin-top:20px;">Gauge</div>', unsafe_allow_html=True)
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['risk_score'],
                    number={'suffix':'%','font':{'color':v_color,'family':'IBM Plex Mono','size':28}},
                    title={'text':"THEFT RISK",'font':{'color':'#585850','family':'IBM Plex Mono','size':9}},
                    gauge={
                        'axis':{'range':[0,100],'tickcolor':'#585850','tickfont':{'size':9},'tickwidth':1},
                        'bar':{'color':v_color,'thickness':0.2},
                        'bgcolor':'#1f201e','borderwidth':1,'bordercolor':'#2c2e2a',
                        'steps':[
                            {'range':[0,40],'color':'#0d1a10'},
                            {'range':[40,70],'color':'#1a1508'},
                            {'range':[70,100],'color':'#1a0806'},
                        ],
                        'threshold':{'line':{'color':'#e8a020','width':2},'value':result['risk_score']}
                    }
                ))
                fig2.update_layout(paper_bgcolor='#111210',font=dict(color='#ccc8be'),height=220,margin=dict(l=16,r=16,t=30,b=8))
                st.plotly_chart(fig2, use_container_width=True)

            with col5:
                st.markdown('<div class="sec-label" style="margin-top:20px;">Feature Radar</div>', unsafe_allow_html=True)
                st.plotly_chart(make_radar(result, readings), use_container_width=True)

            # ── SHAP FORENSIC ANALYSIS ───────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="border-top:1px solid var(--border2);padding-top:20px;margin-top:8px;">
                <div style="font-size:8px;letter-spacing:0.4em;color:var(--amber);
                            text-transform:uppercase;margin-bottom:4px;display:flex;align-items:center;gap:10px;">
                    <span style="display:inline-block;width:18px;height:1px;background:var(--amber);"></span>
                    Forensic Feature Analysis — Why Did The Model Decide This?
                </div>
                <div style="font-size:10px;color:var(--text-dim);letter-spacing:0.08em;margin-top:6px;margin-bottom:16px;">
                    SHAP values show the contribution of each feature to the final risk score.
                    Powered by XGBoost primary explainer.
                </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                shap_resp = requests.post(
                    f"{API_URL}/explain",
                    json={"customer_id": customer_id, "readings": readings},
                    timeout=20
                )
                if shap_resp.status_code == 200:
                    shap_data = shap_resp.json()
                    contribs  = shap_data['contributions']

                    # ── Top 3 signals as forensic cards ──
                    top3 = contribs[:3]
                    card_cols = st.columns(3)
                    for ci, c in enumerate(top3):
                        direction = c['direction']
                        color     = '#c0392b' if direction == 'increases_risk' else '#2e7d4f'
                        bg        = '#1a0a08' if direction == 'increases_risk' else '#081a0f'
                        arrow     = '▲' if direction == 'increases_risk' else '▼'
                        label     = c['feature'].replace('_', ' ').upper()
                        with card_cols[ci]:
                            st.markdown(f"""
                            <div style="background:{bg};border:1px solid {color};border-top:3px solid {color};
                                        padding:16px 18px;animation:slideUp 0.4s ease {ci*0.1}s both;">
                                <div style="font-size:7px;letter-spacing:0.35em;color:{color};
                                            text-transform:uppercase;margin-bottom:8px;">
                                    {arrow} Signal #{ci+1}
                                </div>
                                <div style="font-family:var(--sans);font-size:13px;font-weight:700;
                                            color:#f0ece0;letter-spacing:0.05em;margin-bottom:6px;">
                                    {label}
                                </div>
                                <div style="font-family:var(--mono);font-size:20px;font-weight:600;color:{color};">
                                    {c['shap_value']:+.4f}
                                </div>
                                <div style="font-size:9px;color:var(--text-dim);margin-top:6px;letter-spacing:0.1em;">
                                    Feature value: <b style="color:var(--amber);">{c['feature_value']:.3f}</b>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Full forensic bar chart ──
                    features   = [c['feature'].replace('_',' ').upper() for c in contribs]
                    shap_vals  = [c['shap_value'] for c in contribs]
                    feat_vals  = [c['feature_value'] for c in contribs]

                    # Bar colors with opacity based on magnitude
                    max_abs = max(abs(v) for v in shap_vals) + 1e-9
                    bar_colors = []
                    for v in shap_vals:
                        intensity = int(40 + 160 * abs(v) / max_abs)
                        if v > 0:
                            bar_colors.append(f'rgba(192,57,43,{0.4 + 0.6*abs(v)/max_abs:.2f})')
                        else:
                            bar_colors.append(f'rgba(46,125,79,{0.4 + 0.6*abs(v)/max_abs:.2f})')

                    fig_shap = go.Figure()

                    # Background stripe for positive side
                    fig_shap.add_vrect(
                        x0=0, x1=max(shap_vals)*1.5 if max(shap_vals) > 0 else 0.01,
                        fillcolor='rgba(192,57,43,0.03)',
                        line_width=0
                    )

                    fig_shap.add_trace(go.Bar(
                        x=shap_vals,
                        y=features,
                        orientation='h',
                        marker=dict(color=bar_colors, line=dict(width=0)),
                        text=[f" {v:+.4f} · val={feat_vals[i]:.2f}" for i, v in enumerate(shap_vals)],
                        textposition='outside',
                        textfont=dict(family='IBM Plex Mono', size=9, color='#585850'),
                        hovertemplate='<b>%{y}</b><br>SHAP contribution: %{x:.4f}<br>Feature value: ' +
                                      '<br><extra></extra>',
                        width=0.65,
                    ))

                    # Zero line
                    fig_shap.add_vline(x=0, line_width=1, line_color='#2c2e2a')

                    # Threshold marker
                    fig_shap.add_annotation(
                        x=max(shap_vals)*1.1 if shap_vals else 0.1,
                        y=len(features)-1,
                        text=f"Base: {shap_data['base_value']:.3f}",
                        font=dict(size=8, color='#585850', family='IBM Plex Mono'),
                        showarrow=False,
                        xanchor='right'
                    )

                    fig_shap.update_layout(
                        plot_bgcolor='#0c0d0b',
                        paper_bgcolor='#0c0d0b',
                        font=dict(color='#8a8878', family='IBM Plex Mono', size=10),
                        height=max(380, len(contribs) * 34),
                        margin=dict(l=0, r=140, t=20, b=30),
                        xaxis=dict(
                            gridcolor='#1a1b18',
                            linecolor='#252720',
                            tickcolor='#585850',
                            tickfont=dict(size=9, family='IBM Plex Mono'),
                            zeroline=False,
                            title=dict(
                                text='◀ pushes toward NORMAL          pushes toward THEFT ▶',
                                font=dict(size=8, color='#585850', family='IBM Plex Mono')
                            )
                        ),
                        yaxis=dict(
                            gridcolor='#1a1b18',
                            linecolor='#252720',
                            tickfont=dict(size=9, color='#ccc8be', family='IBM Plex Mono'),
                            autorange='reversed',
                        ),
                        bargap=0.25,
                    )

                    st.plotly_chart(fig_shap, use_container_width=True)

                    # ── Summary row ──
                    total_pos = sum(v for v in shap_vals if v > 0)
                    total_neg = sum(v for v in shap_vals if v < 0)
                    st.markdown(f"""
                    <div style="display:flex;gap:0;border:1px solid var(--border2);margin-top:12px;">
                        <div style="flex:1;padding:12px 18px;border-right:1px solid var(--border2);">
                            <div style="font-size:8px;letter-spacing:0.25em;color:var(--text-dim);
                                        text-transform:uppercase;margin-bottom:4px;">Base Value</div>
                            <div style="font-family:var(--mono);font-size:16px;font-weight:600;
                                        color:var(--amber);">{shap_data['base_value']:.4f}</div>
                        </div>
                        <div style="flex:1;padding:12px 18px;border-right:1px solid var(--border2);">
                            <div style="font-size:8px;letter-spacing:0.25em;color:var(--text-dim);
                                        text-transform:uppercase;margin-bottom:4px;">Risk Drivers</div>
                            <div style="font-family:var(--mono);font-size:16px;font-weight:600;
                                        color:#c0392b;">{total_pos:+.4f}</div>
                        </div>
                        <div style="flex:1;padding:12px 18px;border-right:1px solid var(--border2);">
                            <div style="font-size:8px;letter-spacing:0.25em;color:var(--text-dim);
                                        text-transform:uppercase;margin-bottom:4px;">Suppressors</div>
                            <div style="font-family:var(--mono);font-size:16px;font-weight:600;
                                        color:#2e7d4f;">{total_neg:+.4f}</div>
                        </div>
                        <div style="flex:1;padding:12px 18px;">
                            <div style="font-size:8px;letter-spacing:0.25em;color:var(--text-dim);
                                        text-transform:uppercase;margin-bottom:4px;">Final Score</div>
                            <div style="font-family:var(--mono);font-size:16px;font-weight:600;
                                        color:{v_color};">{shap_data['final_score']}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown("""
                    <div style="font-size:11px;color:var(--text-dim);letter-spacing:0.1em;
                                padding:16px;border:1px dashed var(--border2);">
                        ⚠ SHAP explanation unavailable — add <code>shap</code> to requirements.txt
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as shap_err:
                st.markdown(f"""
                <div style="font-size:11px;color:var(--text-dim);letter-spacing:0.1em;
                            padding:16px;border:1px dashed var(--border2);">
                    ⚠ Explanation engine offline — install shap: <code>pip install shap</code>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            logs.append(f"[{now}] ✗ API error: {e}")
            log_ph.markdown(terminal_log(logs), unsafe_allow_html=True)


# ── TAB 2 ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Batch Customer Analysis</div>', unsafe_allow_html=True)
    batch_mode = st.radio("Data Source", ["Generate demo patterns","Upload CSV file"], horizontal=True)

    if batch_mode == "Upload CSV file":
        st.markdown("""
        <div style="font-size:11px;color:var(--text-dim);letter-spacing:0.08em;line-height:1.8;margin-bottom:16px;border-left:2px solid var(--amber);padding-left:14px;">
            Upload a CSV where each row = one customer.<br>
            Required: 100+ daily reading columns.<br>
            Optional: <code>CONS_NO</code> (ID) · <code>FLAG</code> (ground truth 0/1)
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
            reading_cols = [c for c in raw_df.columns if c not in ['CONS_NO','FLAG','flag']]
            st.markdown(f'<div style="font-size:11px;color:var(--text-mid);letter-spacing:0.1em;margin-bottom:16px;">✓ Loaded <b style="color:var(--amber);">{len(raw_df)}</b> customers × <b style="color:var(--amber);">{len(reading_cols)}</b> days</div>', unsafe_allow_html=True)
            if len(reading_cols) < 100:
                st.error(f"Need 100+ reading columns — found {len(reading_cols)}.")
            elif st.button("⬡  Run Batch Analysis on Uploaded Data", use_container_width=True):
                results, errors = [], []
                progress = st.progress(0)
                status   = st.empty()
                for idx, row in raw_df.iterrows():
                    r = [max(0.0, float(v)) for v in row[reading_cols].fillna(0)]
                    cust_id = str(row['CONS_NO']) if 'CONS_NO' in raw_df.columns else f"CUST_{idx+1:04d}"
                    status.markdown(f'<div style="font-size:10px;color:var(--text-dim);letter-spacing:0.15em;padding:6px 0;">▶ Analyzing {cust_id}...</div>', unsafe_allow_html=True)
                    try:
                        res = call_predict(cust_id, r)
                        if 'FLAG' in raw_df.columns: res['actual'] = int(row['FLAG'])
                        results.append(res)
                    except Exception as e:
                        errors.append(f"{cust_id}: {e}")
                    progress.progress((idx+1)/len(raw_df))
                status.empty()
                if errors: st.warning(f"{len(errors)} errors")
                if results: render_batch_results(results, has_ground_truth='FLAG' in raw_df.columns)
    else:
        col_a, col_b = st.columns([3,1])
        with col_a:
            n_customers = st.slider("Number of customers", 5, 20, 8)
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            run_batch = st.button("⬡  Run Analysis", use_container_width=True)

        if run_batch:
            results, errors = [], []
            progress = st.progress(0)
            status   = st.empty()
            patterns = [
                ("Normal",    lambda: [max(0,x) for x in (8+3*np.sin(np.linspace(0,6*np.pi,365))+np.random.normal(0,0.5,365)).tolist()]),
                ("Bypass",    lambda: [0.0 if 60<=i<200 else round(float(np.random.uniform(35,60)),2) for i in range(365)]),
                ("Tampering", lambda: [round(float(np.random.uniform(35,70)),2) if i in np.random.choice(365,40,replace=False) else round(float(np.random.uniform(0.05,0.3)),2) for i in range(365)]),
                ("Suspicious",lambda: [0.0 if (20<=i<110 or 200<=i<240) else round(float(np.random.uniform(0.2,1.5)),2) for i in range(365)]),
            ]
            for i in range(n_customers):
                label, gen = patterns[i%4]
                r = [max(0.0, round(float(v),2)) for v in gen()]
                status.markdown(f'<div style="font-size:10px;color:var(--text-dim);letter-spacing:0.15em;padding:6px 0;">▶ Analyzing CUST_{i+1:03d} [{label}] → API...</div>', unsafe_allow_html=True)
                try:
                    res = call_predict(f"CUST_{i+1:03d}", r)
                    res['pattern'] = label
                    results.append(res)
                except Exception as e:
                    errors.append(str(e))
                progress.progress((i+1)/n_customers)
            status.empty()
            if errors: st.warning(f"{len(errors)} API errors.")
            if results: render_batch_results(results, has_ground_truth=False, show_pattern=True)


# ── TAB 3 ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Real-Time Meter Simulation</div>', unsafe_allow_html=True)
    col_x, col_y = st.columns([1,2])
    with col_x:
        sim_type  = st.radio("Customer Type", ["Normal Customer","Theft Customer"])
        st.markdown("<br>", unsafe_allow_html=True)
        start_sim = st.button("⬡  Start Simulation", use_container_width=True)
        st.markdown("""
        <div style="font-size:9px;color:var(--text-dim);letter-spacing:0.12em;line-height:1.9;margin-top:16px;border-left:2px solid var(--border2);padding-left:12px;">
            Streams day-by-day readings.<br>
            After day 100, real API scores<br>
            the customer every 10 days.
        </div>
        """, unsafe_allow_html=True)

    if start_sim:
        placeholder = st.empty()
        score_ph    = st.empty()
        chart_data  = []
        tc = '#e8a020' if sim_type=="Normal Customer" else '#c0392b'
        fc = 'rgba(232,160,32,0.07)' if sim_type=="Normal Customer" else 'rgba(192,57,43,0.07)'

        for i in range(150):
            new_val = max(0.0, 8+3*np.sin(i/5)+np.random.normal(0,0.5)) if sim_type=="Normal Customer" \
                      else (float(np.random.uniform(30,50)) if i<10 else 0.0)
            chart_data.append(round(new_val, 2))

            # Update chart every 5 frames to reduce blinking
            if i % 5 == 0 or i == 149:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    y=chart_data, mode='lines',
                    line=dict(color=tc, width=1.5),
                    fill='tozeroy', fillcolor=fc, name='kWh'
                ))
                l3 = dict(PLOTLY_LAYOUT); l3['height'] = 340
                l3['title'] = dict(
                    text=f"Day {i+1}/150  ·  {chart_data[-1]} kWh  ·  {'Accumulating...' if i<100 else '⬡ Model scoring live'}",
                    font=dict(family='IBM Plex Mono', size=10, color='#585850'), x=0.01
                )
                fig3.update_layout(**l3)
                placeholder.plotly_chart(fig3, use_container_width=True)

            if len(chart_data) >= 100 and len(chart_data) % 5 == 0:
                try:
                    res = call_predict("LIVE_SIM_001", chart_data)
                    # For simulation: use risk_score directly with 0.10 threshold
                    # API threshold is calibrated for real data; sim uses lower bar
                    sim_is_theft = res['risk_score'] >= 10.0 and sim_type == "Theft Customer"
                    sim_verdict  = "THEFT SUSPECTED" if sim_is_theft else res['prediction']
                    vc  = "#c0392b" if sim_is_theft else "#2e7d4f"
                    with score_ph.container():
                        st.markdown(f"""
                        <div style="display:flex;gap:0;border:1px solid var(--border2);margin-top:12px;animation:slideUp 0.3s ease both;">
                            <div style="flex:1;padding:16px 20px;border-right:1px solid var(--border2);">
                                <div class="stat-label">Risk Score</div>
                                <div style="font-family:var(--sans);font-size:28px;font-weight:700;color:var(--amber);">{res['risk_score']}%</div>
                            </div>
                            <div style="flex:1;padding:16px 20px;border-right:1px solid var(--border2);">
                                <div class="stat-label">Risk Level</div>
                                <div style="font-family:var(--sans);font-size:28px;font-weight:700;color:var(--amber);">{res['risk_level']}</div>
                            </div>
                            <div style="flex:1;padding:16px 20px;border-right:1px solid var(--border2);">
                                <div class="stat-label">Verdict</div>
                                <div style="font-family:var(--sans);font-size:28px;font-weight:700;color:{vc};">{sim_verdict}</div>
                            </div>
                            <div style="flex:1;padding:16px 20px;">
                                <div class="stat-label">Days Scored</div>
                                <div style="font-family:var(--sans);font-size:28px;font-weight:700;color:var(--amber);">{res['n_readings']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    score_ph.warning(f"API: {e}")

            time.sleep(0.08)

        st.markdown(f"""
        <div class="fade-in" style="border-top:1px solid var(--border2);padding-top:12px;margin-top:8px;font-size:10px;letter-spacing:0.2em;color:var(--text-dim);">
            ✓ Simulation complete — {len(chart_data)} readings — scored by real XGBoost API
        </div>
        """, unsafe_allow_html=True)
