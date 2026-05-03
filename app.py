import warnings
import logging
import os

# Suppress noisy terminal warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')
logging.getLogger('aif360').setLevel(logging.ERROR)

# Filter specific root logger warnings from aif360
class AIF360Filter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ('AdversarialDebiasing' in msg or 'SenSeI' in msg or 'sample_weight' in msg)

logging.getLogger().addFilter(AIF360Filter())

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO
import streamlit.components.v1 as components
st.set_page_config(page_title='Fairness Audit Pipeline | Compliance Platform', layout='wide', initial_sidebar_state='expanded')

# Theme Colors
PRIMARY = '#0F172A'
ACCENT = '#16A34A'
ACCENT_LIGHT = '#DCFCE7'
BG = '#F8FAFC'
CARD_BG = '#FFFFFF'
TEXT = '#111827'
TEXT_MUTED = '#64748B'
BORDER = '#E2E8F0'
RED = '#DC2626'
AMBER = '#F59E0B'
BLUE = '#3B82F6'

# Load CSS
with open('style.css', 'r') as f:
    CSS = f.read()

CSS = CSS.replace('BG_VAR', BG).replace('PRIMARY_VAR', PRIMARY).replace('ACCENT_VAR', ACCENT).replace('ACCENT_LIGHT_VAR', ACCENT_LIGHT).replace('CARD_BG_VAR', CARD_BG).replace('BORDER_VAR', BORDER).replace('TEXT_MUTED_VAR', TEXT_MUTED).replace('TEXT_VAR', TEXT).replace('RED_VAR', RED).replace('AMBER_VAR', AMBER).replace('BLUE_VAR', BLUE)

st.markdown(f'<style>{CSS}</style>', unsafe_allow_html=True)

# Static UI Content for IDE Stability
PIPELINE_STEPS = [('01', 'Data Ingestion', 'Upload and profile loan datasets'), ('02', 'Model Training', 'Train classification models with validation'), ('03', 'Bias Detection', 'Audit for demographic disparities'), ('04', 'Mitigation', 'Apply fairness-aware retraining'), ('05', 'Comparison', 'Before vs after performance analysis'), ('06', 'Explainability', 'SHAP-based decision interpretation'), ('07', 'Reporting', 'Generate compliance documentation')]
FRAMEWORKS = [('ECOA', 'Equal Credit Opportunity Act'), ('FHA', 'Fair Housing Act'), ('EEOC', 'Four-Fifths Rule (Disparate Impact)'), ('EU AI Act', 'High-Risk System Requirements'), ('SR 11-7', 'Federal Reserve Model Risk Guidance'), ('OCC 2011-12', 'Model Risk Management'), ('CFPB', 'Consumer Financial Protection Bureau')]

UI_CONTENT = {
    'pipeline': ''.join((f'<div class="list-item"><p class="list-item-title"><span class="list-item-number">{n}</span><b>{m}</b></p><p class="list-item-desc">{d}</p></div>' for n, m, d in PIPELINE_STEPS)),
    'frameworks': ''.join((f'<div class="list-item"><p class="list-item-title"><span class="list-item-abbr">{a}</span>{f}</p></div>' for a, f in FRAMEWORKS))
}
from utils.preprocessing import preprocess_data, get_data_profile
from utils.training import train_model, evaluate_model
from utils.bias_detection import detect_bias, classify_risk, detect_intersectional_bias
from utils.mitigation import mitigate_bias
from utils.explainability import compute_shap_values, get_feature_importance, get_native_feature_importance, generate_shap_summary_plot
from utils.counterfactuals import generate_counterfactuals, get_actionable_diff
from utils.reporting import generate_report, generate_pdf_report

# Force reload modules to bypass Streamlit's aggressive caching of background files
import importlib
import sys
for mod_name in ['utils.bias_detection', 'utils.preprocessing', 'utils.mitigation']:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

def render_kpi(label, value, color_class=''):
    st.markdown(f'<div class="kpi-card"><p class="kpi-label">{label}</p><p class="kpi-value {color_class}">{value}</p></div>', unsafe_allow_html=True)

def render_info(message):
    st.markdown(f'<div class="info-panel">{message}</div>', unsafe_allow_html=True)

def render_page_header(title, subtitle=''):
    st.markdown(f'<div class="page-header"><h1 class="page-title">{title}</h1><p class="page-subtitle">{subtitle}</p></div>', unsafe_allow_html=True)

def get_prediction_and_confidence(model, row):
    """Safely get prediction and confidence, handling non-probabilistic fairness models."""
    pred = model.predict(row)[0]
    conf = "N/A"
    
    # Try to get confidence from the model or its unwrapped base
    target_model = model
    if hasattr(model, 'model'): # My wrapper
        target_model = model.model
    elif hasattr(model, 'predictors_'): # Fairlearn meta-estimator
        target_model = model.predictors_[0]
        
    if hasattr(target_model, 'predict_proba'):
        try:
            probs = target_model.predict_proba(row)[0]
            conf = f"{probs[pred]:.1%}"
        except:
            pass
    return pred, conf

def render_section(title, content_fn=None):
    container = st.container(border=True)
    with container:
        st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)
        if content_fn:
            content_fn()
    return container

def render_badge(text, variant='gray'):
    return f'<span class="badge badge-{variant}">{text}</span>'

def plotly_theme(fig, height=400):
    fig.update_layout(template='plotly_white', height=height, font=dict(family='Inter, sans-serif', size=13, color=TEXT), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=40, t=50, b=40), title_font=dict(size=15, color=PRIMARY, family='Inter, sans-serif'))
    fig.update_xaxes(gridcolor='#F1F5F9', linecolor=BORDER)
    fig.update_yaxes(gridcolor='#F1F5F9', linecolor=BORDER)
    return fig
STATE_KEYS = {'data': None, 'data_profile': None, 'model': None, 'metrics': None, 'bias_metrics': None, 'approval_rates': None, 'mitigated_model': None, 'mitigated_metrics': None, 'mitigated_bias_metrics': None, 'mitigated_approval_rates': None, 'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None, 'sensitive_col': None, 'model_type': None, 'sf_test': None, 'sf_train': None, 'mitigation_method': None, 'report_text': None, 'report_pdf': None, 'active_page': 'Overview', 'active_category': 'Standard Workflow'}
for key, default in STATE_KEYS.items():
    if key not in st.session_state:
        st.session_state[key] = default

def get_state(key, default=None):
    return st.session_state.get(key, default)

# Navigation Configuration
ORDERED_PAGES = [
    "Overview", "Data Management", "Model Training", "Bias Analysis", 
    "Intersectional Audit", "Mitigation Engine", "Performance Comparison",
    "Explainability", "Real-time Simulator", "What-If Analysis", "Compliance Reports"
]
PAGE_TO_CAT = {p: "Standard Workflow" for p in ORDERED_PAGES[:7]}
PAGE_TO_CAT.update({p: "Advanced Analysis Level" for p in ORDERED_PAGES[7:]})
with st.sidebar:
    st.markdown(f'<div class="sidebar-header"><h2>Fairness Audit Pipeline</h2></div>', unsafe_allow_html=True)
    
    # Category selection
    cat_index = ["Standard Workflow", "Advanced Analysis Level"].index(st.session_state.active_category)
    category = st.selectbox('Category', ["Standard Workflow", "Advanced Analysis Level"], index=cat_index, label_visibility='collapsed')
    
    if category != st.session_state.active_category:
        st.session_state.active_category = category
        st.session_state.active_page = "Overview" if category == "Standard Workflow" else "Explainability"
        st.rerun()

    st.markdown(f'<p class="status-title" style="margin-top: 10px;">{category.upper()}</p>', unsafe_allow_html=True)
    
    cat_pages = ["Overview", "Data Management", "Model Training", "Bias Analysis", "Intersectional Audit", "Mitigation Engine", "Performance Comparison"] if category == "Standard Workflow" else ["Explainability", "Real-time Simulator", "What-If Analysis", "Compliance Reports"]
    
    page_index = cat_pages.index(st.session_state.active_page) if st.session_state.active_page in cat_pages else 0
    page_selection = st.radio('Navigation', cat_pages, index=page_index, label_visibility='collapsed')
    
    if page_selection != st.session_state.active_page:
        st.session_state.active_page = page_selection
        st.rerun()

    st.markdown('<div style="height: 30px;"></div><p class="nav-category-header">Audit Progress</p>', unsafe_allow_html=True)
    steps = [
        ("Data", st.session_state.get('data') is not None),
        ("Model", st.session_state.get('model') is not None),
        ("Audit", st.session_state.get('bias_metrics') is not None),
        ("Fix", st.session_state.get('mitigated_model') is not None)
    ]
    
    progress_html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; margin-top: 8px;">'
    for label, completed in steps:
        icon = "✅" if completed else "⏳"
        color = "#4ADE80" if completed else "rgba(255,255,255,0.4)"
        progress_html += f'<div style="font-size: 0.7rem; color: {color}; display: flex; align-items: center; gap: 6px;">{icon} {label}</div>'
    progress_html += '</div>'
    st.markdown(progress_html, unsafe_allow_html=True)

    st.divider()

def page_overview():
    # Enterprise Hero Banner with live timestamp
    from datetime import datetime
    now = datetime.now().strftime('%d %b %Y  |  %H:%M:%S')
    st.markdown(f'''
        <div class="hero-banner" style="position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; right: 0; width: 300px; height: 300px; background: radial-gradient(circle, rgba(22,163,74,0.15) 0%, transparent 70%); border-radius: 50%;"></div>
            <p style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.15em; color: rgba(255,255,255,0.5); margin: 0 0 8px 0;">Enterprise Compliance Platform</p>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF !important;">Fairness Audit Platform</h1>
            <p style="margin: 10px 0 0 0; font-size: 1rem; opacity: 0.8; color: #FFFFFF !important;">Real-time bias detection, mitigation, and regulatory compliance for loan approval models.</p>
            <div style="margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; color: rgba(255,255,255,0.7);">{now}</span>
                <span style="background: rgba(22,163,74,0.2); border: 1px solid rgba(22,163,74,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; color: #4ADE80;">SYSTEM ONLINE</span>
                <span style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; color: rgba(255,255,255,0.7);">v2.0 Enterprise</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # KPI Section
    st.markdown('<p class="section-title">Key Performance Indicators</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.session_state.get('metrics'):
            val = f"{st.session_state.metrics['Accuracy']:.1%}"
            render_kpi('Model Accuracy', val, 'green')
        else:
            render_kpi('Model Accuracy', '--', '')
    with c2:
        if st.session_state.get('bias_metrics'):
            di = st.session_state.bias_metrics['Disparate Impact']
            _, color_hex = classify_risk(di)
            color_map = {ACCENT: 'green', RED: 'red', AMBER: 'amber'}
            render_kpi('DI Ratio (Fairness)', f'{di:.3f}', color_map.get(color_hex, ''))
        else:
            render_kpi('DI Ratio (Fairness)', '--', '')
    with c3:
        if st.session_state.get('bias_metrics'):
            dpd = st.session_state.bias_metrics['Demographic Parity Difference']
            render_kpi('Parity Difference', f'{abs(dpd):.3f}', 'blue')
        else:
            render_kpi('Parity Difference', '--', '')
    with c4:
        if st.session_state.get('mitigated_model'):
            render_kpi('Compliance Status', 'Compliant', 'green')
        elif st.session_state.get('bias_metrics'):
            di = st.session_state.bias_metrics['Disparate Impact']
            label, _ = classify_risk(di)
            clr = 'red' if 'High' in label else 'amber' if 'Moderate' in label else 'green'
            render_kpi('Compliance Status', label, clr)
        else:
            render_kpi('Compliance Status', 'Pending', '')
    
    st.markdown('<br>', unsafe_allow_html=True)
    
    # System Status & Quick Actions Row
    col_status, col_actions = st.columns([1.5, 1])
    
    with col_status:
        with st.container(border=True):
            st.markdown('<p class="section-title">System Status</p>', unsafe_allow_html=True)
            
            data_loaded = st.session_state.get('data') is not None
            model_trained = st.session_state.get('model') is not None
            bias_done = st.session_state.get('bias_metrics') is not None
            mitigated = st.session_state.get('mitigated_model') is not None
            
            total_steps = 4
            completed = sum([data_loaded, model_trained, bias_done, mitigated])
            pct = int((completed / total_steps) * 100)
            
            st.progress(completed / total_steps, text=f"Pipeline Completion: {pct}%")
            
            status_items = [
                ("Data Ingestion", data_loaded, f"{len(st.session_state.data):,} records" if data_loaded else "Awaiting upload"),
                ("Model Training", model_trained, f"{st.session_state.get('model_type', 'N/A')}" if model_trained else "Not started"),
                ("Bias Analysis", bias_done, f"DI: {st.session_state.bias_metrics['Disparate Impact']:.3f}" if bias_done else "Pending audit"),
                ("Mitigation", mitigated, f"{st.session_state.get('mitigation_method', 'Applied')}" if mitigated else "Not applied"),
            ]
            
            for label, done, detail in status_items:
                dot_color = ACCENT if done else TEXT_MUTED
                st.markdown(f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid {BORDER};"><div style="display: flex; align-items: center; gap: 10px;"><div style="width: 8px; height: 8px; border-radius: 50%; background: {dot_color};"></div><span style="font-size: 0.85rem; font-weight: 600; color: {TEXT};">{label}</span></div><span style="font-size: 0.75rem; color: {dot_color}; font-weight: 500;">{detail}</span></div>', unsafe_allow_html=True)
    
    with col_actions:
        with st.container(border=True):
            st.markdown('<p class="section-title">Quick Actions</p>', unsafe_allow_html=True)
            
            if st.button('Load Sample Data', width='stretch', key='qa_data'):
                st.session_state.active_page = 'Data Management'
                st.session_state.active_category = 'Standard Workflow'
                st.rerun()
            if st.button('Train AI Model', width='stretch', key='qa_train'):
                st.session_state.active_page = 'Model Training'
                st.session_state.active_category = 'Standard Workflow'
                st.rerun()
            if st.button('Run Bias Audit', width='stretch', key='qa_bias'):
                st.session_state.active_page = 'Bias Analysis'
                st.session_state.active_category = 'Standard Workflow'
                st.rerun()
            if st.button('Generate Report', width='stretch', key='qa_report'):
                st.session_state.active_page = 'Compliance Reports'
                st.session_state.active_category = 'Advanced Analysis Level'
                st.rerun()
        
        with st.container(border=True):
            st.markdown('<p class="section-title" style="margin-bottom: 12px;">Tech Stack</p>', unsafe_allow_html=True)
            tech_html = '''
                <div style="display: flex; flex-wrap: wrap; gap: 8px; justify-content: space-between;">
                    <div style="flex: 1 1 30%; background: #EFF6FF; color: #1D4ED8; padding: 6px 0; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-align: center;">Scikit-Learn</div>
                    <div style="flex: 1 1 30%; background: #F0FDF4; color: #15803D; padding: 6px 0; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-align: center;">Fairlearn</div>
                    <div style="flex: 1 1 30%; background: #FFF7ED; color: #C2410C; padding: 6px 0; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-align: center;">AIF360</div>
                    <div style="flex: 1 1 30%; background: #FAF5FF; color: #7E22CE; padding: 6px 0; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-align: center;">SHAP</div>
                    <div style="flex: 1 1 30%; background: #FEF2F2; color: #B91C1C; padding: 6px 0; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-align: center;">DiCE-ML</div>
                    <div style="flex: 1 1 30%; background: #ECFDF5; color: #047857; padding: 6px 0; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-align: center;">Plotly</div>
                </div>
            '''
            st.markdown(tech_html, unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Pipeline & Frameworks
    tab1, tab2 = st.tabs(['Pipeline Architecture', 'Regulatory Framework'])
    with tab1:
        st.markdown(f'<div class="list-container">{UI_CONTENT["pipeline"]}</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown(f'<div class="list-container">{UI_CONTENT["frameworks"]}</div>', unsafe_allow_html=True)

    # Compliance Gauge (only when data exists)
    if st.session_state.get('bias_metrics'):
        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Regulatory Compliance Gauge</p>', unsafe_allow_html=True)
            di = st.session_state.bias_metrics['Disparate Impact']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = di,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Disparate Impact Ratio", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': ACCENT},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.8], 'color': 'rgba(220, 38, 38, 0.1)'},
                        {'range': [0.8, 1], 'color': 'rgba(22, 163, 74, 0.1)'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8}}))
            fig.update_layout(height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig, width='stretch')


def page_data_management():
    render_page_header('Data Management', 'Upload, inspect, and profile loan application datasets')
    st.markdown('<p class="section-title">Offline Datasets</p>', unsafe_allow_html=True)
    col_upload, col_sample_sel, col_sample_btn = st.columns([2, 1.5, 1])
    with col_upload:
        uploaded_file = st.file_uploader('Upload Local Dataset (CSV)', type='csv', label_visibility='collapsed')
    with col_sample_sel:
        mock_datasets = {
            'Default Loan Data': 'data/loan_data.csv',
            'German Credit Risk (Synthetic)': 'data/german_credit_mock.csv',
            'Home Credit Risk (Synthetic)': 'data/home_credit_mock.csv'
        }
        selected_mock = st.selectbox('Select Offline Dataset', list(mock_datasets.keys()), label_visibility='collapsed')
    with col_sample_btn:
        load_sample = st.button('Load Offline Data', width='stretch')
        
    st.markdown('<br>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<p class="section-title">Kaggle Data Synchronization</p>', unsafe_allow_html=True)
        col_k_sel, col_k_btn = st.columns([3, 1])
        with col_k_sel:
            kaggle_datasets = {
                'Lending Club Loan Data': 'adarshsng/lending-club-loan-data-csv',
                'German Credit Risk (Fairness Benchmark)': 'uciml/german-credit',
                'Global Credit Risk Dataset': 'laotse/credit-risk-dataset'
            }
            selected_kaggle = st.selectbox('Select Real-Time Dataset', list(kaggle_datasets.keys()), label_visibility='collapsed')
        with col_k_btn:
            fetch_kaggle = st.button('Fetch Kaggle Data', width='stretch')

    if uploaded_file or load_sample or fetch_kaggle:
        try:
            if uploaded_file:
                st.session_state.data = pd.read_csv(uploaded_file)
            elif load_sample:
                import os
                if not os.path.exists(mock_datasets[selected_mock]):
                    st.error(f"Offline dataset {selected_mock} not found. Please ensure it was generated.")
                    return
                st.session_state.data = pd.read_csv(mock_datasets[selected_mock])
                st.info(f"Successfully loaded offline dataset: {selected_mock}")
            elif fetch_kaggle:
                dataset_id = kaggle_datasets[selected_kaggle]
                with st.spinner(f'Downloading {selected_kaggle} from Kaggle...'):
                    import kagglehub
                    import os
                    path = kagglehub.dataset_download(dataset_id)
                    files = [f for f in os.listdir(path) if f.endswith('.csv')]
                    if not files:
                        st.error('Kaggle download successful, but no CSV file was found in the archive.')
                        return
                    # Prioritize 'train' or 'credit' files if multiple exist
                    target_file = files[0]
                    for f in files:
                        if 'train' in f.lower() or 'credit' in f.lower():
                            target_file = f
                            break
                    CSV_PATH = os.path.join(path, target_file)
                    df = pd.read_csv(CSV_PATH, nrows=200000)
                    st.session_state.data = df
                    st.info(f'Synchronized top {len(df):,} records from {target_file} for memory stability.')
            st.session_state.data_profile = get_data_profile(st.session_state.data)
            st.success('Target environment synchronized with real-time data source.')
        except Exception as e:
            err_str = str(e).lower()
            if "unreachable host" in err_str or "max retries exceeded" in err_str or "connection" in err_str:
                st.error(
                    "**Network Connection Error:** Unable to reach Kaggle servers. \n\n"
                    "It seems your environment is offline or a firewall is blocking the connection. "
                    "Please use the **'Load Mock Data'** button or **'Upload Local Dataset'** options instead."
                )
            else:
                st.error(f'**Error loading data:** {e} \n\nPlease ensure your Kaggle credentials are set up correctly or use the offline data options.')
    if st.session_state.data is not None:
        df = st.session_state.data
        profile = st.session_state.data_profile or get_data_profile(df)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi('Total Records', f"{profile['row_count']:,}")
        with c2:
            render_kpi('Features', str(profile['col_count']))
        with c3:
            render_kpi('Missing Values', f"{profile['total_missing']:,}")
        with c4:
            completeness = (1 - profile['total_missing'] / (profile['row_count'] * profile['col_count'])) * 100
            render_kpi('Completeness', f'{completeness:.1f}%', 'green')
        
        st.markdown('<br>', unsafe_allow_html=True)
        col_preview, col_profile = st.columns([3, 2])
        with col_preview:
            with st.container(border=True):
                st.markdown('<p class="section-title">Data Preview</p>', unsafe_allow_html=True)
                st.dataframe(df.head(10), width='stretch', hide_index=True)
        with col_profile:
            with st.container(border=True):
                st.markdown('<p class="section-title">Column Profile</p>', unsafe_allow_html=True)
                type_data = []
                for col_name in df.columns:
                    dtype_str = str(df[col_name].dtype)
                    missing_count = profile['missing_counts'].get(col_name, 0)
                    missing_pct_val = profile['missing_pct'].get(col_name, 0.0)
                    type_data.append({'Column': col_name, 'Type': dtype_str, 'Missing': f'{missing_count} ({missing_pct_val:.1f}%)'})
                st.dataframe(pd.DataFrame(type_data), width='stretch', hide_index=True)
        st.markdown('<br>', unsafe_allow_html=True)
        col_corr, col_target = st.columns([2, 1])
        
        with col_corr:
            with st.container(border=True):
                st.markdown('<p class="section-title">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
                corr_df = pd.DataFrame(profile['correlations'])
                if not corr_df.empty:
                    fig_corr = go.Figure(data=go.Heatmap(z=corr_df.values, x=corr_df.columns, y=corr_df.index, colorscale='RdBu', zmin=-1, zmax=1))
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(plotly_theme(fig_corr, 400), width='stretch')
                else:
                    st.warning("Insufficient numeric data for correlation analysis.")

        with col_target:
            with st.container(border=True):
                st.markdown('<p class="section-title">Target Distribution</p>', unsafe_allow_html=True)
                target_col = 'loan_approval' # Default or detected
                if target_col in df.columns:
                    target_counts = df[target_col].value_counts()
                    fig_target = go.Figure(data=[go.Pie(labels=['Denied', 'Approved'], values=target_counts.values, hole=.4, marker_colors=[RED, ACCENT])])
                    st.plotly_chart(plotly_theme(fig_target, 400), width='stretch')
                else:
                    st.warning(f"Column '{target_col}' not found.")
        
        missing_cols = {k: v for k, v in profile['missing_counts'].items() if v > 0}
        if missing_cols:
            with st.container(border=True):
                st.markdown('<p class="section-title">Missing Values Distribution</p>', unsafe_allow_html=True)
                fig = go.Figure(go.Bar(x=list(missing_cols.keys()), y=list(missing_cols.values()), marker_color=ACCENT, text=list(missing_cols.values()), textposition='auto'))
                fig.update_layout(xaxis_title='Column', yaxis_title='Missing Count')
                st.plotly_chart(plotly_theme(fig, 350), width='stretch')

def page_model_training():
    render_page_header('Model Training', 'Configure and train a classification model for loan approvals')
    if st.session_state.data is None:
        render_info('Please load a dataset in <b>Data Management</b> before training a model.')
        return
    df = st.session_state.data
    with st.container(border=True):
        st.markdown('<p class="section-title">Training Configuration</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()
        target_keys_fixed = ['loan amount', 'loan_amount']
        sens_keys_fixed = ['age', 'gender', 'genter']
        target_keys_backup = ['loan_status', 'loan', 'status', 'approve', 'default', 'target', 'y']
        sens_keys_backup = ['sex', 'race', 'ethnicity', 'religion', 'marital', 'citizenship']
        target_col = all_cols[-1]
        found_target = False
        for c in all_cols:
            if c.lower() in target_keys_fixed:
                target_col = c
                found_target = True
                break
        if not found_target:
            for c in all_cols:
                c_low = c.lower().replace('_', ' ')
                if any((k in c_low for k in target_keys_backup)):
                    target_col = c
                    break
        sens_keys_strict = ['age', 'gender', 'genter']
        target_keys_fixed = ['loan amount', 'loan_amount']
        target_keys_backup = ['loan_status', 'loan', 'status', 'approve', 'default', 'target', 'y']
        target_col = all_cols[-1]
        found_target = False
        for c in all_cols:
            if c.lower() in target_keys_fixed:
                target_col = c
                found_target = True
                break
        if not found_target:
            for c in all_cols:
                if any((k in c.lower().replace('_', ' ') for k in target_keys_backup)):
                    target_col = c
                    break
        sensitive_cols = []
        for c in all_cols:
            c_low = c.lower().replace('_', ' ')
            if any((k in c_low for k in sens_keys_strict)):
                sensitive_cols.append(c)
        if not sensitive_cols:
            st.error('No demographic attributes (Age or Gender) found in dataset for auditing.')
            return
        if not target_col or not sensitive_cols:
            st.error('Automated detection failed. Please ensure your dataset has clear column names.')
            return
        selected_features = [c for c in all_cols if c != target_col]
        with st.expander('Custom Configuration (Manual Override)', expanded=False):
            c_ed1, c_ed2 = st.columns(2)
            with c_ed1:
                target_col = st.selectbox('Override Target Column', all_cols, index=all_cols.index(target_col), help='Choose the outcome you want to predict.')
            with c_ed2:
                available_sens_options = [c for c in all_cols if c != target_col]
                valid_default_sens = [c for c in sensitive_cols if c in available_sens_options]
                sensitive_cols = st.multiselect('Override Audited Attributes', options=available_sens_options, default=valid_default_sens, help='Choose one or more columns to check for bias.')
            if not sensitive_cols:
                st.error('Select at least one attribute to audit.')
                st.stop()
        status_bar_html = f'<div class="config-status-bar"><div class="config-status-flex"><div class="config-status-item"><p class="config-status-label">STATUS</p><p class="config-status-value accent">Data Ready</p></div><div class="config-status-item"><p class="config-status-label">TARGET</p><p class="config-status-value">{target_col.replace("_", " ")}</p></div><div class="config-status-item"><p class="config-status-label">AUDITING</p><p class="config-status-value">{", ".join([c.replace("_", " ") for c in sensitive_cols[:3]])}</p></div></div></div>'
        st.markdown(status_bar_html, unsafe_allow_html=True)
        c_btn, c_set = st.columns([2, 1])
        with c_btn:
            train_clicked = st.button('LAUNCH TRAINING PIPELINE', width='stretch')
        with c_set:
            with st.popover('Advanced Settings'):
                model_type = st.selectbox('Algorithm Selection', ['Random Forest', 'Logistic Regression'], index=0)
                st.info('Random Forest is recommended for higher precision.')
    if train_clicked:
        with st.spinner('Executing fairness-aware training on all columns...'):
            MAX_ROWS = 100000
            df_to_train = df.copy()
            if len(df_to_train) > MAX_ROWS:
                st.warning(f'Dataset is large ({len(df_to_train):,} rows). Subsampling to {MAX_ROWS:,} for analysis.')
                df_to_train = df_to_train.sample(MAX_ROWS, random_state=42)
            primary_sens = sensitive_cols[0]
            try:
                df_subset = df_to_train[selected_features + [target_col]]
                X, y, sf_proc, encoders, sf_raw = preprocess_data(df_subset, target_col=target_col, sensitive_col=primary_sens)
                with st.spinner('Executing model training...'):
                    model, metrics, X_test, y_test, y_pred, X_train, y_train, sf_train, sf_test = train_model(X, y, sensitive_features=sf_raw, model_type=model_type)
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.sf_train = sf_train
                st.session_state.sf_test = sf_test
                st.session_state.model_type = model_type
                st.session_state.sensitive_col = primary_sens
                st.session_state.all_sensitive_cols = sensitive_cols
                st.session_state.bias_metrics = None
                st.session_state.approval_rates = None
                st.session_state.mitigated_model = None
                st.session_state.mitigated_metrics = None
                st.session_state.mitigated_bias_metrics = None
                st.session_state.mitigated_approval_rates = None
                st.session_state.mitigation_method = None
                st.session_state.report_text = None
                st.success(f'{model_type} trained successfully on {len(X_train)} samples.')
                
                # Show Feature Importance
                st.markdown('<br>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown('<p class="section-title">Model Decision Drivers (Feature Importance)</p>', unsafe_allow_html=True)
                    fi = get_native_feature_importance(st.session_state.model, st.session_state.X_train.columns)
                    fig = go.Figure(go.Bar(
                        x=fi['Importance'],
                        y=fi['Feature'],
                        orientation='h',
                        marker_color=PRIMARY
                    ))
                    fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, width='stretch')

            except Exception as e:
                st.error(f'Training failed: {e}')
                return
    if st.session_state.metrics:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Model Performance Metrics</p>', unsafe_allow_html=True)
            m = st.session_state.metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_kpi('Accuracy', f"{m['Accuracy']:.2%}", 'green')
            with c2:
                render_kpi('Precision', f"{m['Precision']:.2%}", 'blue')
            with c3:
                render_kpi('Recall', f"{m['Recall']:.2%}", 'blue')
            with c4:
                render_kpi('F1 Score', f"{m['F1 Score']:.2%}", 'green')
            st.markdown('<br>', unsafe_allow_html=True)
            metric_names = list(m.keys())
            metric_vals = list(m.values())
            fig = go.Figure(go.Bar(x=metric_names, y=metric_vals, marker_color=[ACCENT, BLUE, BLUE, ACCENT], text=[f'{v:.2%}' for v in metric_vals], textposition='auto'))
            fig.update_layout(title='Performance Summary', xaxis_title='Metric', yaxis_title='Score', yaxis=dict(range=[0, 1]))
            st.plotly_chart(plotly_theme(fig), width='stretch')

def page_bias_analysis():
    render_page_header('Bias Analysis', 'Audit model predictions for demographic disparities')
    if st.session_state.model is None or st.session_state.sf_test is None:
        render_info('Please train a model in <b>Model Training</b> before running bias analysis.')
        return
    all_sens_raw = st.session_state.get('all_sensitive_cols', [st.session_state.sensitive_col])
    priority_keys = ['age', 'gender', 'genter']
    all_sens = [c for c in all_sens_raw if any((k in c.lower() for k in priority_keys))]
    if not all_sens:
        all_sens = all_sens_raw
    with st.container(border=True):
        st.markdown('<p class="section-title">Audit Dimensionality</p>', unsafe_allow_html=True)
        selected_attrs = st.multiselect('Select Dimensions to Audit', options=all_sens, default=[all_sens[0]] if all_sens else [], format_func=lambda x: x.replace('_', ' ').upper())
        if not selected_attrs:
            st.error('Please select at least one dimension (Age or Gender) to perform the audit.')
            return
        run_audit = st.button('RUN FAIRNESS AUDIT', width='stretch')
    if run_audit:
        with st.spinner('Analyzing model disparities...'):
            audit_results = {}
            df_full = st.session_state.data
            X_test_indices = st.session_state.X_test.index
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            for attr in selected_attrs:
                sf_raw_active = df_full.loc[X_test_indices, attr]
                bias_metrics, approval_rates = detect_bias(st.session_state.y_test, y_pred, sf_raw_active)
                audit_results[attr] = {'metrics': bias_metrics, 'rates': approval_rates}
            st.session_state.audit_results = audit_results
            st.session_state.bias_metrics = audit_results[selected_attrs[0]]['metrics']
            st.session_state.approval_rates = audit_results[selected_attrs[0]]['rates']
            st.session_state.active_audit_col = selected_attrs[0]
    if st.session_state.get('audit_results'):
        results = st.session_state.audit_results
        for idx, (attr_name, data) in enumerate(results.items()):
            attr_label = attr_name.replace('_', ' ').upper()
            st.markdown(f'---')
            st.markdown(f'<h3 style="color:{PRIMARY}; margin-bottom:25px;">AUDIT REPORT: {attr_label}</h3>', unsafe_allow_html=True)
            bm = data['metrics']
            apr = data['rates']
            di = bm['Disparate Impact']
            risk_label, risk_color = classify_risk(di)
            badge_variant = 'green' if 'Fair' in risk_label else 'red' if 'High' in risk_label else 'amber'
            risk_card_html = f'<div class="risk-assessment-card" style="border-left: 4px solid {risk_color};"><p>Risk Assessment ({attr_label}): {render_badge(risk_label, badge_variant)}</p></div>'
            st.markdown(risk_card_html, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            color_map = {ACCENT: 'green', RED: 'red', AMBER: 'amber'}
            with c1:
                render_kpi('Disparate Impact', f'{di:.3f}', color_map.get(risk_color, ''))
            with c2:
                render_kpi('Demographic Parity Diff', f"{abs(bm['Demographic Parity Difference']):.3f}", 'blue')
            with c3:
                render_kpi('Equal Opportunity Diff', f"{abs(bm['Equal Opportunity Difference']):.3f}", 'blue')
            st.markdown('<br>', unsafe_allow_html=True)
            if apr:
                with st.container(border=True):
                    st.markdown(f'<p class="section-title">Group-wise Selection Rates by {attr_label}</p>', unsafe_allow_html=True)
                    groups = list(apr.keys())
                    values = list(apr.values())
                    colors = []
                    max_rate = max(values) if values else 1
                    for v in values:
                        ratio = v / max_rate if max_rate > 0 else 1
                        if ratio < 0.8:
                            colors.append(RED)
                        elif ratio < 0.9:
                            colors.append(AMBER)
                        else:
                            colors.append(ACCENT)
                    fig = go.Figure(go.Bar(x=groups, y=values, marker_color=colors, text=[f'{v:.1%}' for v in values], textposition='auto'))
                    fig.update_layout(title=f'Approval Rate by {attr_label}', xaxis_title='Group', yaxis_title='Approval Rate', yaxis=dict(range=[0, 1], tickformat='.0%'))
                    fig.add_hline(y=max_rate * 0.8, line_dash='dash', line_color=RED, annotation_text='80% Threshold (Four-Fifths Rule)', annotation_position='top left')
                    st.plotly_chart(plotly_theme(fig), width='stretch', key=f'chart_{attr_name}_{idx}')
        with st.container(border=True):
            st.markdown('<p class="section-title">Compliance Decision Logic</p>', unsafe_allow_html=True)
            logic_data = [{'Range': 'DI < 0.80', 'Classification': 'High Risk', 'Action': 'Mitigation required'}, {'Range': '0.80 - 0.90', 'Classification': 'Moderate Risk', 'Action': 'Monitoring advised'}, {'Range': 'DI > 0.90', 'Classification': 'Fair', 'Action': 'Within compliance bounds'}]
            st.dataframe(pd.DataFrame(logic_data), width='stretch', hide_index=True)

def page_intersectional_audit():
    render_page_header('Intersectional Audit', 'Identify deep-seated bias across combined demographic groups')
    if st.session_state.model is None:
        render_info('Please train a model in <b>Model Training</b> first.')
        return
    
    all_sens = st.session_state.get('all_sensitive_cols', [])
    if len(all_sens) < 2:
        st.warning("Intersectional audit requires at least two sensitive attributes (e.g., Age and Gender).")
        return

    with st.container(border=True):
        st.markdown('<p class="section-title">Audit Configuration</p>', unsafe_allow_html=True)
        selected_cols = st.multiselect('Select attributes to intersect', options=all_sens, default=all_sens[:2])
        run_audit = st.button('GENERATE INTERSECTIONAL REPORT', width='stretch')

    if run_audit and len(selected_cols) >= 2:
        with st.spinner('Calculating intersectional disparities...'):
            df_full = st.session_state.data
            X_test_indices = st.session_state.X_test.index
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            df_sens_active = df_full.loc[X_test_indices, selected_cols]
            
            metrics, rates = detect_intersectional_bias(st.session_state.y_test, y_pred, df_sens_active)
            
            st.session_state.intersectional_results = {'metrics': metrics, 'rates': rates, 'cols': selected_cols}

    if 'intersectional_results' in st.session_state:
        res = st.session_state.intersectional_results
        metrics = res['metrics']
        rates = res['rates']
        
        c1, c2, c3 = st.columns(3)
        di = metrics['Disparate Impact']
        label, color = classify_risk(di)
        
        with c1:
            render_kpi('Intersectional DI', f'{di:.3f}', 'red' if di < 0.8 else 'green')
        with c2:
            render_kpi('Combined Parity Diff', f"{metrics['Demographic Parity Difference']:.3f}", 'blue')
        with c3:
            render_kpi('Risk Level', label, 'red' if 'High' in label else 'green')

        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f'<p class="section-title">Heatmap: Approval Rates by {", ".join(res["cols"])}</p>', unsafe_allow_html=True)
            
            # Prepare data for heatmap
            group_names = list(rates.keys())
            group_values = list(rates.values())
            
            fig = go.Figure(go.Bar(
                x=group_names,
                y=group_values,
                marker_color=[RED if v < max(group_values)*0.8 else ACCENT for v in group_values],
                text=[f'{v:.1%}' for v in group_values],
                textposition='auto'
            ))
            fig.update_layout(title="Intersectional Selection Rates", xaxis_title="Demographic Intersection", yaxis_title="Approval Rate", yaxis=dict(range=[0,1]))
            st.plotly_chart(plotly_theme(fig), width='stretch')
            
            st.info("💡 Intersectional bias often reveals hidden discrimination that single-attribute audits miss. For example, a model might be fair to 'Women' and fair to 'Black applicants' but unfair to 'Black Women'.")

def page_mitigation():
    render_page_header('Mitigation Engine', 'Apply fairness-aware retraining techniques to reduce model bias')
    if st.session_state.get('model') is None or st.session_state.get('X_test') is None:
        render_info('Please train a model in <b>Model Training</b> first to initialize the audit data.')
        return
    if st.session_state.get('bias_metrics') is None:
        render_info('Please run <b>Bias Analysis</b> first to establish a baseline.')
        return
    with st.container(border=True):
        st.markdown('<p class="section-title">Mitigation Configuration</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            method = st.selectbox('Mitigation Technique', ['Hybrid (Reweighing + Exp Gradient)', 'Exponentiated Gradient', 'Reweighing'], index=0)
        st.markdown('<br>', unsafe_allow_html=True)
        apply_clicked = st.button('Apply Mitigation', width='content')
    if apply_clicked:
            try:
                # Add a sub-spinner for the actual training phase
                with st.status("🛠️ Retraining model with fairness constraints...", expanded=True) as status:
                    st.write("Initializing AIF360 environment...")
                    mit_model = mitigate_bias(st.session_state.X_train, st.session_state.y_train, st.session_state.sf_train, model_type=st.session_state.model_type, method=method)
                    
                    st.write("Recalculating performance metrics...")
                    X_test_active = st.session_state.get('X_test')
                    y_test_active = st.session_state.get('y_test')
                    sf_test_active = st.session_state.get('sf_test')

                    if X_test_active is not None and y_test_active is not None:
                        y_pred_mit = mit_model.predict(X_test_active)
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        new_metrics = {
                            'Accuracy': accuracy_score(y_test_active, y_pred_mit), 
                            'Precision': precision_score(y_test_active, y_pred_mit, zero_division=0), 
                            'Recall': recall_score(y_test_active, y_pred_mit, zero_division=0), 
                            'F1 Score': f1_score(y_test_active, y_pred_mit, zero_division=0)
                        }
                        st.session_state.mitigated_metrics = new_metrics
                        
                        st.write("Analyzing post-mitigation bias...")
                        new_bias, new_rates = detect_bias(y_test_active, y_pred_mit, sf_test_active)
                        st.session_state.mitigated_bias_metrics = new_bias
                        st.session_state.mitigated_approval_rates = new_rates
                        st.session_state.mitigated_model = mit_model
                        st.session_state.mitigation_method = method
                        status.update(label="✅ Mitigation complete!", state="complete", expanded=False)
                    else:
                        st.error("Audit data lost during session. Please re-train the model.")
                        status.update(label="❌ Mitigation failed", state="error")
                
                st.success(f'Bias mitigation via {method} complete.')
            except Exception as e:
                st.error(f'Mitigation failed: {e}')
    if st.session_state.mitigated_metrics:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Fairness-Accuracy Trade-off Frontier</p>', unsafe_allow_html=True)
            
            # Prepare data for trade-off chart
            base_acc = st.session_state.metrics['Accuracy']
            base_di = st.session_state.bias_metrics['Disparate Impact']
            mit_acc = st.session_state.mitigated_metrics['Accuracy']
            mit_di = st.session_state.mitigated_bias_metrics['Disparate Impact']
            
            fig = go.Figure()
            # Baseline point
            fig.add_trace(go.Scatter(x=[base_di], y=[base_acc], mode='markers+text', name='Baseline Model', text=['BASELINE'], textposition='top center', marker=dict(size=15, color=RED)))
            # Mitigated point
            fig.add_trace(go.Scatter(x=[mit_di], y=[mit_acc], mode='markers+text', name='Optimized Model', text=['OPTIMIZED'], textposition='top center', marker=dict(size=15, color=ACCENT)))
            
            fig.update_layout(
                title={'text': "Model Trade-off Analysis", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                xaxis_title='Fairness (Disparate Impact Ratio)', 
                yaxis_title='Accuracy (%)', 
                xaxis=dict(range=[0, 1.1]), 
                yaxis=dict(tickformat='.1%'),
                margin=dict(t=80)
            )
            # Target region
            fig.add_vrect(
                x0=0.8, x1=1.0, 
                fillcolor="rgba(22, 163, 74, 0.1)", 
                layer="below", 
                line_width=0, 
                annotation_text="Regulatory Compliance Zone", 
                annotation_position="bottom left",
                annotation_font_size=12,
                annotation_font_color="green"
            )
            
            st.plotly_chart(plotly_theme(fig, 500), width='stretch')
            st.info("💡 The 'Frontier' illustrates how much predictive accuracy is sacrificed to gain algorithmic fairness. A 'Perfect' model would be in the top-right corner.")

    if st.session_state.mitigated_bias_metrics:
        mbm = st.session_state.mitigated_bias_metrics
        mm = st.session_state.mitigated_metrics
        di_new = mbm['Disparate Impact']
        risk_label, risk_color = classify_risk(di_new)
        badge_variant = 'green' if 'Fair' in risk_label else 'red' if 'High' in risk_label else 'amber'
        st.markdown(f'<div style="background:{CARD_BG}; border: 1px solid {BORDER}; border-left: 4px solid {risk_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px;"><p style="margin:0; font-size:1rem; font-weight:700; color:{PRIMARY};">Post-Mitigation Status: {render_badge(risk_label, badge_variant)}</p></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            color_map = {ACCENT: 'green', RED: 'red', AMBER: 'amber'}
            render_kpi('New Disparate Impact', f'{di_new:.3f}', color_map.get(risk_color, ''))
        with c2:
            render_kpi('New Accuracy', f"{mm['Accuracy']:.2%}", 'blue')
        with c3:
            render_kpi('New Precision', f"{mm['Precision']:.2%}", 'blue')
        with c4:
            render_kpi('New F1 Score', f"{mm['F1 Score']:.2%}", 'green')

        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Mitigation Outcome: Group-wise Fairness Improvement</p>', unsafe_allow_html=True)
            
            # Comparison Data
            apr_before = st.session_state.approval_rates
            apr_after = st.session_state.mitigated_approval_rates
            
            if apr_before and apr_after:
                groups = list(apr_before.keys())
                fig = go.Figure()
                
                # Baseline
                fig.add_trace(go.Bar(
                    name='Baseline Approval',
                    x=groups,
                    y=[apr_before.get(g, 0) for g in groups],
                    marker_color='#94A3B8',
                    text=[f'{apr_before.get(g, 0):.1%}' for g in groups],
                    textposition='auto'
                ))
                
                # Mitigated
                fig.add_trace(go.Bar(
                    name='Mitigated Approval',
                    x=groups,
                    y=[apr_after.get(g, 0) for g in groups],
                    marker_color=ACCENT,
                    text=[f'{apr_after.get(g, 0):.1%}' for g in groups],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    barmode='group',
                    title=f'Successive Impact of {st.session_state.mitigation_method} Mitigation',
                    xaxis_title='Sensitive Group',
                    yaxis_title='Approval Rate',
                    yaxis=dict(range=[0, 1], tickformat='.0%'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Add 80% rule line
                max_rate_after = max(apr_after.values()) if apr_after.values() else 1
                fig.add_hline(y=max_rate_after * 0.8, line_dash="dash", line_color=RED, 
                             annotation_text="Fairness Threshold (80% Rule)", 
                             annotation_position="top left")
                
                st.plotly_chart(plotly_theme(fig), width='stretch')
            else:
                st.warning("Baseline rates not found. Please ensure Bias Analysis was run correctly.")

def page_comparison():
    render_page_header('Performance Comparison', 'Before vs after mitigation: Performance and fairness trade-off analysis')
    if st.session_state.mitigated_metrics is None:
        render_info('Apply mitigation in the <b>Mitigation Engine</b> to view comparison results.')
        return
    m_before = st.session_state.metrics
    m_after = st.session_state.mitigated_metrics
    b_before = st.session_state.bias_metrics
    b_after = st.session_state.mitigated_bias_metrics
    col_left, col_right = st.columns(2)
    with col_left:
        with st.container(border=True):
            st.markdown('<p class="section-title">Before Mitigation</p>', unsafe_allow_html=True)
            for metric_name, val in m_before.items():
                st.markdown(f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};"><b>{metric_name}</b>: {val:.4f}</p>', unsafe_allow_html=True)
            st.divider()
            for metric_name, val in b_before.items():
                st.markdown(f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};"><b>{metric_name}</b>: {val:.4f}</p>', unsafe_allow_html=True)
    with col_right:
        with st.container(border=True):
            st.markdown('<p class="section-title">After Mitigation</p>', unsafe_allow_html=True)
            for metric_name, val in m_after.items():
                st.markdown(f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};"><b>{metric_name}</b>: {val:.4f}</p>', unsafe_allow_html=True)
            st.divider()
            for metric_name, val in b_after.items():
                st.markdown(f'<p style="margin:8px 0; font-size:0.9rem; color:{TEXT};"><b>{metric_name}</b>: {val:.4f}</p>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<p class="section-title">Performance Metrics Comparison</p>', unsafe_allow_html=True)
        perf_metrics = list(m_before.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Before Mitigation', x=perf_metrics, y=[m_before[k] for k in perf_metrics], marker_color='#94A3B8', text=[f'{m_before[k]:.2%}' for k in perf_metrics], textposition='auto'))
        fig.add_trace(go.Bar(name='After Mitigation', x=perf_metrics, y=[m_after[k] for k in perf_metrics], marker_color=ACCENT, text=[f'{m_after[k]:.2%}' for k in perf_metrics], textposition='auto'))
        fig.update_layout(barmode='group', title='Model Performance: Before vs After', xaxis_title='Metric', yaxis_title='Score', yaxis=dict(range=[0, 1]), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(plotly_theme(fig, 420), width='stretch')
    with st.container(border=True):
        st.markdown('<p class="section-title">Fairness Metrics Comparison</p>', unsafe_allow_html=True)
        bias_metric_names = list(b_before.keys())
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Before Mitigation', x=bias_metric_names, y=[abs(b_before[k]) for k in bias_metric_names], marker_color='#94A3B8', text=[f'{abs(b_before[k]):.3f}' for k in bias_metric_names], textposition='auto'))
        fig2.add_trace(go.Bar(name='After Mitigation', x=bias_metric_names, y=[abs(b_after[k]) for k in bias_metric_names], marker_color=ACCENT, text=[f'{abs(b_after[k]):.3f}' for k in bias_metric_names], textposition='auto'))
        fig2.update_layout(barmode='group', title='Fairness Metrics: Before vs After', xaxis_title='Metric', yaxis_title='Value', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        fig2.add_hline(y=0.8, line_dash='dash', line_color=RED, annotation_text='Compliance Threshold (0.8)', annotation_position='top left')
        st.plotly_chart(plotly_theme(fig2, 420), width='stretch')
    with st.container(border=True):
        st.markdown('<p class="section-title">Impact Summary</p>', unsafe_allow_html=True)
        acc_delta = m_after['Accuracy'] - m_before['Accuracy']
        di_delta = b_after['Disparate Impact'] - b_before['Disparate Impact']
        dpd_delta = abs(b_after['Demographic Parity Difference']) - abs(b_before['Demographic Parity Difference'])
        impact_data = [{'Metric': 'Accuracy Change', 'Delta': f'{acc_delta:+.4f}', 'Direction': 'Improved' if acc_delta >= 0 else 'Decreased'}, {'Metric': 'Disparate Impact Change', 'Delta': f'{di_delta:+.4f}', 'Direction': 'Improved' if di_delta > 0 else 'Decreased'}, {'Metric': 'Parity Difference Change', 'Delta': f'{dpd_delta:+.4f}', 'Direction': 'Improved' if dpd_delta < 0 else 'Increased'}]
        st.dataframe(pd.DataFrame(impact_data), width='stretch', hide_index=True)

def page_explainability():
    render_page_header('Model Explainability', 'SHAP-based feature importance and decision interpretation')
    if st.session_state.model is None:
        render_info('Please train a model in <b>Model Training</b> first.')
        return
    model_to_explain = st.session_state.mitigated_model if st.session_state.mitigated_model else st.session_state.model
    model_label = 'Mitigated Model' if st.session_state.mitigated_model else 'Baseline Model'
    with st.container(border=True):
        st.markdown('<p class="section-title">SHAP Analysis Configuration</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.9rem; color:{TEXT_MUTED};">Analyzing: <b style="color:{PRIMARY};">{model_label}</b> ({st.session_state.model_type})</p>', unsafe_allow_html=True)
        compute_clicked = st.button('Compute SHAP Values', width='content')
    if compute_clicked:
        with st.spinner('Computing SHAP values... This may take a moment.'):
            try:
                shap_values, X_sample = compute_shap_values(model_to_explain, st.session_state.X_train, st.session_state.X_test, st.session_state.model_type)
                feature_names = list(st.session_state.X_test.columns) if hasattr(st.session_state.X_test, 'columns') else [f'Feature {i}' for i in range(st.session_state.X_test.shape[1])]
                importance_df = get_feature_importance(shap_values, feature_names)
                st.session_state['shap_importance'] = importance_df
                st.markdown('<br>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown('<p class="section-title">Feature Importance (Mean |SHAP|)</p>', unsafe_allow_html=True)
                    top_n = min(15, len(importance_df))
                    top_features = importance_df.head(top_n).iloc[::-1]
                    fig = go.Figure(go.Bar(x=top_features['Importance'], y=top_features['Feature'], orientation='h', marker_color=ACCENT, text=[f'{v:.4f}' for v in top_features['Importance']], textposition='auto'))
                    fig.update_layout(title='Top Feature Contributions to Model Decisions', xaxis_title='Mean |SHAP Value|', yaxis_title='Feature')
                    st.plotly_chart(plotly_theme(fig, 500), width='stretch')
                with st.container(border=True):
                    st.markdown('<p class="section-title">SHAP Summary Plot</p>', unsafe_allow_html=True)
                    summary_fig = generate_shap_summary_plot(model_to_explain, st.session_state.X_train, st.session_state.X_test, st.session_state.model_type)
                    st.pyplot(summary_fig)
            except Exception as e:
                st.error(f'SHAP computation failed: {e}')

def page_real_time_simulator():
    render_page_header('Real-time Stream Simulator', 'Visualizing live loan decisions and rolling bias monitoring')
    if st.session_state.model is None:
        render_info('Please train a model in <b>Model Training</b> first.')
        return
    import time
    with st.container(border=True):
        st.markdown('<p class="section-title">Simulator Control Panel</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            stream_speed = st.slider('Stream Speed (seconds)', 0.01, 1.0, 0.1)
        with c2:
            max_available = len(st.session_state.get('X_test')) if st.session_state.get('X_test') is not None else 10000
            batch_size = st.number_input('Simulate Applicants (Current Max)', 10, max_available, max_available)
        with c3:
            st.markdown('<br>', unsafe_allow_html=True)
            start_stream = st.button('LAUNCH UNLIMITED AUDIT', width='stretch', type='primary')
    if start_stream:
        ticker_placeholder = st.empty()
        roster_placeholder = st.empty()
        progress_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        X_test_full = st.session_state.get('X_test')
        df_sample = X_test_full.iloc[:batch_size].reset_index(drop=True)
        model = st.session_state.get('mitigated_model') if st.session_state.get('mitigated_model') else st.session_state.get('model')
        
        all_decisions = []
        all_demographics = []
        history_data = []
        
        # Identify key features to show in ticker (dynamic)
        key_features = [c for c in df_sample.columns if 'income' in c.lower() or 'score' in c.lower() or 'amount' in c.lower() or 'credit' in c.lower()][:3]
        
        for i in range(batch_size):
            row = df_sample.iloc[[i]]
            pred, conf = get_prediction_and_confidence(model, row)
            all_decisions.append(pred)
            
            # Get sensitive value
            sf_full = st.session_state.get('sf_test')
            sens_val = sf_full.iloc[i % len(sf_full)]
            all_demographics.append(sens_val)
            
            status = "✅ APPROVED" if pred == 1 else "❌ DENIED"
            color = ACCENT if pred == 1 else RED
            
            # Create feature string
            feat_str = " | ".join([f"{f}: **{row[f].values[0]}**" for f in key_features])
            
            ticker_html = f'''
                <div class="stream-ticker" style="border-left: 5px solid {color}; background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: {PRIMARY}; font-size: 0.85rem;">APPLICANT ID: LN-{5000 + i}</span>
                        <span style="font-weight: 800; color: {color}; font-size: 0.9rem;">{status}</span>
                    </div>
                    <p style="margin: 0; font-size: 0.9rem; color: {TEXT};">{feat_str}</p>
                    <div style="margin-top: 10px; display: flex; gap: 10px;">
                        <span style="background: {BG}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Group: <b>{sens_val}</b></span>
                        <span style="background: {BG}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">Confidence: <b>{conf}</b></span>
                    </div>
                </div>
            '''
            
            # Add to history (keep last 5)
            row_dict = row.to_dict('records')[0]
            row_dict['Decision'] = status
            row_dict['Group'] = sens_val
            history_data.insert(0, row_dict)
            if len(history_data) > 8:
                history_data.pop()

            with ticker_placeholder.container():
                st.markdown(ticker_html, unsafe_allow_html=True)
            
            with roster_placeholder.container(border=True):
                st.markdown(f'<p class="section-title">Live Decision Roster ({i+1}/{batch_size})</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(history_data), width='stretch', hide_index=True)
            
            # Progress tracking
            progress_val = (i + 1) / batch_size
            progress_placeholder.progress(progress_val, text=f"Auditing Data Stream: {i+1} of {batch_size} applicants processed...")

            if (i+1) % 10 == 0:
                y_pred_rolling = np.array(all_decisions)
                sf_rolling = np.array(all_demographics)
                unique_groups = np.unique(sf_rolling)
                rates = {str(g): np.mean(y_pred_rolling[sf_rolling == g]) for g in unique_groups}
                with chart_placeholder.container(border=True):
                    st.markdown('<p class="section-title">Rolling Bias Monitor (Live)</p>', unsafe_allow_html=True)
                    fig = go.Figure(go.Bar(x=list(rates.keys()), y=list(rates.values()), marker_color=ACCENT))
                    fig.update_layout(height=300, yaxis=dict(range=[0,1]))
                    st.plotly_chart(plotly_theme(fig, 300), width='stretch')
            time.sleep(stream_speed)
        st.success(f"Simulation of {batch_size} applications complete.")

def page_what_if():
    render_page_header('What-If Analysis & Actionable Recourse', 'Simulate individual decisions and generate counterfactual explanations')
    if st.session_state.model is None:
        render_info('Please train a model in <b>Model Training</b> first.')
        return
    with st.container(border=True):
        st.markdown('<p class="section-title">Applicant Profile Simulator</p>', unsafe_allow_html=True)
        # Filter out non-predictive columns
        cols = [c for c in st.session_state.X_train.columns if c.lower() not in ['applicant_name', 'name', 'id', 'application_id']]
        input_data = {}
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate(cols):
            target_col = c1 if i % 3 == 0 else c2 if i % 3 == 1 else c3
            with target_col:
                if st.session_state.X_train[col].dtype == 'object' or len(st.session_state.X_train[col].unique()) < 10:
                    options = sorted(list(st.session_state.X_train[col].unique()))
                    input_data[col] = st.selectbox(f'{col}', options)
                else:
                    min_val, max_val, mean_val = float(st.session_state.X_train[col].min()), float(st.session_state.X_train[col].max()), float(st.session_state.X_train[col].mean())
                    # Use number input for cleaner interface without "hover" values
                    input_data[col] = st.number_input(f'{col}', min_val, max_val, mean_val)
    st.markdown('<br>', unsafe_allow_html=True)
    input_df = pd.DataFrame([input_data])
    
    # Re-add missing columns with defaults if any were filtered for prediction
    for c in st.session_state.X_train.columns:
        if c not in input_df.columns:
            input_df[c] = st.session_state.X_train[c].mode()[0] if not pd.api.types.is_numeric_dtype(st.session_state.X_train[c]) else st.session_state.X_train[c].mean()
    
    # Ensure correct feature order
    input_df = input_df[st.session_state.X_train.columns]

    c_base, c_mit = st.columns(2)
    with c_base:
        with st.container(border=True):
            st.markdown('<p class="section-title">Baseline Decision</p>', unsafe_allow_html=True)
            pred_base = st.session_state.model.predict(input_df)[0]
            badge_class = 'decision-badge-approved' if pred_base == 1 else 'decision-badge-denied'
            res_label = 'Approved' if pred_base == 1 else 'Denied'
            st.markdown(f'<div class="decision-badge {badge_class}"><h3>{res_label}</h3></div>', unsafe_allow_html=True)
    with c_mit:
        with st.container(border=True):
            st.markdown('<p class="section-title">Mitigated Decision</p>', unsafe_allow_html=True)
            if st.session_state.mitigated_model:
                pred_mit = st.session_state.mitigated_model.predict(input_df)[0]
                badge_class_m = 'decision-badge-approved' if pred_mit == 1 else 'decision-badge-denied'
                res_label_m = 'Approved' if pred_mit == 1 else 'Denied'
                st.markdown(f'<div class="decision-badge {badge_class_m}"><h3>{res_label_m}</h3></div>', unsafe_allow_html=True)
            else:
                st.info("No mitigation applied.")

    if pred_base == 0:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Actionable Recourse (Counterfactuals)</p>', unsafe_allow_html=True)
            if st.button('HOW CAN THIS APPLICANT GET APPROVED?'):
                with st.spinner('Calculating minimal changes for approval...'):
                    # Prepare data for DiCE
                    train_df = st.session_state.X_train.copy()
                    train_df[st.session_state.sensitive_col + '_target'] = st.session_state.y_train # Mock target name
                    target_name = st.session_state.sensitive_col + '_target'
                    cf_df = generate_counterfactuals(st.session_state.model, train_df, target_name, input_df)
                    if cf_df is not None:
                        st.dataframe(cf_df)
                        diffs = get_actionable_diff(input_df, cf_df)
                        for i, changes in enumerate(diffs):
                            if changes:
                                st.markdown(f"**Option {i+1}:** " + ", ".join([f"Change {c['feature']} from {c['from']} to {c['to']}" for c in changes]))
                    else:
                        st.error("Could not find a path to approval for this specific profile.")

def page_reports():
    render_page_header('Compliance Reports', 'Generate structured regulatory compliance documentation')
    if st.session_state.metrics is None:
        render_info('Complete <b>Model Training</b> and <b>Bias Analysis</b> first.')
        return
    with st.container(border=True):
        st.markdown('<p class="section-title">Report Configuration</p>', unsafe_allow_html=True)
        if st.button('Generate Compliance Report', width='stretch'):
            with st.spinner('Generating...'):
                # Prepare comparison data for the roster
                X_test = st.session_state.X_test
                y_pred_base = st.session_state.model.predict(X_test)
                
                df_comp = st.session_state.data.loc[X_test.index].copy()
                df_comp['Before'] = y_pred_base
                
                if st.session_state.mitigated_model:
                    y_pred_mit = st.session_state.mitigated_model.predict(X_test)
                    df_comp['After'] = y_pred_mit
                else:
                    df_comp['After'] = y_pred_base
                
                df_comp['Decision Changed'] = df_comp['Before'] != df_comp['After']
                
                st.session_state.report_text = generate_report(
                    st.session_state.metrics, 
                    st.session_state.bias_metrics, 
                    st.session_state.mitigated_metrics, 
                    st.session_state.mitigated_bias_metrics, 
                    st.session_state.sensitive_col,
                    model_type=st.session_state.get('model_type'),
                    mitigation_method=st.session_state.get('mitigation_method'),
                    df_comparison=df_comp
                )
                st.session_state.report_pdf = generate_pdf_report(
                    st.session_state.metrics, 
                    st.session_state.bias_metrics, 
                    st.session_state.mitigated_metrics, 
                    st.session_state.mitigated_bias_metrics, 
                    st.session_state.sensitive_col,
                    model_type=st.session_state.get('model_type'),
                    mitigation_method=st.session_state.get('mitigation_method'),
                    df_comparison=df_comp
                )
    if st.session_state.report_text:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<p class="section-title">Audit Report Preview</p>', unsafe_allow_html=True)
            st.markdown(st.session_state.report_text)
        st.download_button('Download Official PDF Report', st.session_state.report_pdf, 'Compliance_Report.pdf', 'application/pdf', width='stretch')

PAGES = {
    'Overview': page_overview, 
    'Data Management': page_data_management, 
    'Model Training': page_model_training, 
    'Bias Analysis': page_bias_analysis, 
    'Intersectional Audit': page_intersectional_audit,
    'Mitigation Engine': page_mitigation, 
    'Performance Comparison': page_comparison, 
    'Explainability': page_explainability, 
    'Real-time Simulator': page_real_time_simulator, 
    'What-If Analysis': page_what_if, 
    'Compliance Reports': page_reports
}

# Render Active Page
PAGES[st.session_state.active_page]()

# Navigation Footer (Back & Next Buttons)
st.markdown('<br>', unsafe_allow_html=True)
st.divider()
curr_idx = ORDERED_PAGES.index(st.session_state.active_page)

c_prev, c_sp, c_next = st.columns([1.2, 1.6, 1.2])

with c_prev:
    if curr_idx > 0:
        prev_page_name = ORDERED_PAGES[curr_idx - 1]
        if st.button(f'← Previous: {prev_page_name}', width='stretch', key=f'prev_step_{curr_idx}'):
            st.session_state.active_page = prev_page_name
            st.session_state.active_category = PAGE_TO_CAT[prev_page_name]
            st.rerun()

with c_next:
    if curr_idx < len(ORDERED_PAGES) - 1:
        next_page_name = ORDERED_PAGES[curr_idx + 1]
        if st.button(f'Next Step: {next_page_name} →', width='stretch', key=f'next_step_{curr_idx}'):
            st.session_state.active_page = next_page_name
            st.session_state.active_category = PAGE_TO_CAT[next_page_name]
            st.rerun()

st.markdown('<br><br>', unsafe_allow_html=True)

