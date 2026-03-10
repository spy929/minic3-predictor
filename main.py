"""
MINIC3 Prediction System
A machine learning-based platform for dual-task prediction of treatment response and adverse events
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MINIC3 Prediction System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1e3c72;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2a5298;
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #2a5298;
    }
    .risk-low { color: #27ae60; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-high { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧬 MINIC3 Prediction System</div>', unsafe_allow_html=True)
st.markdown("#### Machine Learning-Based Dual-Task Prediction Platform")

# Generate synthetic clinical data
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 2000
    
    data = {
        'Patient_ID': [f'P{str(i).zfill(4)}' for i in range(1, n+1)],
        'Age': np.random.normal(62, 12, n).astype(int).clip(25, 90),
        'Gender': np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
        'ECOG': np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.4, 0.3, 0.1]),
        'Dose': np.random.choice([0.3, 1.0, 3.0, 10.0], n, p=[0.1, 0.2, 0.4, 0.3]),
        'Prior_Therapies': np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.4, 0.3, 0.1]),
        'Metastasis_Sites': np.random.poisson(2, n).clip(0, 5),
        'Liver_Mets': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'PDL1': np.random.choice(['Negative', 'Low', 'High'], n, p=[0.3, 0.4, 0.3]),
        'TMB': np.random.exponential(8, n).round(1).clip(0, 50),
        'NLR': np.random.normal(3, 1.5, n).round(2).clip(0.5, 15),
        'LDH': np.random.normal(200, 80, n).round(0).clip(100, 600),
        'CRP': np.random.exponential(15, n).round(1).clip(1, 150),
        'Albumin': np.random.normal(38, 5, n).round(1).clip(25, 50),
    }
    
    df = pd.DataFrame(data)
    
    # Generate outcomes
    response_prob = 0.2 + 0.05*(df['Dose']>1) + 0.1*(df['PDL1']=='High')
    response_prob = response_prob.clip(0.1, 0.8)
    df['Response'] = np.random.binomial(1, response_prob)
    
    ae_prob = 0.3 + 0.05*(df['Dose']>3) + 0.01*(df['Age']-60).clip(0,20)
    ae_prob = ae_prob.clip(0.2, 0.9)
    df['AE'] = np.random.binomial(1, ae_prob)
    
    df['PFS'] = np.where(df['Response']==1, np.random.normal(18,6,n), np.random.normal(5,2,n)).clip(1,48).round(1)
    
    # Risk stratification
    risk_score = df['ECOG']*2 + (df['LDH']>250).astype(int)*3 + (df['NLR']>5).astype(int)*2
    df['Risk_Group'] = pd.cut(risk_score, bins=[0,3,6,10], labels=['Low', 'Medium', 'High'])
    
    return df

# Machine Learning Model
class Predictor:
    def __init__(self):
        self.model_response = None
        self.model_ae = None
        self.scaler = StandardScaler()
        self.feature_cols = ['Age', 'ECOG', 'Dose', 'Prior_Therapies', 'Metastasis_Sites', 
                             'Liver_Mets', 'TMB', 'NLR', 'LDH', 'Albumin']
        self.metrics = {}
        self.roc_data = {}
        self.importance_df = None
        
    def prepare_features(self, df, fit=False):
        df_encoded = df.copy()
        df_encoded['PDL1_encoded'] = df['PDL1'].map({'Negative':0, 'Low':1, 'High':2})
        features = self.feature_cols + ['PDL1_encoded']
        X = df_encoded[features].fillna(0)
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train(self, df):
        with st.spinner('Training model...'):
            X = self.prepare_features(df, fit=True)
            y_res = df['Response']
            y_ae = df['AE']
            
            # Split data
            X_train, X_test, y_res_train, y_res_test = train_test_split(
                X, y_res, test_size=0.2, random_state=42, stratify=y_res
            )
            _, _, y_ae_train, y_ae_test = train_test_split(
                X, y_ae, test_size=0.2, random_state=42, stratify=y_ae
            )
            
            # Train models
            self.model_response = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            self.model_response.fit(X_train, y_res_train)
            
            self.model_ae = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            self.model_ae.fit(X_train, y_ae_train)
            
            # Predictions
            y_res_prob = self.model_response.predict_proba(X_test)[:,1]
            y_ae_prob = self.model_ae.predict_proba(X_test)[:,1]
            y_res_pred = self.model_response.predict(X_test)
            y_ae_pred = self.model_ae.predict(X_test)
            
            # ROC data
            fpr_res, tpr_res, _ = roc_curve(y_res_test, y_res_prob)
            fpr_ae, tpr_ae, _ = roc_curve(y_ae_test, y_ae_prob)
            
            self.roc_data = {
                'response': {'fpr': fpr_res, 'tpr': tpr_res, 'auc': roc_auc_score(y_res_test, y_res_prob)},
                'ae': {'fpr': fpr_ae, 'tpr': tpr_ae, 'auc': roc_auc_score(y_ae_test, y_ae_prob)}
            }
            
            # Feature importance
            self.importance_df = pd.DataFrame({
                'Feature': self.feature_cols + ['PDL1'],
                'Importance': self.model_response.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Metrics
            self.metrics = {
                'response': {
                    'accuracy': accuracy_score(y_res_test, y_res_pred),
                    'precision': precision_score(y_res_test, y_res_pred),
                    'recall': recall_score(y_res_test, y_res_pred),
                    'f1': f1_score(y_res_test, y_res_pred),
                    'auc': self.roc_data['response']['auc']
                },
                'ae': {
                    'accuracy': accuracy_score(y_ae_test, y_ae_pred),
                    'precision': precision_score(y_ae_test, y_ae_pred),
                    'recall': recall_score(y_ae_test, y_ae_pred),
                    'f1': f1_score(y_ae_test, y_ae_pred),
                    'auc': self.roc_data['ae']['auc']
                }
            }
            
            # Cross-validation
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            cv_res = cross_val_score(self.model_response, X, y_res, cv=cv, scoring='roc_auc')
            cv_ae = cross_val_score(self.model_ae, X, y_ae, cv=cv, scoring='roc_auc')
            
            self.metrics['response']['cv_mean'] = cv_res.mean()
            self.metrics['response']['cv_std'] = cv_res.std()
            self.metrics['ae']['cv_mean'] = cv_ae.mean()
            self.metrics['ae']['cv_std'] = cv_ae.std()
            
            return self.metrics
    
    def predict(self, features_df):
        X = self.prepare_features(features_df)
        resp_prob = self.model_response.predict_proba(X)[0][1]
        ae_prob = self.model_ae.predict_proba(X)[0][1]
        return resp_prob, ae_prob

# Initialize
if 'model' not in st.session_state:
    st.session_state.model = Predictor()
    df = generate_data()
    st.session_state.df = df
    metrics = st.session_state.model.train(df)
    st.session_state.metrics = metrics

df = st.session_state.df

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["📊 Data Overview", "🎯 Prediction", "📈 Model Performance", "📉 Survival Analysis"])
    st.markdown("---")
    st.metric("Total Patients", f"{len(df):,}")
    st.metric("Response AUC", f"{st.session_state.metrics['response']['auc']:.3f}")
    st.metric("AE AUC", f"{st.session_state.metrics['ae']['auc']:.3f}")

# Data Overview
if page == "📊 Data Overview":
    st.markdown('<div class="sub-header">📊 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Response Rate", f"{df['Response'].mean()*100:.1f}%")
    with col2:
        st.metric("AE Rate", f"{df['AE'].mean()*100:.1f}%")
    with col3:
        st.metric("Median PFS", f"{df['PFS'].median():.1f} months")
    with col4:
        st.metric("High Risk", f"{(df['Risk_Group']=='High').mean()*100:.1f}%")
    
    st.dataframe(df.head(20), use_container_width=True)

# Prediction
elif page == "🎯 Prediction":
    st.markdown('<div class="sub-header">🎯 Patient Prediction</div>', unsafe_allow_html=True)
    
    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 30, 90, 60)
            ecog = st.selectbox("ECOG", [0, 1, 2, 3])
            dose = st.selectbox("Dose (mg/kg)", [0.3, 1.0, 3.0, 10.0])
            prior = st.number_input("Prior Therapies", 0, 4, 1)
            mets = st.number_input("Metastasis Sites", 0, 5, 1)
            liver = st.checkbox("Liver Metastasis")
        
        with col2:
            pdl1 = st.selectbox("PD-L1", ["Negative", "Low", "High"])
            tmb = st.number_input("TMB (mut/Mb)", 0, 50, 8)
            nlr = st.number_input("NLR", 0.5, 15.0, 3.0)
            ldh = st.number_input("LDH (U/L)", 100, 600, 200)
            crp = st.number_input("CRP (mg/L)", 1, 150, 10)
            alb = st.number_input("Albumin (g/L)", 25, 50, 38)
        
        submitted = st.form_submit_button("Predict", use_container_width=True)
        
        if submitted:
            input_df = pd.DataFrame([{
                'Age': age, 'ECOG': ecog, 'Dose': dose, 'Prior_Therapies': prior,
                'Metastasis_Sites': mets, 'Liver_Mets': 1 if liver else 0,
                'PDL1': pdl1, 'TMB': tmb, 'NLR': nlr, 'LDH': ldh, 'CRP': crp, 'Albumin': alb
            }])
            
            resp_prob, ae_prob = st.session_state.model.predict(input_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Response Probability", f"{resp_prob*100:.1f}%")
                if resp_prob > 0.5:
                    st.success("✅ High probability")
                elif resp_prob > 0.3:
                    st.warning("⚠️ Medium probability")
                else:
                    st.error("❌ Low probability")
                    
            with col2:
                st.metric("AE Risk", f"{ae_prob*100:.1f}%")
                if ae_prob < 0.3:
                    st.success("✅ Low risk")
                elif ae_prob < 0.6:
                    st.warning("⚠️ Medium risk")
                else:
                    st.error("❌ High risk")
            
            if resp_prob > 0.5 and ae_prob < 0.4:
                st.success("✅ Recommended for MINIC3 therapy")
            elif resp_prob > 0.3:
                st.warning("⚠️ Use with caution")
            else:
                st.error("❌ Not recommended")

# Model Performance
elif page == "📈 Model Performance":
    st.markdown('<div class="sub-header">📈 Model Performance</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["ROC Curves", "Feature Importance", "Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['response']['fpr'],
                y=st.session_state.model.roc_data['response']['tpr'],
                mode='lines', name=f"Response (AUC={st.session_state.metrics['response']['auc']:.3f})",
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
            fig.update_layout(title="Response Prediction ROC")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['ae']['fpr'],
                y=st.session_state.model.roc_data['ae']['tpr'],
                mode='lines', name=f"AE (AUC={st.session_state.metrics['ae']['auc']:.3f})",
                line=dict(color='red', width=3)
            ))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
            fig.update_layout(title="AE Prediction ROC")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(st.session_state.model.importance_df.head(10),
                    x='Importance', y='Feature', orientation='h',
                    title='Feature Importance',
                    color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Response Model**")
            st.json({
                "Accuracy": f"{st.session_state.metrics['response']['accuracy']:.3f}",
                "Precision": f"{st.session_state.metrics['response']['precision']:.3f}",
                "Recall": f"{st.session_state.metrics['response']['recall']:.3f}",
                "F1": f"{st.session_state.metrics['response']['f1']:.3f}",
                "AUC": f"{st.session_state.metrics['response']['auc']:.3f}",
                "5-fold CV": f"{st.session_state.metrics['response']['cv_mean']:.3f} (±{st.session_state.metrics['response']['cv_std']:.3f})"
            })
            
        with col2:
            st.markdown("**AE Model**")
            st.json({
                "Accuracy": f"{st.session_state.metrics['ae']['accuracy']:.3f}",
                "Precision": f"{st.session_state.metrics['ae']['precision']:.3f}",
                "Recall": f"{st.session_state.metrics['ae']['recall']:.3f}",
                "F1": f"{st.session_state.metrics['ae']['f1']:.3f}",
                "AUC": f"{st.session_state.metrics['ae']['auc']:.3f}",
                "5-fold CV": f"{st.session_state.metrics['ae']['cv_mean']:.3f} (±{st.session_state.metrics['ae']['cv_std']:.3f})"
            })

# Survival Analysis
elif page == "📉 Survival Analysis":
    st.markdown('<div class="sub-header">📉 Survival Analysis</div>', unsafe_allow_html=True)
    
    group = st.selectbox("Group by", ["Dose", "PDL1", "ECOG", "Risk_Group"])
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    for val in df[group].unique():
        data = df[df[group]==val]['PFS']
        times = np.sort(data.unique())
        survival = [(data >= t).mean() for t in times]
        ax.step(times, survival, where='post', label=str(val), linewidth=2)
    
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'Kaplan-Meier Curves by {group}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.subheader("Median PFS")
    cols = st.columns(len(df[group].unique()))
    for i, val in enumerate(df[group].unique()):
        with cols[i]:
            median = df[df[group]==val]['PFS'].median()
            st.metric(str(val), f"{median:.1f} months")

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: gray;'>© 2024 MINIC3 Prediction System | Last updated: {datetime.now().strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
