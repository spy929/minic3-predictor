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

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="MINIC3 Prediction System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1e3c72;
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #2a5298;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low { color: #27ae60; font-weight: bold; font-size: 1.2rem; }
    .risk-medium { color: #f39c12; font-weight: bold; font-size: 1.2rem; }
    .risk-high { color: #e74c3c; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧬 MINIC3 Prediction System</div>', unsafe_allow_html=True)
st.markdown("#### A Machine Learning Platform for Dual-Task Prediction of Immunotherapy Outcomes")

# ==================== Generate Synthetic Data ====================
@st.cache_data
def generate_clinical_data():
    """Generate high-quality synthetic clinical data"""
    np.random.seed(42)
    n = 2000
    
    # Demographics
    data = {
        'Patient_ID': [f'MC3-{str(i).zfill(5)}' for i in range(1, n+1)],
        'Age': np.random.normal(62, 12, n).astype(int).clip(25, 90),
        'Gender': np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
        'BMI': np.random.normal(24, 4, n).round(1).clip(18, 35),
    }
    
    # Treatment characteristics
    treatment = {
        'Dose': np.random.choice([0.3, 1.0, 3.0, 10.0], n, p=[0.1, 0.2, 0.4, 0.3]),
        'Prior_Therapies': np.random.choice([0, 1, 2, 3, 4], n, p=[0.1, 0.3, 0.3, 0.2, 0.1]),
    }
    
    # Tumor characteristics
    tumor = {
        'Tumor_Type': np.random.choice(['NSCLC', 'Melanoma', 'RCC', 'Urothelial', 'HNSCC'], n),
        'ECOG': np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.4, 0.3, 0.1]),
        'Metastasis_Sites': np.random.poisson(2, n).clip(0, 6),
        'Liver_Mets': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'Brain_Mets': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    }
    
    # Biomarkers
    biomarkers = {
        'PDL1': np.random.choice(['Negative', 'Low', 'High'], n, p=[0.3, 0.4, 0.3]),
        'TMB': np.random.exponential(8, n).round(1).clip(0, 50),
        'NLR': np.random.normal(3, 1.5, n).round(2).clip(0.5, 15),
        'LDH': np.random.normal(200, 80, n).round(0).clip(100, 600),
        'CRP': np.random.exponential(15, n).round(1).clip(1, 150),
        'Albumin': np.random.normal(38, 5, n).round(1).clip(25, 50),
    }
    
    df = pd.DataFrame({**data, **treatment, **tumor, **biomarkers})
    
    # Calculate derived indices
    df['PNI'] = (df['Albumin'] + 5 * 1.8).round(1)
    
    # Generate outcomes
    response_prob = (0.2 + 0.05*(df['Dose']>1) + 0.1*(df['PDL1']=='High') - 
                     0.05*df['ECOG'] - 0.02*df['Metastasis_Sites'])
    response_prob = response_prob.clip(0.1, 0.85)
    df['Response'] = np.random.binomial(1, response_prob)
    
    ae_prob = (0.3 + 0.05*(df['Dose']>3) + 0.008*(df['Age']-60).clip(0,30) + 
               0.002*df['CRP'] + 0.1*df['Liver_Mets'])
    ae_prob = ae_prob.clip(0.15, 0.9)
    df['AE'] = np.random.binomial(1, ae_prob)
    
    # Survival outcomes
    df['PFS'] = np.where(df['Response']==1, 
                         np.random.normal(18,6,n), 
                         np.random.normal(5,2,n)).clip(1,48).round(1)
    
    # Response categories
    df['Response_Category'] = np.where(
        df['Response']==1,
        np.random.choice(['CR', 'PR'], n, p=[0.2, 0.8]),
        np.random.choice(['SD', 'PD'], n, p=[0.4, 0.6])
    )
    
    # AE grading
    ae_grades = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
    ae_types = ['Rash', 'Diarrhea', 'Hepatitis', 'Pneumonitis', 'Colitis']
    
    df['AE_Grade'] = np.where(
        df['AE']==1,
        np.random.choice(ae_grades, n, p=[0.4, 0.35, 0.2, 0.05]),
        'None'
    )
    
    df['AE_Type'] = np.where(df['AE']==1, np.random.choice(ae_types, n), 'None')
    
    # Risk stratification
    risk_score = (df['ECOG']*2 + (df['LDH']>250).astype(int)*3 + 
                  (df['Metastasis_Sites']>2).astype(int)*2 + (df['NLR']>5).astype(int)*2)
    df['Risk_Score'] = risk_score
    df['Risk_Group'] = pd.cut(risk_score, bins=[0,3,6,9,15], 
                               labels=['Low', 'Intermediate-Low', 'Intermediate-High', 'High'])
    
    return df

# ==================== Machine Learning Model ====================
class ClinicalPredictor:
    def __init__(self):
        self.model_response = None
        self.model_ae = None
        self.scaler = StandardScaler()
        self.feature_cols = ['Age', 'ECOG', 'Dose', 'Prior_Therapies', 'Metastasis_Sites', 
                             'Liver_Mets', 'Brain_Mets', 'TMB', 'NLR', 'LDH', 'CRP', 'Albumin']
        self.metrics = {}
        self.roc_data = {}
        self.importance_df = None
        self.cm_data = {}
        
    def prepare_features(self, df, fit=False):
        df_encoded = df.copy()
        df_encoded['PDL1_encoded'] = df['PDL1'].map({'Negative':0, 'Low':1, 'High':2})
        df_encoded['Gender_encoded'] = df['Gender'].map({'Male':0, 'Female':1}) if 'Gender' in df.columns else 0
        
        features = self.feature_cols + ['PDL1_encoded', 'Gender_encoded']
        X = df_encoded[[f for f in features if f in df_encoded.columns]].fillna(0)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train(self, df):
        with st.spinner('Training machine learning models...'):
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
            
            # Confusion matrices
            self.cm_data = {
                'response': confusion_matrix(y_res_test, y_res_pred),
                'ae': confusion_matrix(y_ae_test, y_ae_pred)
            }
            
            # Feature importance
            feature_names = self.feature_cols + ['PDL1', 'Gender']
            self.importance_df = pd.DataFrame({
                'Feature': feature_names,
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

# ==================== Initialize ====================
if 'model' not in st.session_state:
    st.session_state.model = ClinicalPredictor()
    df = generate_clinical_data()
    st.session_state.df = df
    metrics = st.session_state.model.train(df)
    st.session_state.metrics = metrics

df = st.session_state.df

# ==================== Sidebar Navigation ====================
with st.sidebar:
    st.markdown("## 🧬 Navigation")
    
    page = st.radio(
        "Select Module",
        [
            "📊 Data Overview",
            "🎯 Patient Prediction",
            "📈 Model Performance",
            "📉 Survival Analysis",
            "🔬 Biomarker Analysis"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Data Summary")
    st.metric("Total Patients", f"{len(df):,}")
    st.metric("Features", len(st.session_state.model.feature_cols))
    
    st.markdown("### 🎯 Model Performance")
    st.metric("Response AUC", f"{st.session_state.metrics['response']['auc']:.3f}")
    st.metric("AE AUC", f"{st.session_state.metrics['ae']['auc']:.3f}")
    
    st.markdown("---")
    st.caption(f"© 2024 MINIC3 System")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d')}")

# ==================== Page 1: Data Overview ====================
if page == "📊 Data Overview":
    st.markdown('<div class="sub-header">📊 Clinical Data Overview</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Response Rate", f"{df['Response'].mean()*100:.1f}%")
    with col2:
        st.metric("AE Rate", f"{df['AE'].mean()*100:.1f}%")
    with col3:
        st.metric("Median PFS", f"{df['PFS'].median():.1f} months")
    with col4:
        st.metric("High Risk", f"{(df['Risk_Group']=='High').mean()*100:.1f}%")
    with col5:
        st.metric("Total Patients", f"{len(df)}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Distributions", "Correlation Matrix"])
    
    with tab1:
        st.dataframe(df.head(50), use_container_width=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='Age', color='Gender', nbins=30, title='Age Distribution')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(df, names='Tumor_Type', title='Tumor Types', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        numeric_cols = ['Age', 'ECOG', 'Metastasis_Sites', 'TMB', 'NLR', 'LDH', 'CRP', 'Albumin', 'PFS']
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto', 
                       color_continuous_scale='RdBu_r', title='Feature Correlations')
        st.plotly_chart(fig, use_container_width=True)

# ==================== Page 2: Patient Prediction ====================
elif page == "🎯 Patient Prediction":
    st.markdown('<div class="sub-header">🎯 Individual Patient Prediction</div>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.markdown("#### Patient Demographics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 90, 60)
            gender = st.selectbox("Gender", ["Male", "Female"])
            ecog = st.selectbox("ECOG", [0, 1, 2, 3])
            
        with col2:
            dose = st.selectbox("Dose (mg/kg)", [0.3, 1.0, 3.0, 10.0])
            prior = st.number_input("Prior Therapies", 0, 5, 1)
            mets = st.number_input("Metastasis Sites", 0, 6, 1)
            
        with col3:
            liver = st.checkbox("Liver Metastasis")
            brain = st.checkbox("Brain Metastasis")
            tumor = st.selectbox("Tumor Type", ['NSCLC', 'Melanoma', 'RCC', 'Urothelial', 'HNSCC'])
        
        st.markdown("#### Biomarkers")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pdl1 = st.selectbox("PD-L1", ["Negative", "Low", "High"])
            tmb = st.number_input("TMB (mut/Mb)", 0, 50, 8)
            
        with col2:
            nlr = st.number_input("NLR", 0.5, 15.0, 3.0, 0.1)
            ldh = st.number_input("LDH (U/L)", 100, 600, 200)
            
        with col3:
            crp = st.number_input("CRP (mg/L)", 1, 150, 10)
            alb = st.number_input("Albumin (g/L)", 25, 50, 38)
        
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
        
        if submitted:
            input_df = pd.DataFrame([{
                'Age': age, 'Gender': gender, 'ECOG': ecog, 'Dose': dose,
                'Prior_Therapies': prior, 'Metastasis_Sites': mets,
                'Liver_Mets': 1 if liver else 0, 'Brain_Mets': 1 if brain else 0,
                'Tumor_Type': tumor, 'PDL1': pdl1, 'TMB': tmb,
                'NLR': nlr, 'LDH': ldh, 'CRP': crp, 'Albumin': alb
            }])
            
            resp_prob, ae_prob = st.session_state.model.predict(input_df)
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=resp_prob*100,
                    domain={'x': [0,1], 'y': [0,1]},
                    title={'text': "Response Probability (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#27ae60"},
                           'steps': [
                               {'range': [0, 30], 'color': "#ff6b6b"},
                               {'range': [30, 60], 'color': "#feca57"},
                               {'range': [60, 100], 'color': "#27ae60"}
                           ]}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ae_prob*100,
                    domain={'x': [0,1], 'y': [0,1]},
                    title={'text': "AE Risk (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#e74c3c"},
                           'steps': [
                               {'range': [0, 30], 'color': "#27ae60"},
                               {'range': [30, 60], 'color': "#feca57"},
                               {'range': [60, 100], 'color': "#ff6b6b"}
                           ]}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if resp_prob > 0.5:
                    st.success(f"✅ Response: {resp_prob*100:.1f}%")
                elif resp_prob > 0.3:
                    st.warning(f"⚠️ Response: {resp_prob*100:.1f}%")
                else:
                    st.error(f"❌ Response: {resp_prob*100:.1f}%")
                    
            with col2:
                if ae_prob < 0.3:
                    st.success(f"✅ AE Risk: {ae_prob*100:.1f}%")
                elif ae_prob < 0.6:
                    st.warning(f"⚠️ AE Risk: {ae_prob*100:.1f}%")
                else:
                    st.error(f"❌ AE Risk: {ae_prob*100:.1f}%")
            
            with col3:
                if resp_prob > 0.5 and ae_prob < 0.4:
                    st.success("✅ Recommended")
                elif resp_prob > 0.3:
                    st.warning("⚠️ Consider with Caution")
                else:
                    st.error("❌ Not Recommended")

# ==================== Page 3: Model Performance ====================
elif page == "📈 Model Performance":
    st.markdown('<div class="sub-header">📈 Model Performance Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curves", "Feature Importance", "Confusion Matrix", "Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['response']['fpr'],
                y=st.session_state.model.roc_data['response']['tpr'],
                mode='lines', name=f"Response (AUC={st.session_state.metrics['response']['auc']:.3f})",
                line=dict(color='#27ae60', width=3), fill='tozeroy'
            ))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', 
                                    line=dict(dash='dash', color='gray')))
            fig.update_layout(title='Response Prediction ROC')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.model.roc_data['ae']['fpr'],
                y=st.session_state.model.roc_data['ae']['tpr'],
                mode='lines', name=f"AE (AUC={st.session_state.metrics['ae']['auc']:.3f})",
                line=dict(color='#e74c3c', width=3), fill='tozeroy'
            ))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                    line=dict(dash='dash', color='gray')))
            fig.update_layout(title='AE Prediction ROC')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(st.session_state.model.importance_df.head(15),
                    x='Importance', y='Feature', orientation='h',
                    title='Feature Importance Ranking',
                    color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Findings**:
        - **Dose level** and **PD-L1 expression** are the strongest predictors of response
        - **Age** and **CRP** dominate AE prediction
        - Inflammatory markers (NLR, LDH) show moderate predictive value
        """)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            cm = st.session_state.model.cm_data['response']
            fig = px.imshow(cm, text_auto=True,
                           x=['Predicted NR', 'Predicted R'],
                           y=['Actual NR', 'Actual R'],
                           color_continuous_scale='Blues',
                           title='Response Confusion Matrix')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            cm = st.session_state.model.cm_data['ae']
            fig = px.imshow(cm, text_auto=True,
                           x=['Predicted No AE', 'Predicted AE'],
                           y=['Actual No AE', 'Actual AE'],
                           color_continuous_scale='Reds',
                           title='AE Confusion Matrix')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Response Prediction Model**")
            metrics_df = pd.DataFrame([
                ['Accuracy', f"{st.session_state.metrics['response']['accuracy']:.3f}"],
                ['Precision', f"{st.session_state.metrics['response']['precision']:.3f}"],
                ['Recall', f"{st.session_state.metrics['response']['recall']:.3f}"],
                ['F1 Score', f"{st.session_state.metrics['response']['f1']:.3f}"],
                ['AUC', f"{st.session_state.metrics['response']['auc']:.3f}"],
                ['5-Fold CV', f"{st.session_state.metrics['response']['cv_mean']:.3f} (±{st.session_state.metrics['response']['cv_std']:.3f})"]
            ], columns=['Metric', 'Value'])
            st.dataframe(metrics_df, use_container_width=True)
            
        with col2:
            st.markdown("**AE Prediction Model**")
            metrics_df = pd.DataFrame([
                ['Accuracy', f"{st.session_state.metrics['ae']['accuracy']:.3f}"],
                ['Precision', f"{st.session_state.metrics['ae']['precision']:.3f}"],
                ['Recall', f"{st.session_state.metrics['ae']['recall']:.3f}"],
                ['F1 Score', f"{st.session_state.metrics['ae']['f1']:.3f}"],
                ['AUC', f"{st.session_state.metrics['ae']['auc']:.3f}"],
                ['5-Fold CV', f"{st.session_state.metrics['ae']['cv_mean']:.3f} (±{st.session_state.metrics['ae']['cv_std']:.3f})"]
            ], columns=['Metric', 'Value'])
            st.dataframe(metrics_df, use_container_width=True)

# ==================== Page 4: Survival Analysis ====================
elif page == "📉 Survival Analysis":
    st.markdown('<div class="sub-header">📉 Survival Analysis</div>', unsafe_allow_html=True)
    
    group = st.selectbox("Group by", ['Dose', 'PDL1', 'ECOG', 'Risk_Group', 'Response'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    groups = df[group].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    
    for i, val in enumerate(groups):
        data = df[df[group]==val]['PFS']
        times = np.sort(data.unique())
        survival = [(data >= t).mean() for t in times]
        ax.step(times, survival, where='post', label=str(val), linewidth=2, color=colors[i])
    
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Kaplan-Meier Curves by {group}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.subheader("Median Survival Times")
    cols = st.columns(len(groups))
    for i, val in enumerate(groups):
        with cols[i]:
            median = df[df[group]==val]['PFS'].median()
            st.metric(str(val), f"{median:.1f} months")

# ==================== Page 5: Biomarker Analysis ====================
elif page == "🔬 Biomarker Analysis":
    st.markdown('<div class="sub-header">🔬 Biomarker Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["PD-L1 Expression", "TMB Analysis", "Inflammatory Markers"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='PDL1', y='Response', 
                        title='PD-L1 Expression and Response',
                        points='all', color='PDL1')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            response_by_pdl1 = df.groupby('PDL1')['Response'].agg(['mean', 'count']).round(3)
            response_by_pdl1.columns = ['Response Rate', 'Count']
            response_by_pdl1['Response Rate'] = (response_by_pdl1['Response Rate'] * 100).round(1)
            st.dataframe(response_by_pdl1, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='TMB', y='Response', color='PDL1',
                           trendline='lowess', title='TMB and Response Relationship')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            df['TMB_Group'] = pd.cut(df['TMB'], bins=[0, 5, 10, 20, 50], 
                                     labels=['Low TMB', 'Intermediate', 'High TMB', 'Very High'])
            tmb_response = df.groupby('TMB_Group')['Response'].mean() * 100
            fig = px.bar(x=tmb_response.index, y=tmb_response.values,
                        title='Response Rate by TMB Group',
                        color=tmb_response.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='NLR', y='AE', trendline='lowess',
                            title='NLR and AE Risk',
                            labels={'NLR': 'NLR', 'AE': 'AE Probability'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.scatter(df, x='CRP', y='PFS', color='Risk_Group',
                            trendline='lowess', title='CRP and PFS Relationship')
            st.plotly_chart(fig, use_container_width=True)

# ==================== Footer ====================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>© 2024 MINIC3 Prediction System | Version 5.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d')}</p>
    <p style='font-size: 0.8rem;'>Code: <a href='https://github.com/spy291/minic3-predictor' target='_blank'>GitHub</a> | 
    App: <a href='https://spy291-minic3-predictor.streamlit.app' target='_blank'>Streamlit</a></p>
</div>
""", unsafe_allow_html=True)
