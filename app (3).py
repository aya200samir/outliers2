# -*- coding: utf-8 -*-
"""
===========================================================================
🛡️ AI ADMINISTRATIVE AUDIT & CORRUPTION DETECTION SYSTEM
===========================================================================
نظام متكامل لتحليل البيانات المالية والإدارية، كشف الفساد والرشوة، 
وتحليل الأحكام القانونية مع تقنيات الذكاء الاصطناعي القابل للتفسير

الإصدار: 1.0 (Professional Edition)
المطور: النظام الذكي للرقابة الإدارية
===========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import warnings
import os
import re
import io
import base64
from datetime import datetime
import time
from collections import Counter
import hashlib
import json

warnings.filterwarnings('ignore')

# ==================== مكتبات التعلم الآلي ====================
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# XGBoost للتنبؤ المتقدم
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

# SHAP لتفسير النماذج
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# ==================== مكتبات معالجة النصوص العربية ====================
try:
    from wordcloud import WordCloud, STOPWORDS
    import arabic_reshaper
    from bidi.algorithm import get_display
    import PyPDF2
    from textblob import TextBlob
    TEXT_ANALYSIS_AVAILABLE = True
except:
    TEXT_ANALYSIS_AVAILABLE = False

# ==================== إعدادات الصفحة المتقدمة ====================
st.set_page_config(
    page_title="AI Audit & Corruption Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.ai-audit-system.com',
        'Report a bug': "https://github.com/ai-audit/issues",
        'About': "# AI Administrative Audit System\nالإصدار الاحترافي 1.0"
    }
)

# ==================== CSS احترافي متطور ====================
PROFESSIONAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* الأساسيات */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* خلفية داكنة متدرجة */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', 'Cairo', sans-serif;
    }
    
    /* هيدر رئيسي بتأثير زجاجي */
    .main-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 0 0 40px 40px;
        padding: 3rem 2rem;
        margin-bottom: 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 255, 136, 0.1) 0%, transparent 70%);
        animation: rotate 30s linear infinite;
        z-index: 0;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00ff88 0%, #00cc88 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 30px rgba(0, 255, 136, 0.3); }
        50% { text-shadow: 0 0 50px rgba(0, 255, 136, 0.6); }
    }
    
    .main-header p {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.8);
        max-width: 800px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    /* كروت زجاجية متطورة */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.02), transparent);
        transition: left 0.8s;
    }
    
    .glass-card:hover::after {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(0, 255, 136, 0.3);
        box-shadow: 0 20px 40px rgba(0, 255, 136, 0.1);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid rgba(0, 255, 136, 0.2);
        padding-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* مقاييس نيون متألقة */
    .metric-neon {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 200, 100, 0.05) 100%);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    
    .metric-neon::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00ff88, transparent, #00ff88);
        border-radius: 22px;
        z-index: -1;
        animation: borderGlow 3s linear infinite;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .metric-neon:hover::before {
        opacity: 1;
    }
    
    @keyframes borderGlow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .metric-neon:hover {
        transform: scale(1.05);
        border-color: #00ff88;
    }
    
    .metric-neon-value {
        font-size: 3rem;
        font-weight: 900;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        line-height: 1.2;
    }
    
    .metric-neon-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    
    /* شارات متخصصة */
    .badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 1px;
        margin: 0.3rem;
        transition: all 0.3s;
    }
    
    .badge:hover {
        transform: translateY(-2px);
        filter: brightness(1.2);
    }
    
    .badge-primary {
        background: linear-gradient(135deg, #00ff88, #00cc88);
        color: #0f172a;
        box-shadow: 0 5px 15px rgba(0, 255, 136, 0.3);
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #ff4b4b, #dc2626);
        color: white;
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.3);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #fbbf24, #d97706);
        color: #0f172a;
        box-shadow: 0 5px 15px rgba(251, 191, 36, 0.3);
    }
    
    .badge-info {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* أزرار نيون متطورة */
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc88 100%);
        color: #0f172a;
        border: none;
        border-radius: 14px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s;
        box-shadow: 0 8px 20px rgba(0, 255, 136, 0.3);
        position: relative;
        overflow: hidden;
        width: 100%;
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transform: rotate(45deg);
        animation: buttonShine 3s infinite;
    }
    
    @keyframes buttonShine {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 30px rgba(0, 255, 136, 0.5);
    }
    
    /* تبويبات متطورة */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.8rem;
        background: rgba(255, 255, 255, 0.03);
        padding: 0.5rem;
        border-radius: 60px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.6);
        border: none;
        transition: all 0.3s;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ff88, #00cc88) !important;
        color: #0f172a !important;
        box-shadow: 0 8px 20px rgba(0, 255, 136, 0.4);
        font-weight: 700;
    }
    
    /* شريط جانبي متطور */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .sidebar-content {
        padding: 2rem 1rem;
    }
    
    /* شريط التقدم */
    .progress-container {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #00ff88, #00cc88);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* تذييل احترافي */
    .footer {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-top: 1px solid rgba(0, 255, 136, 0.2);
        padding: 3rem 2rem;
        margin-top: 4rem;
        text-align: center;
        border-radius: 40px 40px 0 0;
        position: relative;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ff88, transparent);
    }
    
    .footer h3 {
        color: white;
        font-size: 2rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #00ff88, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .footer p {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1rem;
    }
    
    /* أنيميشن للعناصر */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .float-animation {
        animation: float 4s ease-in-out infinite;
    }
    
    /* تنسيق الجداول */
    .dataframe {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .dataframe th {
        background: rgba(0, 255, 136, 0.1);
        color: #00ff88;
        font-weight: 600;
        padding: 1rem;
    }
    
    .dataframe td {
        color: rgba(255, 255, 255, 0.8);
        padding: 0.8rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* تنسيق التنبيهات */
    .alert {
        padding: 1.2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid;
        backdrop-filter: blur(10px);
    }
    
    .alert-success {
        background: rgba(0, 255, 136, 0.1);
        border-color: rgba(0, 255, 136, 0.3);
        color: #00ff88;
    }
    
    .alert-warning {
        background: rgba(251, 191, 36, 0.1);
        border-color: rgba(251, 191, 36, 0.3);
        color: #fbbf24;
    }
    
    .alert-danger {
        background: rgba(255, 75, 75, 0.1);
        border-color: rgba(255, 75, 75, 0.3);
        color: #ff4b4b;
    }
    
    .alert-info {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
        color: #3b82f6;
    }
    
    /* تحسين ظهور النصوص العربية */
    .arabic-text {
        direction: rtl;
        font-family: 'Cairo', sans-serif;
    }
</style>
"""

# ==================== تطبيق CSS ====================
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ==================== تهيئة حالة الجلسة ====================
def init_session_state():
    """تهيئة جميع متغيرات الجلسة"""
    defaults = {
        'data_loaded': False,
        'model_trained': False,
        'df': None,
        'df_original': None,
        'anomalies': None,
        'model_pack': None,
        'bias_report': None,
        'predictions': None,
        'shap_values': None,
        'legal_texts': [],
        'analysis_history': [],
        'theme': 'dark',
        'processing_time': 0,
        'file_info': {},
        'corruption_cases': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== دوال معالجة البيانات ====================

def detect_data_quality(df):
    """تحليل جودة البيانات واكتشاف المشكلات"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'data_types': df.dtypes.value_counts().to_dict(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'columns_info': {}
    }
    
    # تحليل كل عمود
    for col in df.columns:
        col_info = {
            'type': str(df[col].dtype),
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'unique': df[col].nunique()
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'skew': float(df[col].skew()) if not pd.isna(df[col].skew()) else None
            })
        
        report['columns_info'][col] = col_info
    
    return report

def clean_dataframe(df):
    """تنظيف البيانات بشكل ذكي"""
    df_clean = df.copy()
    
    # 1. إزالة الصفوف المكررة
    initial_len = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    
    # 2. معالجة القيم المفقودة
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            # للأعمدة الرقمية: تعبئة بالوسيط
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        else:
            # للأعمدة النصية: تعبئة بالقيمة الأكثر تكراراً
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
    
    # 3. إزالة القيم المتطرفة (للأعمدة الرقمية)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    removed_rows = initial_len - len(df_clean)
    
    return df_clean, removed_rows

def extract_text_from_pdf(pdf_file):
    """استخراج النصوص من ملف PDF"""
    if not TEXT_ANALYSIS_AVAILABLE:
        return ["مكتبات تحليل النصوص غير متوفرة"]
    
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.split('\n')
    except Exception as e:
        return [f"خطأ في قراءة PDF: {str(e)}"]

# ==================== دوال كشف الفساد والشذوذ ====================

def detect_fraud_patterns(df):
    """كشف أنماط الاحتيال والفساد المتقدمة"""
    fraud_report = {
        'total_cases': len(df),
        'suspicious_cases': 0,
        'fraud_indicators': [],
        'high_risk_transactions': [],
        'corruption_score': 0,
        'patterns': []
    }
    
    # البحث عن مؤشرات الفساد
    indicators = []
    
    # 1. تحليل المعاملات المالية
    if 'amount' in df.columns or 'المبلغ' in df.columns:
        amount_col = 'amount' if 'amount' in df.columns else 'المبلغ'
        
        # معاملات أكبر من 3 انحرافات معيارية
        mean_amount = df[amount_col].mean()
        std_amount = df[amount_col].std()
        threshold = mean_amount + 3 * std_amount
        
        large_transactions = df[df[amount_col] > threshold]
        if len(large_transactions) > 0:
            indicators.append({
                'type': 'large_transactions',
                'count': len(large_transactions),
                'description': f'معاملات كبيرة الحجم (أكبر من {threshold:,.2f})'
            })
            fraud_report['high_risk_transactions'].extend(large_transactions.index.tolist())
    
    # 2. تحليل التكرار (معاملات متكررة)
    if 'user_id' in df.columns or 'client_id' in df.columns:
        id_col = 'user_id' if 'user_id' in df.columns else 'client_id' if 'client_id' in df.columns else None
        if id_col:
            freq = df[id_col].value_counts()
            very_frequent = freq[freq > freq.quantile(0.95)]
            if len(very_frequent) > 0:
                indicators.append({
                    'type': 'frequent_users',
                    'count': len(very_frequent),
                    'description': 'مستخدمون بتكرار غير طبيعي'
                })
    
    # 3. تحليل التوقيت (معاملات في أوقات غير معتادة)
    if 'timestamp' in df.columns or 'date' in df.columns:
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df['hour'] = df[date_col].dt.hour
            unusual_hours = df[(df['hour'] < 6) | (df['hour'] > 22)]
            if len(unusual_hours) > 0:
                indicators.append({
                    'type': 'unusual_hours',
                    'count': len(unusual_hours),
                    'description': 'معاملات في ساعات متأخرة'
                })
        except:
            pass
    
    fraud_report['fraud_indicators'] = indicators
    fraud_report['suspicious_cases'] = len(set().union(*[ind.get('count', 0) for ind in indicators]))
    fraud_report['corruption_score'] = min(fraud_report['suspicious_cases'] / len(df) * 100, 100)
    
    return fraud_report

def detect_anomalies_advanced(df, contamination=0.1):
    """كشف متقدم للشذوذ باستخدام تقنيات متعددة"""
    
    # تجهيز البيانات الرقمية
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
    
    if len(numeric_df.columns) == 0:
        return None, None
    
    # توحيد المقاييس
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    
    # 1. Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    iso_pred = iso_forest.fit_predict(X_scaled)
    
    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20
    )
    lof_pred = lof.fit_predict(X_scaled)
    
    # 3. DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_pred = dbscan.fit_predict(X_scaled)
    dbscan_outliers = (dbscan_pred == -1).astype(int)
    
    # دمج النتائج (التصويت)
    ensemble_score = (iso_pred + lof_pred + dbscan_outliers) / 3
    ensemble_score = (ensemble_score + 1) / 2  # تطبيع إلى [0, 1]
    
    # إنشاء DataFrame بالنتائج
    results = df.copy()
    results['anomaly_score_iso'] = (iso_pred == -1).astype(int)
    results['anomaly_score_lof'] = (lof_pred == -1).astype(int)
    results['anomaly_score_dbscan'] = dbscan_outliers
    results['anomaly_score_ensemble'] = ensemble_score
    
    # تحديد الشاذ بناءً على متوسط الدرجات
    results['is_anomaly'] = results[['anomaly_score_iso', 'anomaly_score_lof', 'anomaly_score_dbscan']].mean(axis=1) > 0.5
    
    return results, numeric_df.columns.tolist()

# ==================== دوال التنبؤ بالفساد ====================

def train_corruption_model(df, target_col=None):
    """تدريب نموذج للتنبؤ بالفساد"""
    
    if target_col is None:
        # البحث عن عمود مناسب كهدف
        possible_targets = ['fraud', 'corruption', 'churn', 'default', 'risk', 'label', 'class']
        for col in df.columns:
            if any(target in col.lower() for target in possible_targets):
                target_col = col
                break
    
    if target_col is None:
        # إذا لم يتم العثور على هدف، استخدم نتائج كشف الشذوذ كهدف
        return None, "لم يتم العثور على عمود هدف للتدريب"
    
    # تجهيز الميزات
    feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
    
    if len(feature_cols) == 0:
        return None, "لا توجد ميزات رقمية كافية"
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # تحويل الهدف إلى قيم ثنائية إذا كان نصياً
    if y.dtype == 'object':
        y = (y == y.mode()[0]).astype(int)
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # تدريب النموذج
    if XGB_AVAILABLE:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    
    # تقييم النموذج
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    # أهمية الميزات
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    result = {
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return result, None

# ==================== دوال تحليل النصوص القانونية ====================

def analyze_legal_text(texts):
    """تحليل النصوص القانونية"""
    
    if not TEXT_ANALYSIS_AVAILABLE or not texts:
        return {"error": "مكتبات تحليل النصوص غير متوفرة"}
    
    results = {}
    
    # دمج النصوص
    full_text = ' '.join(texts)
    
    # تنظيف النص العربي
    import re
    full_text = re.sub(r'[^\w\s]', '', full_text)
    full_text = re.sub(r'\d+', '', full_text)
    
    # كلمات التوقف العربية
    arabic_stopwords = set(['في', 'من', 'إلى', 'على', 'كان', 'هذا', 'أن', 
                            'قد', 'لا', 'ما', 'هل', 'لم', 'لقد', 'إن',
                            'عند', 'مع', 'هذه', 'ذلك', 'يمكن', 'سوف'])
    
    # تحليل التكرارات
    words = [w for w in full_text.split() if len(w) > 2 and w not in arabic_stopwords]
    word_counts = Counter(words).most_common(30)
    results['top_words'] = word_counts
    
    # إنشاء Word Cloud
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        
        # إعادة تشكيل النص العربي
        reshaped_text = arabic_reshaper.reshape(full_text)
        bidi_text = get_display(reshaped_text)
        
        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color='black',
            colormap='Greens',
            max_words=100,
            random_state=42
        ).generate(bidi_text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('الكلمات الأكثر تكراراً في النصوص القانونية', color='white', fontsize=16)
        plt.tight_layout()
        
        results['wordcloud'] = fig
    except Exception as e:
        results['wordcloud_error'] = str(e)
    
    # تحليل المشاعر (إذا أمكن)
    try:
        blob = TextBlob(full_text)
        results['sentiment'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except:
        pass
    
    return results

# ==================== دوال التفسير (SHAP) ====================

def explain_predictions(model, X, feature_names):
    """تفسير تنبؤات النموذج باستخدام SHAP"""
    
    if not SHAP_AVAILABLE:
        return None, "SHAP غير متوفر"
    
    try:
        # إنشاء explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X[:100])
        
        # حساب قيم SHAP
        shap_values = explainer.shap_values(X[:100])
        
        return shap_values, None
    except Exception as e:
        return None, str(e)

# ==================== دوال التصور المتقدم ====================

def create_correlation_heatmap(df):
    """إنشاء خريطة حرارية للارتباطات"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return None
    
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='مصفوفة الارتباط بين المتغيرات',
        height=600,
        width=800,
        xaxis_title='المتغيرات',
        yaxis_title='المتغيرات'
    )
    
    return fig

def create_anomaly_dashboard(anomaly_df, original_df):
    """إنشاء لوحة تحكم متكاملة للشذوذ"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('توزيع الشذوذ', 'درجات الشذوذ', 'المعاملات المشبوهة', 'تحليل الميزات'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'heatmap'}]]
    )
    
    # 1. توزيع الشذوذ
    anomaly_counts = anomaly_df['is_anomaly'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=['طبيعي', 'شاذ'],
            values=[anomaly_counts.get(False, 0), anomaly_counts.get(True, 0)],
            marker=dict(colors=['#00ff88', '#ff4b4b']),
            textinfo='label+percent'
        ),
        row=1, col=1
    )
    
    # 2. درجات الشذوذ
    fig.add_trace(
        go.Bar(
            x=anomaly_df.index[:30],
            y=anomaly_df['anomaly_score_ensemble'][:30],
            marker_color=anomaly_df['anomaly_score_ensemble'][:30],
            marker_colorscale='RdYlGn_r',
            name='درجات الشذوذ'
        ),
        row=1, col=2
    )
    
    # 3. المعاملات المشبوهة
    if 'amount' in anomaly_df.columns or 'المبلغ' in anomaly_df.columns:
        amount_col = 'amount' if 'amount' in anomaly_df.columns else 'المبلغ'
        fig.add_trace(
            go.Scatter(
                x=anomaly_df.index[:50],
                y=anomaly_df[amount_col][:50],
                mode='markers',
                marker=dict(
                    size=anomaly_df['anomaly_score_ensemble'][:50] * 20,
                    color=anomaly_df['is_anomaly'][:50],
                    colorscale=[[0, '#00ff88'], [1, '#ff4b4b']],
                    showscale=True
                ),
                name='المعاملات'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="لوحة تحليل الشذوذ المتكاملة",
        title_font_size=20
    )
    
    return fig

# ==================== دوال واجهة المستخدم ====================

def display_header():
    """عرض الهيدر الرئيسي"""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ AI ADMINISTRATIVE AUDIT SYSTEM</h1>
        <p>نظام متكامل لكشف الفساد المالي والإداري وتحليل الأحكام القانونية باستخدام الذكاء الاصطناعي</p>
        <div style="margin-top: 2rem;">
            <span class="badge badge-primary">✨ ذكاء اصطناعي</span>
            <span class="badge badge-info">🔍 كشف الشذوذ</span>
            <span class="badge badge-warning">⚖️ تحليل قانوني</span>
            <span class="badge badge-danger">🚫 مكافحة فساد</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_metrics_card(title, value, subtitle, color='primary'):
    """عرض بطاقة مقاييس"""
    color_class = f"badge-{color}"
    st.markdown(f"""
    <div class="metric-neon">
        <div class="metric-neon-value">{value}</div>
        <div class="metric-neon-label">{title}</div>
        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: rgba(255,255,255,0.5);">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def display_alert(message, type='info'):
    """عرض تنبيه"""
    alert_class = f"alert-{type}"
    st.markdown(f"""
    <div class="alert {alert_class}">
        {message}
    </div>
    """, unsafe_allow_html=True)

# ==================== الصفحة الرئيسية ====================

def main():
    # عرض الهيدر
    display_header()
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #00ff88;">🔧 لوحة التحكم</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # رفع الملفات
        st.markdown("### 📁 رفع البيانات")
        data_file = st.file_uploader(
            "رفع ملف البيانات (CSV, Excel)",
            type=['csv', 'xlsx', 'xls'],
            key='data_uploader'
        )
        
        legal_file = st.file_uploader(
            "رفع الأحكام القانونية (PDF, TXT)",
            type=['pdf', 'txt'],
            key='legal_uploader'
        )
        
        if data_file is not None:
            if st.button("🚀 تحميل وتحليل البيانات", use_container_width=True):
                with st.spinner("جاري تحميل وتحليل البيانات..."):
                    try:
                        if data_file.name.endswith('.csv'):
                            df = pd.read_csv(data_file)
                        else:
                            df = pd.read_excel(data_file)
                        
                        st.session_state.df_original = df.copy()
                        st.session_state.file_info = {
                            'name': data_file.name,
                            'size': f"{data_file.size / 1024:.2f} KB",
                            'rows': len(df),
                            'columns': len(df.columns)
                        }
                        
                        # تنظيف البيانات
                        df_clean, removed = clean_dataframe(df)
                        st.session_state.df = df_clean
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ تم تحميل {len(df)} سجل وتنظيف {removed} صف")
                    except Exception as e:
                        st.error(f"خطأ في قراءة الملف: {str(e)}")
        
        if legal_file is not None:
            if st.button("📄 تحليل النصوص القانونية", use_container_width=True):
                with st.spinner("جاري تحليل النصوص..."):
                    if legal_file.name.endswith('.pdf'):
                        texts = extract_text_from_pdf(legal_file)
                    else:
                        texts = legal_file.getvalue().decode('utf-8').split('\n')
                    
                    st.session_state.legal_texts = texts
                    st.success(f"✅ تم تحميل {len(texts)} سطر نصي")
        
        st.markdown("---")
        
        # إعدادات التحليل
        if st.session_state.data_loaded:
            st.markdown("### ⚙️ إعدادات التحليل")
            
            contamination = st.slider(
                "حساسية كشف الشذوذ",
                min_value=0.01,
                max_value=0.3,
                value=0.05,
                step=0.01,
                help="نسبة الحالات المتوقعة كشاذة"
            )
            
            if st.button("🔍 كشف الشذوذ", use_container_width=True):
                with st.spinner("جاري تحليل البيانات..."):
                    anomalies_df, features = detect_anomalies_advanced(
                        st.session_state.df,
                        contamination=contamination
                    )
                    
                    if anomalies_df is not None:
                        st.session_state.anomalies = anomalies_df
                        
                        # تحليل أنماط الفساد
                        fraud_report = detect_fraud_patterns(anomalies_df)
                        st.session_state.fraud_report = fraud_report
                        
                        st.success(f"✅ تم اكتشاف {anomalies_df['is_anomaly'].sum()} حالة مشبوهة")
            
            if st.button("🤖 تدريب نموذج التنبؤ", use_container_width=True):
                with st.spinner("جاري تدريب النموذج..."):
                    model_result, error = train_corruption_model(st.session_state.df)
                    
                    if model_result is not None:
                        st.session_state.model_pack = model_result
                        st.success(f"✅ تم تدريب النموذج بدقة: {model_result['metrics']['accuracy']*100:.1f}%")
                    else:
                        st.warning(f"⚠️ {error}")
        
        st.markdown("---")
        
        # معلومات الملف
        if st.session_state.file_info:
            st.markdown("### 📊 معلومات الملف")
            info = st.session_state.file_info
            st.markdown(f"""
            <div style="background: rgba(0,255,136,0.05); padding: 1rem; border-radius: 12px;">
                <p><strong>الاسم:</strong> {info['name']}</p>
                <p><strong>الحجم:</strong> {info['size']}</p>
                <p><strong>الأسطر:</strong> {info['rows']:,}</p>
                <p><strong>الأعمدة:</strong> {info['columns']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # المحتوى الرئيسي
    if not st.session_state.data_loaded and not st.session_state.legal_texts:
        # شاشة الترحيب
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="glass-card float-animation">
                <div style="font-size: 3rem; text-align: center;">📊</div>
                <h3 style="color: #00ff88; text-align: center;">تحليل البيانات</h3>
                <p style="color: rgba(255,255,255,0.7); text-align: center;">Auto-EDA متقدم مع تصورات تفاعلية</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card float-animation" style="animation-delay: 0.2s;">
                <div style="font-size: 3rem; text-align: center;">🔍</div>
                <h3 style="color: #00ff88; text-align: center;">كشف الفساد</h3>
                <p style="color: rgba(255,255,255,0.7); text-align: center;">أنماط غير عادية وشبهات فساد</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card float-animation" style="animation-delay: 0.4s;">
                <div style="font-size: 3rem; text-align: center;">⚖️</div>
                <h3 style="color: #00ff88; text-align: center;">تحليل قانوني</h3>
                <p style="color: rgba(255,255,255,0.7); text-align: center;">معالجة النصوص العربية وفهم الأحكام</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # إنشاء التبويبات
    tabs = st.tabs([
        "📊 لوحة المعلومات",
        "🔍 كشف الشذوذ",
        "🤖 التنبؤ بالفساد",
        "⚖️ التحليل القانوني",
        "📈 التقارير المتقدمة"
    ])
    
    # ========== تبويب لوحة المعلومات ==========
    with tabs[0]:
        if st.session_state.df is not None:
            df = st.session_state.df
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 نظرة عامة على البيانات</div>', unsafe_allow_html=True)
            
            # مقاييس سريعة
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                display_metrics_card(
                    "إجمالي السجلات",
                    f"{len(df):,}",
                    f"{len(df.columns)} عمود"
                )
            
            with col2:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                display_metrics_card(
                    "أعمدة رقمية",
                    str(numeric_cols),
                    f"{len(df.columns) - numeric_cols} أعمدة نصية"
                )
            
            with col3:
                missing = df.isnull().sum().sum()
                display_metrics_card(
                    "قيم مفقودة",
                    f"{missing:,}",
                    f"{(missing/(len(df)*len(df.columns)))*100:.1f}%"
                )
            
            with col4:
                duplicates = df.duplicated().sum()
                display_metrics_card(
                    "مكررات",
                    f"{duplicates:,}",
                    f"{(duplicates/len(df))*100:.1f}% من البيانات"
                )
            
            # عرض البيانات
            st.markdown("### 📋 عينة من البيانات")
            st.dataframe(df.head(10), use_container_width=True)
            
            # تحليل جودة البيانات
            if st.button("🔍 تحليل جودة البيانات", use_container_width=True):
                with st.spinner("جاري تحليل جودة البيانات..."):
                    quality_report = detect_data_quality(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 إحصائيات عامة")
                        st.json({
                            'إجمالي الأسطر': quality_report['total_rows'],
                            'إجمالي الأعمدة': quality_report['total_columns'],
                            'قيم مفقودة': quality_report['missing_values'],
                            'مكررات': quality_report['duplicates'],
                            'حجم الذاكرة': f"{quality_report['memory_usage']:.2f} MB"
                        })
                    
                    with col2:
                        st.markdown("#### 🔢 أنواع البيانات")
                        st.json(quality_report['data_types'])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== تبويب كشف الشذوذ ==========
    with tabs[1]:
        if st.session_state.anomalies is not None:
            anomalies_df = st.session_state.anomalies
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 تحليل الشذوذ المتقدم</div>', unsafe_allow_html=True)
            
            # مقاييس الشذوذ
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                anomaly_count = anomalies_df['is_anomaly'].sum()
                display_metrics_card(
                    "حالات شاذة",
                    str(anomaly_count),
                    f"{(anomaly_count/len(anomalies_df))*100:.2f}%"
                )
            
            with col2:
                avg_anomaly_score = anomalies_df['anomaly_score_ensemble'].mean()
                display_metrics_card(
                    "متوسط درجة الشذوذ",
                    f"{avg_anomaly_score:.3f}",
                    "0-1 (أعلى = شاذ)"
                )
            
            with col3:
                if 'amount' in anomalies_df.columns or 'المبلغ' in anomalies_df.columns:
                    amount_col = 'amount' if 'amount' in anomalies_df.columns else 'المبلغ'
                    anomaly_amount = anomalies_df[anomalies_df['is_anomaly']][amount_col].sum()
                    display_metrics_card(
                        "قيمة المشبوهة",
                        f"{anomaly_amount:,.0f}",
                        "إجمالي المعاملات المشبوهة"
                    )
            
            with col4:
                if st.session_state.fraud_report:
                    corruption_score = st.session_state.fraud_report.get('corruption_score', 0)
                    display_metrics_card(
                        "مؤشر الفساد",
                        f"{corruption_score:.1f}%",
                        "نسبة الخطورة"
                    )
            
            # عرض الحالات الشاذة
            st.markdown("### 🚨 الحالات المشبوهة")
            anomalies_only = anomalies_df[anomalies_df['is_anomaly']]
            st.dataframe(anomalies_only, use_container_width=True)
            
            # تصور الشذوذ
            st.markdown("### 📊 تصور الشذوذ")
            fig = create_anomaly_dashboard(anomalies_df, df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # تحليل الفساد
            if st.session_state.fraud_report:
                fraud_report = st.session_state.fraud_report
                
                st.markdown("### 🕵️ تحليل أنماط الفساد")
                
                if fraud_report['fraud_indicators']:
                    for indicator in fraud_report['fraud_indicators']:
                        display_alert(
                            f"**{indicator['description']}**: {indicator['count']} حالة",
                            type='warning' if indicator['count'] > 10 else 'info'
                        )
                else:
                    st.info("لم يتم العثور على مؤشرات فساد واضحة")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 قم بتشغيل كشف الشذوذ من الشريط الجانبي أولاً")
    
    # ========== تبويب التنبؤ بالفساد ==========
    with tabs[2]:
        if st.session_state.model_pack is not None:
            model_pack = st.session_state.model_pack
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🤖 نموذج التنبؤ بالفساد</div>', unsafe_allow_html=True)
            
            # مقاييس النموذج
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                display_metrics_card(
                    "الدقة",
                    f"{model_pack['metrics']['accuracy']*100:.1f}%",
                    "Accuracy"
                )
            
            with col2:
                display_metrics_card(
                    "Precision",
                    f"{model_pack['metrics']['precision']*100:.1f}%",
                    "دقة التنبؤ"
                )
            
            with col3:
                display_metrics_card(
                    "Recall",
                    f"{model_pack['metrics']['recall']*100:.1f}%",
                    "تغطية الحالات"
                )
            
            with col4:
                display_metrics_card(
                    "F1 Score",
                    f"{model_pack['metrics']['f1']*100:.1f}%",
                    "متوسط الوزن"
                )
            
            # أهمية الميزات
            st.markdown("### 📊 أهمية المتغيرات في التنبؤ")
            fig = px.bar(
                model_pack['feature_importance'].head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='أهم 10 متغيرات مؤثرة',
                color='importance',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # تنبؤات جديدة
            st.markdown("### 🔮 تنبؤات جديدة")
            
            # إنشاء نموذج إدخال للتنبؤ
            input_data = {}
            cols = st.columns(3)
            
            for i, feature in enumerate(model_pack['feature_cols'][:6]):
                with cols[i % 3]:
                    if feature in df.columns:
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        mean_val = float(df[feature].mean())
                        
                        input_data[feature] = st.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=(max_val - min_val) / 100
                        )
            
            if st.button("🔮 تنبؤ", use_container_width=True):
                # تجهيز بيانات الإدخال
                input_df = pd.DataFrame([input_data])
                
                # التنبؤ
                prediction = model_pack['model'].predict(input_df)[0]
                probability = model_pack['model'].predict_proba(input_df)[0]
                
                # عرض النتيجة
                if prediction == 1:
                    display_alert(
                        f"⚠️ **احتمالية فساد عالية**: {probability[1]*100:.1f}%",
                        type='danger'
                    )
                else:
                    display_alert(
                        f"✅ **احتمالية فساد منخفضة**: {probability[0]*100:.1f}%",
                        type='success'
                    )
            
            # تفسير SHAP
            if SHAP_AVAILABLE and st.button("📊 تفسير النموذج (SHAP)", use_container_width=True):
                with st.spinner("جاري حساب تفسيرات SHAP..."):
                    shap_values, error = explain_predictions(
                        model_pack['model'],
                        model_pack['X_test'],
                        model_pack['feature_cols']
                    )
                    
                    if shap_values is not None:
                        st.success("✅ تم حساب تفسيرات SHAP")
                        
                        # عرض summary plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(
                            shap_values,
                            model_pack['X_test'],
                            feature_names=model_pack['feature_cols'],
                            show=False
                        )
                        st.pyplot(fig)
                        plt.close()
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 قم بتدريب نموذج التنبؤ من الشريط الجانبي أولاً")
    
    # ========== تبويب التحليل القانوني ==========
    with tabs[3]:
        if st.session_state.legal_texts:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">⚖️ تحليل النصوص القانونية</div>', unsafe_allow_html=True)
            
            # تحليل النصوص
            if st.button("🔍 تحليل النصوص", use_container_width=True):
                with st.spinner("جاري تحليل النصوص..."):
                    analysis_results = analyze_legal_text(st.session_state.legal_texts)
                    
                    if 'wordcloud' in analysis_results:
                        st.markdown("### ☁️ Word Cloud")
                        st.pyplot(analysis_results['wordcloud'])
                    
                    if 'top_words' in analysis_results:
                        st.markdown("### 📊 الكلمات الأكثر تكراراً")
                        words_df = pd.DataFrame(
                            analysis_results['top_words'][:20],
                            columns=['الكلمة', 'التكرار']
                        )
                        
                        fig = px.bar(
                            words_df,
                            x='التكرار',
                            y='الكلمة',
                            orientation='h',
                            color='التكرار',
                            color_continuous_scale='Greens'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if 'sentiment' in analysis_results:
                        st.markdown("### 😊 تحليل المشاعر")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Polarity", f"{analysis_results['sentiment']['polarity']:.2f}")
                        with col2:
                            st.metric("Subjectivity", f"{analysis_results['sentiment']['subjectivity']:.2f}")
            
            # عرض النصوص
            st.markdown("### 📄 النصوص القانونية")
            for i, text in enumerate(st.session_state.legal_texts[:5]):
                with st.expander(f"نص {i+1}"):
                    st.write(text)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 قم برفع ملف PDF أو TXT من الشريط الجانبي")
    
    # ========== تبويب التقارير المتقدمة ==========
    with tabs[4]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 التقارير المتقدمة</div>', unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            # اختيار نوع التقرير
            report_type = st.selectbox(
                "نوع التقرير",
                ["تحليل إحصائي", "تحليل زمني", "تحليل علاقات", "تقرير شامل"]
            )
            
            if report_type == "تحليل إحصائي":
                st.markdown("### 📊 إحصائيات وصفية")
                st.dataframe(
                    st.session_state.df.describe(include='all'),
                    use_container_width=True
                )
                
                # خريطة حرارية
                st.markdown("### 🔥 خريطة الارتباطات")
                fig = create_correlation_heatmap(st.session_state.df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            elif report_type == "تحليل زمني":
                # البحث عن أعمدة تاريخ
                date_cols = [col for col in st.session_state.df.columns 
                           if 'date' in col.lower() or 'time' in col.lower() or 'تاريخ' in col]
                
                if date_cols:
                    date_col = st.selectbox("اختر عمود التاريخ", date_cols)
                    
                    try:
                        df_time = st.session_state.df.copy()
                        df_time[date_col] = pd.to_datetime(df_time[date_col])
                        df_time['year'] = df_time[date_col].dt.year
                        df_time['month'] = df_time[date_col].dt.month
                        
                        # تحليل زمني
                        time_analysis = df_time.groupby('year').size().reset_index(name='count')
                        
                        fig = px.line(
                            time_analysis,
                            x='year',
                            y='count',
                            title='تطور المعاملات عبر الزمن',
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except:
                        st.error("لا يمكن تحويل العمود إلى تاريخ")
                else:
                    st.warning("لا توجد أعمدة تاريخ في البيانات")
            
            elif report_type == "تحليل علاقات":
                # اختيار متغيرين للتحليل
                numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) >= 2:
                    col1 = st.selectbox("المتغير الأول", numeric_cols, index=0)
                    col2 = st.selectbox("المتغير الثاني", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    # رسم scatter plot
                    fig = px.scatter(
                        st.session_state.df,
                        x=col1,
                        y=col2,
                        title=f'العلاقة بين {col1} و {col2}',
                        trendline='ols'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # حساب معامل الارتباط
                    correlation = st.session_state.df[col1].corr(st.session_state.df[col2])
                    display_alert(
                        f"معامل الارتباط: {correlation:.3f}",
                        type='info' if abs(correlation) < 0.5 else 'warning' if abs(correlation) < 0.8 else 'danger'
                    )
                else:
                    st.warning("تحتاج إلى عمودين رقميين على الأقل")
            
            elif report_type == "تقرير شامل":
                if st.button("📊 إنشاء تقرير شامل", use_container_width=True):
                    with st.spinner("جاري إنشاء التقرير الشامل..."):
                        # تقرير جودة البيانات
                        quality_report = detect_data_quality(st.session_state.df)
                        
                        # تقرير الشذوذ
                        anomalies_df, _ = detect_anomalies_advanced(st.session_state.df)
                        
                        # تقرير الفساد
                        fraud_report = detect_fraud_patterns(st.session_state.df)
                        
                        # عرض التقرير
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 ملخص البيانات")
                            st.json({
                                'إجمالي السجلات': quality_report['total_rows'],
                                'إجمالي الأعمدة': quality_report['total_columns'],
                                'قيم مفقودة': quality_report['missing_values'],
                                'مكررات': quality_report['duplicates']
                            })
                            
                            st.markdown("#### 🚨 مؤشرات الفساد")
                            st.json({
                                'حالات مشبوهة': fraud_report['suspicious_cases'],
                                'درجة الفساد': f"{fraud_report['corruption_score']:.1f}%",
                                'مؤشرات مكتشفة': len(fraud_report['fraud_indicators'])
                            })
                        
                        with col2:
                            if anomalies_df is not None:
                                st.markdown("#### 🔍 تحليل الشذوذ")
                                st.json({
                                    'حالات شاذة': int(anomalies_df['is_anomaly'].sum()),
                                    'نسبة الشذوذ': f"{(anomalies_df['is_anomaly'].sum()/len(anomalies_df))*100:.1f}%",
                                    'متوسط درجة الشذوذ': f"{anomalies_df['anomaly_score_ensemble'].mean():.3f}"
                                })
                            
                            st.markdown("#### ⚠️ توصيات")
                            if fraud_report['corruption_score'] > 30:
                                st.error("مؤشر فساد مرتفع - ينصح بمراجعة عاجلة")
                            elif fraud_report['corruption_score'] > 15:
                                st.warning("مؤشر فساد متوسط - يحتاج متابعة")
                            else:
                                st.success("مؤشر فساد منخفض - أداء جيد")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # الفوتر
    st.markdown("""
    <div class="footer">
        <h3>🛡️ AI Administrative Audit System</h3>
        <p>الإصدار الاحترافي 1.0 | جميع الحقوق محفوظة © 2026</p>
        <p style="margin-top: 1rem; font-size: 0.9rem;">
            نظام متكامل لكشف الفساد المالي والإداري باستخدام أحدث تقنيات الذكاء الاصطناعي
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
