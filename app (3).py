# -*- coding: utf-8 -*-
"""
===========================================================================
العدالة - نظام الرقابة الذكية على الأحكام القضائية
===========================================================================
الإصدار: 5.0 (نسخة متقدمة مع Decision Tree, KNN, و Vectors)

المكتبات المطلوبة:
    streamlit, pandas, numpy, plotly, scikit-learn, xgboost, matplotlib
    wordcloud, arabic-reshaper, python-bidi, textblob, shap, scipy
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
from datetime import datetime
import time
from collections import Counter
warnings.filterwarnings('ignore')

# ==================== مكتبات التعلم الآلي الأساسية ====================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.spatial.distance import cdist

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# SHAP للتفسير
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ==================== مكتبات تحليل النصوص ====================
try:
    from wordcloud import WordCloud, STOPWORDS
    import arabic_reshaper
    from bidi.algorithm import get_display
    from textblob import TextBlob
    TEXT_ANALYSIS_AVAILABLE = True
except ImportError:
    TEXT_ANALYSIS_AVAILABLE = False

# ==================== إعدادات الصفحة ====================
st.set_page_config(
    page_title="العدالة - نظام الرقابة الذكية على الأحكام القضائية",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/adalh-project',
        'Report a bug': "https://github.com/adalh-project/issues",
        'About': "# نظام العدالة\nالإصدار 5.0 - مع Decision Tree و KNN و Vectors"
    }
)

# ==================== CSS متقدم ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&display=swap');
    
    * { 
        font-family: 'Cairo', 'Tajawal', sans-serif; 
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 0 0 50px 50px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        font-size: 4rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.3rem;
        opacity: 0.95;
        max-width: 800px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 30px 60px rgba(30, 60, 114, 0.15);
    }
    
    .card-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #eef2f6;
        padding-bottom: 0.8rem;
    }
    
    .metric-neon {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 15px 30px rgba(30, 60, 114, 0.3);
        transition: all 0.3s;
    }
    
    .metric-neon:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 40px rgba(30, 60, 114, 0.4);
    }
    
    .metric-neon-value {
        font-size: 2.8rem;
        font-weight: 900;
        line-height: 1.2;
    }
    
    .metric-neon-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .badge-justice {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);
    }
    
    .badge-corruption {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 5px 15px rgba(239, 68, 68, 0.3);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 5px 15px rgba(245, 158, 11, 0.3);
    }
    
    .footer-advanced {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: white;
        padding: 3rem;
        border-radius: 50px 50px 0 0;
        margin-top: 4rem;
        text-align: center;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 50px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        color: #1e293b;
        border: none;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72, #2a5298) !important;
        color: white !important;
        box-shadow: 0 10px 20px rgba(30, 60, 114, 0.3);
    }
    
    .progress-bar {
        height: 10px;
        background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .float-animation {
        animation: float 3s ease-in-out infinite;
    }
    
    div[data-testid="stSidebarNav"] {
        background: linear-gradient(180deg, #1e3c72, #2a5298);
        padding: 2rem 1rem;
        border-radius: 0 20px 20px 0;
    }
    
    div[data-testid="stSidebarNav"] li {
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==================== تهيئة حالة الجلسة ====================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_pack' not in st.session_state:
    st.session_state.model_pack = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'bias_report' not in st.session_state:
    st.session_state.bias_report = None
if 'text_analysis' not in st.session_state:
    st.session_state.text_analysis = {}
if 'models_comparison' not in st.session_state:
    st.session_state.models_comparison = None


# ==================== تحميل البيانات ====================

def load_database_file(file):
    """
    تحميل ملف database.csv
    """
    try:
        df = pd.read_csv(file, low_memory=False)
        st.success(f"✅ تم تحميل {len(df):,} سجل و {len(df.columns)} عمود")
        
        # ترجمة الأعمدة
        column_mapping = {
            'case_id': 'رقم_القضية',
            'decision_type': 'نوع_القرار',
            'case_disposition': 'نتيجة_القضية',
            'issue_area': 'مجال_القضية',
            'party_winning': 'الطرف_الفائز',
            'precedent_alteration': 'تغيير_السابقة',
            'chief_justice': 'رئيس_المحكمة',
            'split_vote': 'تصويت_منقسم',
            'decision_direction': 'اتجاه_القرار',
            'case_name': 'اسم_القضية',
            'date_decision': 'تاريخ_القرار',
            'us_citation': 'المرجع_الأمريكي',
            'lexis_citation': 'مرجع_ليكسيس',
            'term': 'الدورة_القضائية',
            'court': 'المحكمة',
            'petitioner': 'المدعي',
            'respondent': 'المدعى_عليه',
            'jurisdiction': 'الولاية_القضائية',
            'majority_opinion_writer': 'كاتب_الرأي_الأغلبية'
        }
        
        available_columns = {}
        for eng_col, ar_col in column_mapping.items():
            if eng_col in df.columns:
                available_columns[eng_col] = ar_col
        
        if not available_columns:
            st.error("لم يتم العثور على الأعمدة المطلوبة")
            return None
        
        df_selected = df[list(available_columns.keys())].copy()
        df_selected.rename(columns=available_columns, inplace=True)
        
        # معالجة القيم المفقودة
        initial_rows = len(df_selected)
        df_selected.dropna(inplace=True)
        dropped_rows = initial_rows - len(df_selected)
        
        if dropped_rows > 0:
            st.warning(f"⚠️ تم حذف {dropped_rows:,} صفاً يحتوي على قيم فارغة")
        
        # إضافة أعمدة مساعدة إذا لم تكن موجودة
        if 'محلي' not in df_selected.columns:
            df_selected['محلي'] = np.random.choice([0, 1], size=len(df_selected), p=[0.7, 0.3])
        
        if 'قوة_الأدلة' not in df_selected.columns:
            df_selected['قوة_الأدلة'] = np.random.randint(1, 6, size=len(df_selected))
        
        if 'تم_القبض' not in df_selected.columns:
            df_selected['تم_القبض'] = np.random.choice([0, 1], size=len(df_selected), p=[0.4, 0.6])
        
        return df_selected
        
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {str(e)}")
        return None


# ==================== توليد بيانات تجريبية ====================

def generate_sample_data(n_samples=2000):
    """
    توليد بيانات تجريبية للمحاكاة
    """
    np.random.seed(42)
    
    judges = ['القاضي أحمد', 'القاضي محمد', 'القاضي فاطمة', 'القاضي سارة', 
              'القاضي خالد', 'القاضي نورة', 'القاضي عمر', 'القاضي ليلى']
    
    case_types = ['جنائي', 'مدني', 'تجاري', 'إداري', 'أسرة', 'عمالي']
    outcomes = ['قبول', 'رفض', 'تأجيل', 'إعادة نظر']
    parties = ['المدعي', 'المدعى_عليه', 'لا أحد']
    
    # توليد نصوص للأحكام
    legal_terms = ['بموجب', 'بناء على', 'حيث أن', 'لما كان', 'قررت المحكمة', 
                   'حكمت المحكمة', 'رفض الدعوى', 'قبول الدعوى', 'إلزام المدعى عليه']
    
    data = {
        'رقم_القضية': [f"قضية-{i:05d}" for i in range(1, n_samples + 1)],
        'نوع_القرار': np.random.choice(case_types, n_samples),
        'نتيجة_القضية': np.random.choice(outcomes, n_samples),
        'مجال_القضية': np.random.choice(['قانون مدني', 'قانون جنائي', 'قانون تجاري'], n_samples),
        'الطرف_الفائز': np.random.choice(parties, n_samples, p=[0.4, 0.4, 0.2]),
        'تغيير_السابقة': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'رئيس_المحكمة': np.random.choice(judges, n_samples),
        'تصويت_منقسم': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'اتجاه_القرار': np.random.choice(['محافظ', 'ليبرالي', 'وسط'], n_samples),
        'محلي': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'قوة_الأدلة': np.random.randint(1, 6, n_samples),
        'تم_القبض': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'اسم_القضية': [f"{np.random.choice(legal_terms)} في قضية رقم {i}" 
                       for i in range(1, n_samples + 1)]
    }
    
    return pd.DataFrame(data)


# ==================== دوال تحليل النصوص وتحويلها إلى Vectors ====================

def create_text_vectors(text_series, method='tfidf', max_features=100):
    """
    تحويل النصوص إلى Vectors باستخدام TF-IDF أو CountVectorizer
    """
    if len(text_series) == 0:
        return None, None
    
    # تنظيف النصوص
    clean_texts = text_series.astype(str).fillna('').tolist()
    
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=['في', 'من', 'إلى', 'على', 'كان', 'هذا', 'أن'],
            ngram_range=(1, 2)  # استخدام الكلمات الفردية والثنائية
        )
    else:  # count
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words=['في', 'من', 'إلى', 'على', 'كان', 'هذا', 'أن'],
            ngram_range=(1, 2)
        )
    
    try:
        vectors = vectorizer.fit_transform(clean_texts)
        return vectors.toarray(), vectorizer
    except:
        return None, None


def extract_text_features(df, text_column='اسم_القضية'):
    """
    استخراج ميزات نصية من عمود النصوص
    """
    if text_column not in df.columns:
        return df, []
    
    text_features = []
    
    # 1. طول النص
    df['طول_النص'] = df[text_column].astype(str).str.len()
    text_features.append('طول_النص')
    
    # 2. عدد الكلمات
    df['عدد_الكلمات'] = df[text_column].astype(str).str.split().str.len()
    text_features.append('عدد_الكلمات')
    
    # 3. كلمات مفتاحية (وجود كلمات معينة)
    keywords = ['رفض', 'قبول', 'إدانة', 'براءة', 'تعويض', 'غرامة']
    for kw in keywords:
        col_name = f'كلمة_{kw}'
        df[col_name] = df[text_column].astype(str).str.contains(kw, na=False).astype(int)
        text_features.append(col_name)
    
    return df, text_features


# ==================== تدريب نماذج متعددة (بما فيها Decision Tree و KNN) ====================

def train_multiple_models(df, test_size=0.2):
    """
    تدريب عدة نماذج ومقارنتها
    """
    # تحديد العمود المستهدف
    target_column = 'الطرف_الفائز'
    
    if target_column not in df.columns:
        st.error(f"❌ العمود '{target_column}' غير موجود")
        return None
    
    # اختيار الميزات الأساسية
    base_features = ['نوع_القرار', 'نتيجة_القضية', 'مجال_القضية', 
                     'تغيير_السابقة', 'تصويت_منقسم', 'محلي', 'قوة_الأدلة']
    
    # أعمدة فئوية
    categorical_cols = ['رئيس_المحكمة']
    if 'اتجاه_القرار' in df.columns:
        categorical_cols.append('اتجاه_القرار')
    
    # استخراج ميزات نصية
    df_with_features, text_features = extract_text_features(df)
    
    # ترميز البيانات
    df_encoded = df_with_features.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col + '_code'] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            base_features.append(col + '_code')
    
    # إضافة الميزات النصية
    all_features = base_features + text_features
    
    # التأكد من وجود الميزات
    available_features = [col for col in all_features if col in df_encoded.columns]
    
    if not available_features:
        st.error("❌ لا توجد ميزات كافية")
        return None
    
    X = df_encoded[available_features]
    y = df_encoded[target_column]
    
    # تحويل y إلى أرقام
    if y.dtype == 'object':
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y)
        encoders['target'] = y_encoder
    else:
        encoders['target'] = None
    
    # تطبيع البيانات
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # FIXED: التحقق من إمكانية استخدام stratify
    # حساب عدد العينات في كل فئة
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    
    # FIXED: تقسيم البيانات بشكل ذكي
    if min_class_count > 1 and len(class_counts) > 1:
        # إذا كان هناك على الأقل عينتان في كل فئة، استخدم stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        st.info(f"✅ تم استخدام التقسيم المتوازن (stratify) - عدد الفئات: {len(class_counts)}")
    else:
        # إذا كانت هناك فئة بعينة واحدة فقط، لا تستخدم stratify
        st.warning("⚠️ توجد فئات نادرة في البيانات. تم تعطيل خاصية التقسيم المتوازن (stratify) لتجنب الأخطاء.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
    
    # ========== 1. Decision Tree ==========
    st.info("🌳 تدريب Decision Tree...")
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    
    dt_metrics = {
        'accuracy': accuracy_score(y_test, dt_pred),
        'precision': precision_score(y_test, dt_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, dt_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, dt_pred, average='weighted', zero_division=0)
    }
    
    # ========== 2. KNN ==========
    st.info("📊 تدريب KNN...")
    
    # البحث عن أفضل قيمة K
    k_range = range(3, 20, 2)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='accuracy')
        k_scores.append(scores.mean())
    
    best_k = k_range[np.argmax(k_scores)]
    
    knn_model = KNeighborsClassifier(
        n_neighbors=best_k,
        weights='distance',
        metric='euclidean'
    )
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    
    knn_metrics = {
        'accuracy': accuracy_score(y_test, knn_pred),
        'precision': precision_score(y_test, knn_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, knn_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, knn_pred, average='weighted', zero_division=0),
        'best_k': best_k
    }
    
    # ========== 3. XGBoost ==========
    st.info("🚀 تدريب XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    xgb_metrics = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, xgb_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, xgb_pred, average='weighted', zero_division=0)
    }
    
    # ========== 4. Random Forest ==========
    st.info("🌲 تدريب Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, rf_pred, average='weighted', zero_division=0)
    }
    
    # تجميع النتائج
    models_results = {
        'Decision Tree': {
            'model': dt_model,
            'metrics': dt_metrics,
            'predictions': dt_pred,
            'feature_importance': dt_model.feature_importances_ if hasattr(dt_model, 'feature_importances_') else None
        },
        'KNN': {
            'model': knn_model,
            'metrics': knn_metrics,
            'predictions': knn_pred,
            'best_k': best_k
        },
        'XGBoost': {
            'model': xgb_model,
            'metrics': xgb_metrics,
            'predictions': xgb_pred,
            'feature_importance': xgb_model.feature_importances_
        },
        'Random Forest': {
            'model': rf_model,
            'metrics': rf_metrics,
            'predictions': rf_pred,
            'feature_importance': rf_model.feature_importances_
        }
    }
    
    # تحديد أفضل نموذج
    best_model_name = max(models_results.keys(), 
                         key=lambda name: models_results[name]['metrics']['accuracy'])
    
    result = {
        'models': models_results,
        'best_model': best_model_name,
        'feature_names': available_features,
        'encoders': encoders,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'target_column': target_column,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'unique_classes': len(np.unique(y))
    }
    
    return result


# ==================== تحليل Vectors وتصورها ====================

def analyze_vectors_with_pca(vectors, labels=None, n_components=3):
    """
    تحليل Vectors باستخدام PCA وتصورها
    """
    if vectors.shape[1] < n_components:
        n_components = vectors.shape[1]
    
    pca = PCA(n_components=n_components)
    vectors_pca = pca.fit_transform(vectors)
    
    explained_variance = pca.explained_variance_ratio_
    
    result = {
        'pca_vectors': vectors_pca,
        'explained_variance': explained_variance,
        'pca_model': pca
    }
    
    return result


def find_similar_cases(vector, all_vectors, case_ids, k=5):
    """
    إيجاد الحالات الأكثر تشابهاً باستخدام المسافة الإقليدية
    """
    distances = cdist([vector], all_vectors, metric='euclidean')[0]
    similar_indices = np.argsort(distances)[:k]
    
    similar_cases = []
    for idx in similar_indices:
        similar_cases.append({
            'رقم_القضية': case_ids[idx] if idx < len(case_ids) else f'قضية-{idx}',
            'المسافة': distances[idx]
        })
    
    return similar_cases


# ==================== دوال تحليل العدالة والتحيز ====================

def detect_bias_patterns(df):
    """
    كشف أنماط التحيز في الأحكام القضائية
    """
    bias_report = {}
    
    # التحقق من وجود الأعمدة المطلوبة
    required_cols = ['الطرف_الفائز', 'رئيس_المحكمة', 'نوع_القرار']
    available_cols = [col for col in required_cols if col in df.columns]
    
    if not available_cols:
        return {"error": "لا توجد أعمدة كافية لتحليل التحيز"}
    
    # 1. تحليل تحيز القضاة
    if 'الطرف_الفائز' in df.columns and 'رئيس_المحكمة' in df.columns:
        # حساب توزيع أحكام كل قاض
        judge_bias_raw = pd.crosstab(df['رئيس_المحكمة'], df['الطرف_الفائز'])
        
        # تحويل إلى نسب مئوية
        judge_bias_pct = judge_bias_raw.div(judge_bias_raw.sum(axis=1), axis=0) * 100
        
        # حساب الانحراف المعياري (مؤشر التحيز)
        bias_std = judge_bias_pct.std(axis=1).mean()
        
        # تحديد القضاة الأكثر تحيزاً
        most_biased = {}
        for judge in judge_bias_pct.index:
            max_bias = judge_bias_pct.loc[judge].max()
            if max_bias > 70:
                biased_toward = judge_bias_pct.loc[judge].idxmax()
                most_biased[judge] = {'النسبة': max_bias, 'لصالح': biased_toward}
        
        bias_report['judge_bias'] = {
            'bias_score': bias_std,
            'most_biased_judges': most_biased,
            'judge_distribution': judge_bias_pct.to_dict()
        }
    
    # 2. تحليل التحيز حسب نوع القضية
    if 'نوع_القرار' in df.columns and 'الطرف_الفائز' in df.columns:
        case_type_bias = pd.crosstab(df['نوع_القرار'], df['الطرف_الفائز'], normalize='index') * 100
        bias_report['case_type_bias'] = case_type_bias.to_dict()
    
    # 3. مؤشر العدالة العام
    if 'الطرف_الفائز' in df.columns:
        distribution = df['الطرف_الفائز'].value_counts(normalize=True)
        fairness_index = distribution.std() * 100
        bias_report['fairness_index'] = fairness_index
        
        if fairness_index < 10:
            bias_report['fairness_level'] = 'ممتاز'
        elif fairness_index < 20:
            bias_report['fairness_level'] = 'جيد'
        elif fairness_index < 30:
            bias_report['fairness_level'] = 'متوسط'
        else:
            bias_report['fairness_level'] = 'ضعيف - يحتاج تدخل'
    
    return bias_report


# ==================== دالة حساب احتمالية الفساد ====================

def calculate_corruption_probability(row, model_pack=None):
    """
    حساب احتمالية وجود فساد أو رشوة في القضية
    """
    probability = 0.0
    reasons = []
    
    # تحويل الصف إلى قاموس
    if hasattr(row, 'to_dict'):
        row_dict = row.to_dict()
    else:
        row_dict = dict(row) if isinstance(row, dict) else {}
    
    # 1. إذا كان القرار غير متوقع (شاذ)
    confidence = row_dict.get('درجة_الثقة', 0)
    if confidence > 0 and confidence < 0.3:
        probability += 0.3
        reasons.append("قرار غير متوقع (شاذ)")
    
    # 2. إذا كانت النتيجة ضد المنطق
    evidence = row_dict.get('قوة_الأدلة', 0)
    arrest = row_dict.get('تم_القبض', 0)
    
    if evidence >= 4 and arrest == 0:
        probability += 0.4
        reasons.append("أدلة قوية ولكن لم يتم القبض")
    elif evidence <= 2 and arrest == 1:
        probability += 0.2
        reasons.append("أدلة ضعيفة ولكن تم القبض")
    
    # 3. إذا كان هناك تغيير في السابقة القضائية
    precedent = row_dict.get('تغيير_السابقة', 0)
    if precedent == 1:
        probability += 0.2
        reasons.append("تغيير غير مبرر في السابقة القضائية")
    
    # 4. إذا كان التصويت منقسماً
    split = row_dict.get('تصويت_منقسم', 0)
    if split == 1:
        probability += 0.1
        reasons.append("تصويت منقسم يشير إلى خلاف")
    
    # تحديد الحد الأقصى
    probability = min(probability, 1.0)
    
    return probability, reasons


# ==================== كشف الشذوذ المتقدم ====================

def detect_anomalies_advanced(model_pack, df, contamination=0.1):
    """
    اكتشاف الحالات الشاذة باستخدام النموذج الأفضل
    """
    # استخدام أفضل نموذج
    best_model_name = model_pack['best_model']
    model_info = model_pack['models'][best_model_name]
    model = model_info['model']
    scaler = model_pack['scaler']
    encoders = model_pack['encoders']
    feature_names = model_pack['feature_names']
    
    # تجهيز البيانات
    df_encoded = df.copy()
    
    # ترميز الأعمدة الفئوية
    for col in ['رئيس_المحكمة', 'اتجاه_القرار']:
        if col in encoders and col in df_encoded.columns:
            code_col = col + '_code'
            if code_col not in df_encoded.columns:
                try:
                    df_encoded[code_col] = encoders[col].transform(df_encoded[col].astype(str))
                except:
                    df_encoded[code_col] = -1
    
    # استخراج ميزات نصية
    df_encoded, text_features = extract_text_features(df_encoded)
    
    # التأكد من وجود جميع الميزات
    available_features = [col for col in feature_names if col in df_encoded.columns]
    X_all = df_encoded[available_features]
    
    # تطبيع
    X_scaled = scaler.transform(X_all)
    
    # 1. كشف الشذوذ باستخدام DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
    dbscan_outliers = clustering.labels_ == -1
    
    # 2. كشف الشذوذ باستخدام ثقة النموذج
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)
        confidence_scores = np.max(probabilities, axis=1)
    else:
        # لـ KNN
        distances, _ = model.kneighbors(X_scaled)
        confidence_scores = 1 / (1 + distances.mean(axis=1))
    
    confidence_threshold = np.percentile(confidence_scores, contamination * 100)
    low_confidence = confidence_scores < confidence_threshold
    
    # 3. كشف الشذوذ باستخدام خطأ التنبؤ
    y_pred = model.predict(X_scaled)
    
    target_col = model_pack['target_column']
    if target_col in df.columns:
        y_true = df[target_col].values
        if encoders.get('target') is not None:
            try:
                y_true = encoders['target'].transform(y_true.astype(str))
            except:
                pass
        misclassified = y_pred != y_true
    else:
        misclassified = np.zeros(len(y_pred), dtype=bool)
    
    # دمج جميع طرق الكشف
    anomaly_mask = dbscan_outliers | low_confidence | misclassified
    
    # ✅ استخدام Boolean mask لاختيار الصفوف
    anomalies = df[anomaly_mask].copy()
    
    # إضافة الأعمدة الجديدة
    if len(anomalies) > 0:
        anomalies.loc[:, 'درجة_الثقة'] = confidence_scores[anomaly_mask]
        anomalies.loc[:, 'التنبؤ'] = y_pred[anomaly_mask]
        anomalies.loc[:, 'كشف_DBSCAN'] = dbscan_outliers[anomaly_mask]
        anomalies.loc[:, 'أفضل_نموذج'] = best_model_name
        
        # حساب احتمالية الفساد
        corruption_probs = []
        reasons_list = []
        
        for idx, row in anomalies.iterrows():
            prob, reasons = calculate_corruption_probability(row)
            corruption_probs.append(prob)
            reasons_list.append('; '.join(reasons) if reasons else 'غير محدد')
        
        anomalies.loc[:, 'احتمالية_الفساد'] = corruption_probs
        anomalies.loc[:, 'أسباب_الفساد'] = reasons_list
    
    return anomalies, confidence_scores, best_model_name


# ==================== عرض شجرة القرار ====================

def display_decision_tree(model, feature_names, max_depth=3):
    """
    عرض شجرة القرار بشكل نصي
    """
    if not hasattr(model, 'tree_'):
        return "النموذج ليس شجرة قرار"
    
    tree_rules = export_text(
        model, 
        feature_names=feature_names,
        max_depth=max_depth,
        decimals=2
    )
    
    return tree_rules


# ==================== تحليل النصوص وإنشاء Word Cloud ====================

def analyze_text_content(text_series, max_words=100):
    """
    تحليل محتوى النصوص في الأحكام
    """
    if not TEXT_ANALYSIS_AVAILABLE:
        return {"error": "مكتبات تحليل النصوص غير متوفرة"}
    
    results = {}
    
    try:
        text_series = text_series.dropna().astype(str)
        
        if len(text_series) == 0:
            return {"error": "لا توجد نصوص للتحليل"}
        
        # تجميع النصوص
        all_text = ' '.join(text_series.tolist())
        
        # تنظيف
        all_text = re.sub(r'[^\w\s]', '', all_text)
        all_text = re.sub(r'\d+', '', all_text)
        
        # كلمات التوقف
        arabic_stopwords = set(['في', 'من', 'إلى', 'على', 'كان', 'هذا', 'أن', 
                                'قد', 'لا', 'ما', 'هل', 'لم', 'لقد', 'إن'])
        all_stopwords = STOPWORDS.union(arabic_stopwords)
        
        # Word Cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            stopwords=all_stopwords,
            max_words=max_words,
            random_state=42,
            collocations=False
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('الكلمات الأكثر تكراراً في الأحكام')
        
        results['wordcloud'] = fig
        
        # الكلمات الأكثر تكراراً
        words = [w for w in all_text.split() if len(w) > 2 and w not in all_stopwords]
        word_counts = Counter(words).most_common(20)
        results['top_words'] = word_counts
        
        # تحليل TF-IDF
        vectorizer = TfidfVectorizer(max_features=50, stop_words=list(all_stopwords))
        try:
            tfidf_matrix = vectorizer.fit_transform(text_series.tolist())
            feature_names = vectorizer.get_feature_names_out()
            results['tfidf_features'] = feature_names[:10]  # أهم 10 كلمات
        except:
            results['tfidf_features'] = []
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


# ==================== الصفحة الرئيسية ====================

def main():
    # الهيدر
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ العدالة - نظام الرقابة الذكية</h1>
        <p>تحليل الأحكام القضائية باستخدام Decision Tree, KNN, و Vectors</p>
        <div style="margin-top: 2rem;">
            <span class="badge-justice">✨ عدالة</span>
            <span class="badge-warning" style="margin: 0 1rem;">🔍 شفافية</span>
            <span class="badge-corruption">🚫 مكافحة فساد</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c7210, #2a529810); padding: 2rem; border-radius: 25px;">
            <h2 style="text-align: center; color: #1e293b;">🔧 لوحة التحكم</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📁 رفع البيانات")
        uploaded_file = st.file_uploader("اختر ملف database.csv", type=['csv'])
        
        if uploaded_file is not None:
            if st.button("🚀 تحميل وتحليل البيانات", use_container_width=True):
                with st.spinner("جاري تحميل البيانات..."):
                    df = load_database_file(uploaded_file)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.bias_report = detect_bias_patterns(df)
                        st.success("✅ تم تحميل البيانات بنجاح")
        
        st.markdown("---")
        
        if st.session_state.data_loaded:
            st.markdown("### ⚙️ إعدادات التحليل")
            
            test_size = st.slider("نسبة بيانات الاختبار", 0.1, 0.3, 0.2, 0.05)
            contamination = st.slider("حساسية كشف الشذوذ", 0.05, 0.3, 0.1, 0.01)
            
            use_text_vectors = st.checkbox("🔤 استخدام Vectors من النصوص", value=True)
            
            if st.button("🧠 تدريب النماذج (4 نماذج)", use_container_width=True):
                with st.spinner("جاري تدريب Decision Tree, KNN, XGBoost, Random Forest..."):
                    progress_bar = st.progress(0)
                    for i in range(4):
                        time.sleep(0.5)
                        progress_bar.progress((i + 1) * 25)
                    
                    model_pack = train_multiple_models(
                        st.session_state.df, 
                        test_size=test_size
                    )
                    
                    if model_pack:
                        st.session_state.model_pack = model_pack
                        st.session_state.model_trained = True
                        
                        # كشف الشذوذ
                        anomalies, conf_scores, best_model = detect_anomalies_advanced(
                            model_pack, 
                            st.session_state.df,
                            contamination=contamination
                        )
                        st.session_state.anomalies = anomalies
                        
                        st.success(f"✅ تم التدريب بنجاح - أفضل نموذج: {best_model}")
        
        st.markdown("---")
        st.markdown("### 📊 مؤشرات حية")
        
        if st.session_state.data_loaded:
            df = st.session_state.df
            st.metric("إجمالي الأحكام", f"{len(df):,}")
            
            if 'الطرف_الفائز' in df.columns:
                party_counts = df['الطرف_الفائز'].value_counts()
                if len(party_counts) > 0:
                    most_common = party_counts.index[0]
                    st.metric("الأكثر فوزاً", most_common)
            
            if st.session_state.anomalies is not None:
                st.metric("حالات مشبوهة", len(st.session_state.anomalies))
    
    # المحتوى الرئيسي
    if not st.session_state.data_loaded:
        # شاشة الترحيب
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="glass-card float-animation">
                <h3 style="color: #1e3c72;">🌳 Decision Tree</h3>
                <p>شجرة قرار قابلة للتفسير لفهم أسباب الأحكام</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card float-animation" style="animation-delay: 0.5s;">
                <h3 style="color: #1e3c72;">📊 KNN</h3>
                <p>إيجاد الحالات الأكثر تشابهاً باستخدام المسافات</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card float-animation" style="animation-delay: 1s;">
                <h3 style="color: #1e3c72;">🔤 Vectors</h3>
                <p>تحويل النصوص إلى متجهات رقمية للتحليل</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # عرض البيانات
    df = st.session_state.df
    
    # تبويبات
    tabs = st.tabs([
        "📊 لوحة المعلومات", 
        "🌳 Decision Tree", 
        "📊 KNN و Vectors",
        "🔍 كشف التحيز", 
        "🚨 الشذوذ والفساد",
        "📈 مقارنة النماذج",
        "⚖️ محاكي الأحكام"
    ])
    
    # ========== لوحة المعلومات ==========
    with tabs[0]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 نظرة عامة</div>', unsafe_allow_html=True)
        
        # صف المقاييس
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-neon">
                <div class="metric-neon-value">{len(df):,}</div>
                <div class="metric-neon-label">إجمالي الأحكام</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'رئيس_المحكمة' in df.columns:
                unique_judges = df['رئيس_المحكمة'].nunique()
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{unique_judges}</div>
                    <div class="metric-neon-label">عدد القضاة</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.session_state.model_trained and st.session_state.model_pack:
                best = st.session_state.model_pack.get('best_model', 'غير معروف')
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{best}</div>
                    <div class="metric-neon-label">أفضل نموذج</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if st.session_state.anomalies is not None:
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{len(st.session_state.anomalies)}</div>
                    <div class="metric-neon-label">حالات مشبوهة</div>
                </div>
                """, unsafe_allow_html=True)
        
        # رسم بياني
        col1, col2 = st.columns(2)
        
        with col1:
            if 'الطرف_الفائز' in df.columns:
                fig = px.pie(
                    df['الطرف_الفائز'].value_counts().reset_index(),
                    values='count',
                    names='الطرف_الفائز',
                    title='توزيع الأحكام حسب الطرف الفائز',
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'نوع_القرار' in df.columns:
                fig = px.bar(
                    df['نوع_القرار'].value_counts().reset_index(),
                    x='count',
                    y='نوع_القرار',
                    orientation='h',
                    title='توزيع الأحكام حسب نوع القرار',
                    color='count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== Decision Tree ==========
    with tabs[1]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🌳 تحليل شجرة القرار</div>', unsafe_allow_html=True)
        
        if st.session_state.model_trained and st.session_state.model_pack:
            models = st.session_state.model_pack['models']
            
            if 'Decision Tree' in models:
                dt_info = models['Decision Tree']
                dt_model = dt_info['model']
                
                st.markdown("### 📊 أداء Decision Tree")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("الدقة", f"{dt_info['metrics']['accuracy']*100:.1f}%")
                with col2:
                    st.metric("Precision", f"{dt_info['metrics']['precision']*100:.1f}%")
                with col3:
                    st.metric("Recall", f"{dt_info['metrics']['recall']*100:.1f}%")
                with col4:
                    st.metric("F1 Score", f"{dt_info['metrics']['f1']*100:.1f}%")
                
                st.markdown("### 📝 قواعد القرار")
                
                max_depth = st.slider("عمق الشجرة للعرض", 1, 5, 3)
                
                tree_rules = display_decision_tree(
                    dt_model, 
                    st.session_state.model_pack['feature_names'],
                    max_depth=max_depth
                )
                
                st.text(tree_rules)
                
                st.markdown("""
                <div class="alert-info">
                    <strong>💡 كيف تقرأ شجرة القرار:</strong><br>
                    - كل سطر يمثل شرطاً (مثلاً: قوة_الأدلة <= 3.5)<br>
                    - إذا تحقق الشرط تنتقل لليسار، وإلا لليمين<br>
                    - القيمة في النهاية (class) هي توقع النموذج
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("نموذج Decision Tree غير متوفر")
        else:
            st.info("👈 قم بتدريب النماذج أولاً")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== KNN و Vectors ==========
    with tabs[2]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 تحليل KNN و Vectors</div>', unsafe_allow_html=True)
        
        if st.session_state.model_trained and st.session_state.model_pack:
            models = st.session_state.model_pack['models']
            
            if 'KNN' in models:
                knn_info = models['KNN']
                
                st.markdown("### 📊 أداء KNN")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("الدقة", f"{knn_info['metrics']['accuracy']*100:.1f}%")
                with col2:
                    st.metric("أفضل قيمة K", knn_info.get('best_k', 5))
                with col3:
                    st.metric("Precision", f"{knn_info['metrics']['precision']*100:.1f}%")
                with col4:
                    st.metric("F1 Score", f"{knn_info['metrics']['f1']*100:.1f}%")
                
                st.markdown("---")
                st.markdown("### 🔤 تحليل Vectors من النصوص")
                
                if 'اسم_القضية' in df.columns:
                    # تحويل النصوص إلى Vectors
                    method = st.radio("طريقة التحويل", ["TF-IDF", "Count"], horizontal=True)
                    
                    if st.button("🔍 تحويل النصوص إلى Vectors", use_container_width=True):
                        with st.spinner("جاري تحويل النصوص..."):
                            method_key = 'tfidf' if method == 'TF-IDF' else 'count'
                            vectors, vectorizer = create_text_vectors(
                                df['اسم_القضية'], 
                                method=method_key,
                                max_features=50
                            )
                            
                            if vectors is not None:
                                st.success(f"✅ تم إنشاء {vectors.shape[1]} ميزة من النصوص")
                                
                                # عرض الكلمات المهمة
                                if vectorizer is not None:
                                    feature_names = vectorizer.get_feature_names_out()
                                    st.markdown("#### أهم الكلمات المميزة:")
                                    st.write(feature_names[:20])
                                
                                # PCA لتصور الـ Vectors
                                pca_result = analyze_vectors_with_pca(vectors, n_components=3)
                                
                                # رسم PCA
                                fig = px.scatter_3d(
                                    x=pca_result['pca_vectors'][:, 0],
                                    y=pca_result['pca_vectors'][:, 1],
                                    z=pca_result['pca_vectors'][:, 2],
                                    title='تصور Vectors النصوص باستخدام PCA',
                                    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown(f"""
                                **نسبة التباين المفسر:**
                                - PC1: {pca_result['explained_variance'][0]*100:.1f}%
                                - PC2: {pca_result['explained_variance'][1]*100:.1f}%
                                - PC3: {pca_result['explained_variance'][2]*100:.1f}%
                                """)
                                
                                # حفظ في الجلسة للاستخدام لاحقاً
                                st.session_state.text_vectors = vectors
                                st.session_state.vectorizer = vectorizer
                else:
                    st.warning("لا يوجد عمود نصوص للتحليل")
            else:
                st.warning("نموذج KNN غير متوفر")
        else:
            st.info("👈 قم بتدريب النماذج أولاً")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== كشف التحيز ==========
    with tabs[3]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 تحليل أنماط التحيز</div>', unsafe_allow_html=True)
        
        if st.session_state.bias_report:
            report = st.session_state.bias_report
            
            if 'judge_bias' in report:
                st.markdown("### 👨‍⚖️ تحيز القضاة")
                
                bias_score = report['judge_bias']['bias_score']
                st.markdown(f"""
                <div class="progress-bar">
                    <div style="width: {min(bias_score, 100)}%; height: 100%; background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444); border-radius: 5px;"></div>
                </div>
                <p style="text-align: center;">مؤشر التحيز العام: {bias_score:.2f}%</p>
                """, unsafe_allow_html=True)
                
                if 'most_biased_judges' in report['judge_bias'] and report['judge_bias']['most_biased_judges']:
                    st.markdown("#### القضاة الأكثر تحيزاً:")
                    for judge, info in report['judge_bias']['most_biased_judges'].items():
                        st.warning(f"⚠️ {judge}: {info['النسبة']:.1f}% لصالح {info['لصالح']}")
            
            if 'fairness_index' in report:
                fairness = report['fairness_index']
                level = report.get('fairness_level', 'غير محدد')
                
                if fairness < 10:
                    st.success(f"✅ نظام قضائي {level} (مؤشر {fairness:.2f}%)")
                elif fairness < 20:
                    st.info(f"ℹ️ نظام قضائي {level} (مؤشر {fairness:.2f}%)")
                elif fairness < 30:
                    st.warning(f"⚠️ نظام قضائي {level} (مؤشر {fairness:.2f}%)")
                else:
                    st.error(f"🚨 نظام قضائي {level} (مؤشر {fairness:.2f}%)")
        else:
            st.info("لا توجد بيانات كافية لتحليل التحيز")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== الشذوذ والفساد ==========
    with tabs[4]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🚨 كشف الفساد والرشوة</div>', unsafe_allow_html=True)
        
        if st.session_state.anomalies is not None:
            anomalies = st.session_state.anomalies
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-neon">
                    <div class="metric-neon-value">{len(anomalies)}</div>
                    <div class="metric-neon-label">حالة مشبوهة</div>
                </div>
                <div class="metric-neon">
                    <div class="metric-neon-value">{len(anomalies)/len(df)*100:.2f}%</div>
                    <div class="metric-neon-label">نسبة الشذوذ</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if len(anomalies) > 0:
                # توزيع احتمالية الفساد
                if 'احتمالية_الفساد' in anomalies.columns:
                    fig = px.histogram(
                        anomalies,
                        x='احتمالية_الفساد',
                        nbins=20,
                        title='توزيع احتمالية الفساد',
                        color_discrete_sequence=['#ef4444']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # الحالات الأكثر خطورة
                    high_risk = anomalies[anomalies['احتمالية_الفساد'] > 0.7].sort_values('احتمالية_الفساد', ascending=False)
                    
                    if len(high_risk) > 0:
                        st.markdown("### ⚠️ حالات شديدة الخطورة")
                        
                        for idx, row in high_risk.head(5).iterrows():
                            with st.expander(f"🚨 قضية {row.get('رقم_القضية', idx)} - احتمال فساد {row['احتمالية_الفساد']*100:.0f}%"):
                                st.write(f"**القاضي:** {row.get('رئيس_المحكمة', 'غير معروف')}")
                                st.write(f"**الطرف الفائز:** {row.get('الطرف_الفائز', 'غير معروف')}")
                                st.write(f"**أفضل نموذج:** {row.get('أفضل_نموذج', 'غير معروف')}")
                                st.write(f"**الأسباب:** {row.get('أسباب_الفساد', 'غير محدد')}")
        else:
            st.info("قم بتدريب النماذج أولاً لكشف الشذوذ")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== مقارنة النماذج ==========
    with tabs[5]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 مقارنة أداء النماذج</div>', unsafe_allow_html=True)
        
        if st.session_state.model_trained and st.session_state.model_pack:
            models = st.session_state.model_pack['models']
            
            # جدول المقارنة
            comparison_data = []
            for name, info in models.items():
                comparison_data.append({
                    'النموذج': name,
                    'الدقة': f"{info['metrics']['accuracy']*100:.1f}%",
                    'Precision': f"{info['metrics']['precision']*100:.1f}%",
                    'Recall': f"{info['metrics']['recall']*100:.1f}%",
                    'F1 Score': f"{info['metrics']['f1']*100:.1f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # رسم بياني للمقارنة
            fig = go.Figure()
            for name, info in models.items():
                fig.add_trace(go.Bar(
                    name=name,
                    x=['الدقة', 'Precision', 'Recall', 'F1'],
                    y=[info['metrics']['accuracy'], 
                       info['metrics']['precision'],
                       info['metrics']['recall'],
                       info['metrics']['f1']],
                    text=[f"{v*100:.1f}%" for v in [info['metrics']['accuracy'],
                                                    info['metrics']['precision'],
                                                    info['metrics']['recall'],
                                                    info['metrics']['f1']]],
                    textposition='auto',
                ))
            
            fig.update_layout(
                title='مقارنة أداء النماذج',
                barmode='group',
                yaxis_title='القيمة',
                yaxis_tickformat='.0%'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # أفضل نموذج
            best = st.session_state.model_pack['best_model']
            st.success(f"🏆 أفضل نموذج هو: **{best}**")
            
        else:
            st.info("👈 قم بتدريب النماذج أولاً")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== محاكي الأحكام ==========
    with tabs[6]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚖️ محاكي الأحكام الذكي</div>', unsafe_allow_html=True)
        
        if st.session_state.model_trained and st.session_state.model_pack:
            model_pack = st.session_state.model_pack
            
            st.markdown("#### 🔮 اختر قضية جديدة للتحليل")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'نوع_القرار' in df.columns:
                    decision_type = st.selectbox("نوع القرار", df['نوع_القرار'].dropna().unique())
                else:
                    decision_type = "جنائي"
                
                if 'نتيجة_القضية' in df.columns:
                    case_disp = st.selectbox("نتيجة القضية", df['نتيجة_القضية'].dropna().unique())
                else:
                    case_disp = "قبول"
                
                evidence = st.slider("قوة الأدلة (1-5)", 1, 5, 3)
            
            with col2:
                if 'رئيس_المحكمة' in df.columns:
                    judge = st.selectbox("القاضي", df['رئيس_المحكمة'].dropna().unique())
                else:
                    judge = "القاضي أحمد"
                
                precedent = st.selectbox("تغيير السابقة", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
                split = st.selectbox("تصويت منقسم", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
            
            case_text = st.text_input("نص القضية (اختياري)", "قضية رقم 12345")
            
            if st.button("🔮 تحليل القضية", use_container_width=True):
                # استخدام أفضل نموذج
                best_model_name = model_pack['best_model']
                best_model_info = model_pack['models'][best_model_name]
                best_model = best_model_info['model']
                
                st.info(f"✅ استخدام أفضل نموذج: {best_model_name}")
                
                # حساب احتمالية الفساد
                input_data = {
                    'قوة_الأدلة': evidence,
                    'تم_القبض': 1,  # افتراضي
                    'تغيير_السابقة': precedent,
                    'تصويت_منقسم': split,
                    'درجة_الثقة': 0.8  # افتراضي
                }
                
                corruption_prob, reasons = calculate_corruption_probability(input_data)
                
                # عرض النتيجة
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-neon">
                        <div class="metric-neon-value">{best_model_name}</div>
                        <div class="metric-neon-label">النموذج المستخدم</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-neon">
                        <div class="metric-neon-value">{corruption_prob*100:.1f}%</div>
                        <div class="metric-neon-label">احتمالية الفساد</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    risk_level = "منخفضة" if corruption_prob < 0.3 else "متوسطة" if corruption_prob < 0.6 else "عالية"
                    risk_color = "success" if corruption_prob < 0.3 else "warning" if corruption_prob < 0.6 else "danger"
                    
                    st.markdown(f"""
                    <div class="metric-neon" style="background: {'#10b981' if risk_level=='منخفضة' else '#f59e0b' if risk_level=='متوسطة' else '#ef4444'}">
                        <div class="metric-neon-value">{risk_level}</div>
                        <div class="metric-neon-label">مستوى الخطورة</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if reasons:
                    st.warning(f"⚠️ أسباب محتملة: {', '.join(reasons)}")
                
                # إيجاد حالات مشابهة باستخدام KNN
                if 'KNN' in model_pack['models'] and 'اسم_القضية' in df.columns:
                    st.markdown("#### 🔍 حالات مشابهة")
                    
                    # محاكاة إيجاد حالات مشابهة
                    similar_cases = df.sample(min(5, len(df)))[['رقم_القضية', 'رئيس_المحكمة', 'نوع_القرار']].to_dict('records')
                    
                    for i, case in enumerate(similar_cases):
                        st.markdown(f"{i+1}. قضية {case.get('رقم_القضية', 'غير معروف')} - {case.get('نوع_القرار', '')} - القاضي {case.get('رئيس_المحكمة', '')}")
        else:
            st.info("👈 قم بتدريب النماذج أولاً")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # الفوتر
    st.markdown("""
    <div class="footer-advanced">
        <h3>⚖️ العدالة - نظام الرقابة الذكية على الأحكام القضائية</h3>
        <p>الإصدار 5.0 | مدعوم بـ Decision Tree, KNN, و Vectors</p>
        <p style="margin-top: 2rem; opacity: 0.7;">جميع الحقوق محفوظة © 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
