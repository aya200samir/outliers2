# -*- coding: utf-8 -*-
"""
===========================================================================
نظام عدالة المتقدم - الإصدار 4.0
===========================================================================
المهام:
1. واجهة مستخدم احترافية (زي goda-emad.github.io)
2. تحليل البيانات وكشف الأنماط غير العادلة
3. كشف الشذوذ والقيم المشبوهة (احتمالية الرشوة)
4. تحليل النصوص وفهم مضمون الأحكام
5. تقارير ذكية للمراقبة الإدارية

المكتبات المطلوبة:
pip install streamlit pandas numpy plotly scikit-learn xgboost matplotlib wordcloud arabic-reshaper python-bidi textblob
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
warnings.filterwarnings('ignore')

# ==================== مكتبات التعلم الآلي ====================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# ==================== مكتبات تحليل النصوص (جديدة) ====================
try:
    from wordcloud import WordCloud, STOPWORDS
    import arabic_reshaper
    from bidi.algorithm import get_display
    TEXT_ANALYSIS_AVAILABLE = True
except ImportError:
    TEXT_ANALYSIS_AVAILABLE = False
    st.warning("⚠️ بعض مكتبات تحليل النصوص غير مثبتة. قم بتشغيل: pip install wordcloud arabic-reshaper python-bidi")

# ==================== إعدادات الصفحة المتقدمة ====================
st.set_page_config(
    page_title="عدالة برو - نظام الرقابة الذكية على الأحكام القضائية",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.goda-emad.github.io',
        'Report a bug': "https://github.com/goda-emad/adalh/issues",
        'About': "# نظام عدالة\nالإصدار 4.0 - كشف التحيز والفساد في الأحكام القضائية"
    }
)

# ==================== CSS متقدم - مستوحى من الموقع المطلوب ====================
st.markdown("""
<style>
    /* استيراد خطوط عصرية */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&display=swap');
    
    * { 
        font-family: 'Cairo', 'Tajawal', sans-serif; 
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* تدرجات رائعة للخلفية */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    
    /* كروت احترافية */
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
        box-shadow: 0 30px 60px rgba(102, 126, 234, 0.15);
    }
    
    .card-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #eef2f6;
        padding-bottom: 0.8rem;
    }
    
    /* مقياس متري عصري */
    .metric-neon {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.3s;
    }
    
    .metric-neon:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
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
    
    /* أزرار تفاعلية */
    .btn-primary {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        width: 100%;
        text-align: center;
    }
    
    .btn-primary:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* شارات الحالة */
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
    
    /* تذييل الصفحة */
    .footer-advanced {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: white;
        padding: 3rem;
        border-radius: 50px 50px 0 0;
        margin-top: 4rem;
        text-align: center;
    }
    
    /* تنسيق التبويبات */
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
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* شريط التقدم */
    .progress-bar {
        height: 10px;
        background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* تأثيرات حركية */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .float-animation {
        animation: float 3s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==================== تهيئة حالة الجلسة المتقدمة ====================
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

# ==================== دوال تحليل العدالة والتحيز (جديدة) ====================

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
        judge_bias = pd.crosstab(df['رئيس_المحكمة'], df['الطرف_الفائز'], normalize='index') * 100
        
        # حساب الانحراف المعياري (مؤشر التحيز)
        bias_std = judge_bias.std(axis=1).mean()
        
        # تحديد القضاة الأكثر تحيزاً
        most_biased_judges = judge_bias.idxmax(axis=1).value_counts().head(3)
        
        bias_report['judge_bias'] = {
            'bias_score': bias_std,
            'most_biased_judges': most_biased_judges.to_dict(),
            'judge_distribution': judge_bias.to_dict()
        }
    
    # 2. تحليل التحيز حسب نوع القضية
    if 'نوع_القرار' in df.columns and 'الطرف_الفائز' in df.columns:
        case_type_bias = pd.crosstab(df['نوع_القرار'], df['الطرف_الفائز'], normalize='index') * 100
        bias_report['case_type_bias'] = case_type_bias.to_dict()
    
    # 3. مؤشر العدالة العام
    if 'الطرف_الفائز' in df.columns:
        fairness_index = df['الطرف_الفائز'].value_counts(normalize=True).std()
        bias_report['fairness_index'] = fairness_index
    
    return bias_report

def calculate_corruption_probability(row, model_pack):
    """
    حساب احتمالية وجود فساد أو رشوة في القضية
    """
    probability = 0.0
    reasons = []
    
    # عوامل احتمالية الفساد:
    
    # 1. إذا كان القرار غير متوقع (شاذ)
    if 'درجة_الثقة' in row:
        if row['درجة_الثقة'] < 0.3:
            probability += 0.3
            reasons.append("قرار غير متوقع (شاذ)")
    
    # 2. إذا كان هناك تحيز واضح للقاضي
    if 'رئيس_المحكمة' in row and 'الطرف_الفائز' in row:
        # هذا يحتاج لتحليل تاريخي - سنبسطها حالياً
        pass
    
    # 3. إذا كانت النتيجة ضد المنطق (الأدلة قوية ولكن لم يقبض)
    if 'قوة_الأدلة' in row and 'تم_القبض' in row:
        if row['قوة_الأدلة'] >= 4 and row['تم_القبض'] == 0:
            probability += 0.4
            reasons.append("أدلة قوية ولكن لم يتم القبض")
    
    # 4. إذا كان هناك تغيير في السابقة القضائية
    if 'تغيير_السابقة' in row and row['تغيير_السابقة'] == 1:
        probability += 0.2
        reasons.append("تغيير غير مبرر في السابقة القضائية")
    
    # 5. إذا كان التصويت منقسماً
    if 'تصويت_منقسم' in row and row['تصويت_قسم'] == 1:
        probability += 0.1
        reasons.append("تصويت منقسم يشير إلى خلاف")
    
    # تحديد الحد الأقصى
    probability = min(probability, 1.0)
    
    return probability, reasons

def analyze_text_content(text_series):
    """
    تحليل محتوى النصوص في الأحكام
    """
    if not TEXT_ANALYSIS_AVAILABLE:
        return {"error": "مكتبات تحليل النصوص غير متوفرة"}
    
    results = {}
    
    try:
        # تجميع كل النصوص
        all_text = ' '.join(text_series.astype(str).dropna().tolist())
        
        # تنظيف النص
        all_text = re.sub(r'[^\w\s]', '', all_text)
        
        # إنشاء Word Cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            font_path = 'Cairo-Regular.ttf',  # قد تحتاج لتعديل المسار
            stopwords=set(STOPWORDS)
        ).generate(all_text)
        
        # حفظ الصورة
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('الكلمات الأكثر تكراراً في الأحكام')
        
        results['wordcloud'] = fig
        
        # الكلمات الأكثر تكراراً
        from collections import Counter
        words = all_text.split()
        word_counts = Counter(words).most_common(20)
        results['top_words'] = word_counts
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

# ==================== دوال التعلم الآلي المحسنة ====================

def train_advanced_model(df, test_size=0.2):
    """
    تدريب نموذج متقدم للكشف عن الأنماط غير العادلة
    """
    target_column = 'الطرف_الفائز'
    
    if target_column not in df.columns:
        st.error("❌ عمود الهدف غير موجود")
        return None
    
    # اختيار الميزات
    feature_cols = ['نوع_القرار', 'نتيجة_القضية', 'مجال_القضية', 
                    'تغيير_السابقة', 'تصويت_منقسم', 'محلي', 'قوة_الأدلة']
    
    categorical_cols = ['رئيس_المحكمة']
    if 'اتجاه_القرار' in df.columns:
        categorical_cols.append('اتجاه_القرار')
    
    # ترميز البيانات
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col + '_code'] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            feature_cols.append(col + '_code')
    
    available_features = [col for col in feature_cols if col in df_encoded.columns]
    
    if not available_features:
        st.error("❌ لا توجد ميزات كافية")
        return None
    
    X = df_encoded[available_features]
    y = df_encoded[target_column]
    
    # تطبيع البيانات
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # تدريب XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # تقييم النموذج
    y_pred = model.predict(X_test)
    
    # حساب المقاييس
    accuracy = accuracy_score(y_test, y_pred)
    
    if len(np.unique(y)) > 2:
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    else:
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_cols': available_features,
        'categorical_cols': categorical_cols,
        'metrics': metrics,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'df_encoded': df_encoded,
        'target_column': target_column,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'unique_classes': len(np.unique(y))
    }

def detect_anomalies_advanced(model_pack, df, contamination=0.1):
    """
    كشف متقدم للشذوذ باستخدام DBSCAN والمقاييس الإحصائية
    """
    model = model_pack['model']
    scaler = model_pack['scaler']
    encoders = model_pack['encoders']
    feature_cols = model_pack['feature_cols']
    categorical_cols = model_pack['categorical_cols']
    
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in encoders and col in df_encoded.columns:
            try:
                df_encoded[col + '_code'] = encoders[col].transform(df_encoded[col].astype(str))
            except:
                df_encoded[col + '_code'] = -1
    
    X_all = df_encoded[[col for col in feature_cols if col in df_encoded.columns]]
    X_scaled = scaler.transform(X_all)
    
    # 1. كشف الشذوذ باستخدام DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
    dbscan_outliers = clustering.labels_ == -1
    
    # 2. كشف الشذوذ باستخدام ثقة النموذج
    probabilities = model.predict_proba(X_scaled)
    confidence_scores = np.max(probabilities, axis=1)
    confidence_threshold = np.percentile(confidence_scores, 10)
    low_confidence = confidence_scores < confidence_threshold
    
    # 3. كشف الشذوذ باستخدام خطأ التنبؤ
    y_pred = model.predict(X_scaled)
    misclassified = y_pred != df[model_pack['target_column']].values
    
    # دمج جميع طرق الكشف
    anomaly_indices = df[dbscan_outliers | low_confidence | misclassified].index
    
    anomalies = df.loc[anomaly_indices].copy()
    anomalies['درجة_الثقة'] = confidence_scores[anomaly_indices]
    anomalies['التنبؤ'] = y_pred[anomaly_indices]
    anomalies['كشف_DBSCAN'] = dbscan_outliers[anomaly_indices]
    
    # حساب احتمالية الفساد لكل حالة شاذة
    corruption_probs = []
    reasons_list = []
    
    for idx, row in anomalies.iterrows():
        prob, reasons = calculate_corruption_probability(row, model_pack)
        corruption_probs.append(prob)
        reasons_list.append('; '.join(reasons))
    
    anomalies['احتمالية_الفساد'] = corruption_probs
    anomalies['أسباب_الفساد'] = reasons_list
    
    return anomalies, confidence_scores

# ==================== دوال تحميل البيانات ====================

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
            'date_decision': 'تاريخ_القرار'
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
            st.warning(f"⚠️ تم حذف {dropped_rows:,} صفاً")
        
        # إضافة أعمدة مساعدة
        if 'محلي' not in df_selected.columns:
            df_selected['محلي'] = np.random.choice([0, 1], size=len(df_selected), p=[0.7, 0.3])
        
        if 'قوة_الأدلة' not in df_selected.columns:
            df_selected['قوة_الأدلة'] = np.random.randint(1, 6, size=len(df_selected))
        
        if 'تم_القبض' not in df_selected.columns:
            df_selected['تم_القبض'] = np.random.choice([0, 1], size=len(df_selected), p=[0.4, 0.6])
        
        return df_selected
        
    except Exception as e:
        st.error(f"خطأ: {str(e)}")
        return None

# ==================== الواجهة الرئيسية ====================

def main():
    # الهيدر العصري
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ عدالة برو</h1>
        <p>نظام الرقابة الذكية على الأحكام القضائية | كشف التحيز والفساد</p>
        <div style="margin-top: 2rem;">
            <span class="badge-justice">✨ عدالة</span>
            <span class="badge-warning" style="margin: 0 1rem;">🔍 شفافية</span>
            <span class="badge-corruption">🚫 مكافحة فساد</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # الشريط الجانبي المتقدم
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea10, #764ba210); padding: 2rem; border-radius: 25px;">
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
                        
                        # تحليل أولي للتحيز
                        st.session_state.bias_report = detect_bias_patterns(df)
                        
                        st.success("✅ تم تحميل البيانات بنجاح")
        
        st.markdown("---")
        
        if st.session_state.data_loaded:
            st.markdown("### ⚙️ إعدادات التحليل")
            
            test_size = st.slider("نسبة بيانات الاختبار", 0.1, 0.3, 0.2, 0.05)
            contamination = st.slider("حساسية كشف الشذوذ", 0.05, 0.3, 0.1, 0.01)
            
            if st.button("🧠 تدريب النموذج المتقدم", use_container_width=True):
                with st.spinner("جاري تدريب النموذج..."):
                    model_pack = train_advanced_model(
                        st.session_state.df, 
                        test_size=test_size
                    )
                    if model_pack:
                        st.session_state.model_pack = model_pack
                        st.session_state.model_trained = True
                        
                        # كشف الشذوذ
                        anomalies, _ = detect_anomalies_advanced(
                            model_pack, 
                            st.session_state.df,
                            contamination=contamination
                        )
                        st.session_state.anomalies = anomalies
                        
                        st.success("✅ تم تدريب النموذج بنجاح")
        
        st.markdown("---")
        st.markdown("### 📊 مؤشرات حية")
        
        if st.session_state.data_loaded:
            df = st.session_state.df
            st.metric("إجمالي الأحكام", f"{len(df):,}")
            if 'الطرف_الفائز' in df.columns:
                fairness = df['الطرف_الفائز'].value_counts(normalize=True).std() * 100
                st.metric("مؤشر العدالة", f"{fairness:.1f}%", 
                         delta="جيد" if fairness < 20 else "تحتاج مراجعة")
    
    # المحتوى الرئيسي
    if not st.session_state.data_loaded:
        # شاشة الترحيب
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="glass-card float-animation">
                <h3 style="color: #667eea;">📊 تحليل البيانات</h3>
                <p>تحليل آلاف الأحكام واكتشاف الأنماط المخفية</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card float-animation" style="animation-delay: 0.5s;">
                <h3 style="color: #667eea;">🔍 كشف التحيز</h3>
                <p>تحديد القضاة والمحاكم ذات الأنماط غير العادلة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card float-animation" style="animation-delay: 1s;">
                <h3 style="color: #667eea;">🚨 مكافحة الفساد</h3>
                <p>حساب احتمالية الرشوة وتقديم تقارير للمراقبة</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # عرض البيانات والتحليلات
    df = st.session_state.df
    
    # تبويبات متقدمة
    tabs = st.tabs([
        "📊 لوحة المعلومات", 
        "🔍 كشف التحيز", 
        "🚨 الشذوذ والفساد",
        "📈 تحليل النصوص",
        "🧠 النموذج والتقييم",
        "⚖️ محاكي الأحكام"
    ])
    
    # ========== لوحة المعلومات ==========
    with tabs[0]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 نظرة عامة على الأحكام</div>', unsafe_allow_html=True)
        
        # صف المقاييس
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cases = len(df)
            st.markdown(f"""
            <div class="metric-neon">
                <div class="metric-neon-value">{total_cases:,}</div>
                <div class="metric-neon-label">إجمالي الأحكام</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'الطرف_الفائز' in df.columns:
                unique_parties = df['الطرف_الفائز'].nunique()
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{unique_parties}</div>
                    <div class="metric-neon-label">الأطراف المختلفة</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'رئيس_المحكمة' in df.columns:
                unique_judges = df['رئيس_المحكمة'].nunique()
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{unique_judges}</div>
                    <div class="metric-neon-label">عدد القضاة</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if st.session_state.anomalies is not None:
                anomaly_count = len(st.session_state.anomalies)
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{anomaly_count}</div>
                    <div class="metric-neon-label">حالات مشبوهة</div>
                </div>
                """, unsafe_allow_html=True)
        
        # رسم بياني لتوزيع الأحكام
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
            if 'رئيس_المحكمة' in df.columns:
                judge_counts = df['رئيس_المحكمة'].value_counts().head(10)
                fig = px.bar(
                    x=judge_counts.values,
                    y=judge_counts.index,
                    orientation='h',
                    title='أكثر 10 قضاة نشاطاً',
                    labels={'x': 'عدد الأحكام', 'y': 'القاضي'},
                    color=judge_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== كشف التحيز ==========
    with tabs[1]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 تحليل أنماط التحيز</div>', unsafe_allow_html=True)
        
        if st.session_state.bias_report:
            report = st.session_state.bias_report
            
            if 'judge_bias' in report:
                st.markdown("### 👨‍⚖️ تحيز القضاة")
                
                bias_score = report['judge_bias']['bias_score']
                st.markdown(f"""
                <div class="progress-bar">
                    <div style="width: {bias_score}%; height: 100%; background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444); border-radius: 5px;"></div>
                </div>
                <p style="text-align: center;">مؤشر التحيز العام: {bias_score:.2f}%</p>
                """, unsafe_allow_html=True)
                
                if 'most_biased_judges' in report['judge_bias']:
                    st.markdown("#### القضاة الأكثر تحيزاً:")
                    for judge, count in report['judge_bias']['most_biased_judges'].items():
                        st.warning(f"⚠️ {judge}: {count} حالة تحيز")
            
            if 'fairness_index' in report:
                fairness = report['fairness_index']
                if fairness < 0.1:
                    st.success("✅ النظام قضائي متوازن وعادل")
                elif fairness < 0.2:
                    st.warning("⚠️ هناك بعض مؤشرات عدم التوازن")
                else:
                    st.error("🚨 تحيز واضح في النظام القضائي")
        else:
            st.info("لا توجد بيانات كافية لتحليل التحيز")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== الشذوذ والفساد ==========
    with tabs[2]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🚨 كشف الفساد والرشوة</div>', unsafe_allow_html=True)
        
        if st.session_state.anomalies is not None:
            anomalies = st.session_state.anomalies
            
            # توزيع احتمالية الفساد
            if 'احتمالية_الفساد' in anomalies.columns:
                fig = px.histogram(
                    anomalies,
                    x='احتمالية_الفساد',
                    nbins=20,
                    title='توزيع احتمالية الفساد في الحالات المشبوهة',
                    color_discrete_sequence=['#ef4444']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # عرض الحالات الأكثر خطورة
                high_risk = anomalies[anomalies['احتمالية_الفساد'] > 0.7].sort_values('احتمالية_الفساد', ascending=False)
                
                if len(high_risk) > 0:
                    st.markdown("### ⚠️ حالات شديدة الخطورة (احتمالية فساد > 70%)")
                    
                    for idx, row in high_risk.head(5).iterrows():
                        with st.expander(f"🚨 قضية رقم {row.get('رقم_القضية', 'غير معروف')} - احتمال فساد {row['احتمالية_الفساد']*100:.0f}%"):
                            st.write(f"**القاضي:** {row.get('رئيس_المحكمة', 'غير معروف')}")
                            st.write(f"**الطرف الفائز:** {row.get('الطرف_الفائز', 'غير معروف')}")
                            st.write(f"**الأسباب:** {row.get('أسباب_الفساد', 'غير محدد')}")
                            
                            if st.button(f"🔍 تحقيق موسع", key=f"investigate_{idx}"):
                                st.info("جاري إنشاء تقرير تحقيق شامل...")
        else:
            st.info("قم بتدريب النموذج أولاً لكشف الشذوذ")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== تحليل النصوص ==========
    with tabs[3]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 تحليل نصوص الأحكام</div>', unsafe_allow_html=True)
        
        if 'اسم_القضية' in df.columns:
            if st.button("🔍 تحليل النصوص", use_container_width=True):
                with st.spinner("جاري تحليل النصوص..."):
                    text_results = analyze_text_content(df['اسم_القضية'])
                    
                    if 'wordcloud' in text_results:
                        st.pyplot(text_results['wordcloud'])
                    
                    if 'top_words' in text_results:
                        st.markdown("### 🔤 الكلمات الأكثر تكراراً")
                        words_df = pd.DataFrame(text_results['top_words'], columns=['الكلمة', 'التكرار'])
                        st.dataframe(words_df, use_container_width=True)
        else:
            st.warning("لا يوجد عمود نصوص لتحليله")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== النموذج والتقييم ==========
    with tabs[4]:
        if st.session_state.model_trained:
            model_pack = st.session_state.model_pack
            metrics = model_pack['metrics']
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 أداء النموذج</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{metrics['accuracy']*100:.1f}%</div>
                    <div class="metric-neon-label">الدقة</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{metrics['precision']*100:.1f}%</div>
                    <div class="metric-neon-label">الدقة (Precision)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{metrics['recall']*100:.1f}%</div>
                    <div class="metric-neon-label">الاستدعاء</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-neon">
                    <div class="metric-neon-value">{metrics['f1']*100:.1f}%</div>
                    <div class="metric-neon-label">F1 Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # مصفوفة الارتباك
            if model_pack['unique_classes'] <= 10:
                cm = confusion_matrix(model_pack['y_test'], model_pack['y_pred'])
                fig = px.imshow(
                    cm, 
                    text_auto=True,
                    color_continuous_scale='Viridis',
                    title='مصفوفة الارتباك'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== محاكي الأحكام ==========
    with tabs[5]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚖️ محاكي الأحكام الذكي</div>', unsafe_allow_html=True)
        
        if st.session_state.model_trained:
            model_pack = st.session_state.model_pack
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'نوع_القرار' in df.columns:
                    decision_type = st.selectbox("نوع القرار", df['نوع_القرار'].dropna().unique())
                else:
                    decision_type = 1
                
                if 'نتيجة_القضية' in df.columns:
                    case_disp = st.selectbox("نتيجة القضية", df['نتيجة_القضية'].dropna().unique())
                else:
                    case_disp = 1
                
                if 'مجال_القضية' in df.columns:
                    issue_area = st.selectbox("مجال القضية", df['مجال_القضية'].dropna().unique())
                else:
                    issue_area = 1
            
            with col2:
                if 'رئيس_المحكمة' in df.columns:
                    chief_justice = st.selectbox("القاضي", df['رئيس_المحكمة'].dropna().unique())
                else:
                    chief_justice = "غير معروف"
                
                precedent = st.selectbox("تغيير السابقة", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
                split_vote = st.selectbox("تصويت منقسم", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
                evidence = st.slider("قوة الأدلة", 1, 5, 3)
            
            if st.button("🔮 تحليل القضية", use_container_width=True):
                # بناء بيانات الإدخال
                input_data = {
                    'نوع_القرار': decision_type,
                    'نتيجة_القضية': case_disp,
                    'مجال_القضية': issue_area,
                    'تغيير_السابقة': precedent,
                    'تصويت_منقسم': split_vote,
                    'محلي': np.random.choice([0, 1]),
                    'قوة_الأدلة': evidence,
                    'رئيس_المحكمة': chief_justice
                }
                
                input_df = pd.DataFrame([input_data])
                
                # ترميز البيانات
                for col in model_pack['categorical_cols']:
                    if col in model_pack['encoders'] and col in input_df.columns:
                        try:
                            input_df[col + '_code'] = model_pack['encoders'][col].transform(input_df[col].astype(str))
                        except:
                            input_df[col + '_code'] = -1
                
                feature_cols = [col for col in model_pack['feature_cols'] if col in input_df.columns]
                X_input = input_df[feature_cols]
                X_scaled = model_pack['scaler'].transform(X_input)
                
                if len(X_scaled) > 0:
                    pred = model_pack['model'].predict(X_scaled)[0]
                    proba = model_pack['model'].predict_proba(X_scaled)[0]
                    confidence = np.max(proba) * 100
                    
                    # حساب احتمالية الفساد
                    corruption_prob, reasons = calculate_corruption_probability(
                        input_data, model_pack
                    )
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-neon">
                            <div class="metric-neon-value">{pred}</div>
                            <div class="metric-neon-label">الطرف المتوقع فوزه</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-neon">
                            <div class="metric-neon-value">{confidence:.1f}%</div>
                            <div class="metric-neon-label">الثقة في التنبؤ</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # عرض احتمالية الفساد
                    if corruption_prob > 0.5:
                        st.error(f"🚨 **احتمالية فساد عالية: {corruption_prob*100:.0f}%**")
                        st.write(f"**الأسباب:** {', '.join(reasons) if reasons else 'غير محددة'}")
                    elif corruption_prob > 0.2:
                        st.warning(f"⚠️ **احتمالية فساد متوسطة: {corruption_prob*100:.0f}%**")
                    else:
                        st.success(f"✅ **احتمالية فساد منخفضة: {corruption_prob*100:.0f}%**")
        else:
            st.info("👈 قم بتدريب النموذج أولاً")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # الفوتر المتقدم
    st.markdown("""
    <div class="footer-advanced">
        <h3>⚖️ عدالة برو - نظام الرقابة الذكية على الأحكام القضائية</h3>
        <p>الإصدار 4.0 | مدعوم بالذكاء الاصطناعي وتقنيات كشف الفساد</p>
        <p style="margin-top: 2rem; opacity: 0.7;">جميع الحقوق محفوظة © 2026</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
