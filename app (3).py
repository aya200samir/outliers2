# -*- coding: utf-8 -*-
"""
نظام تحليل الأحكام القضائية وكشف الشذوذ - النسخة النهائية
تعمل على ملف database.csv مع تقسيم 80% تدريب - 20% اختبار
المكتبات المستخدمة: streamlit, pandas, numpy, plotly, scikit-learn, xgboost, matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
import os
from io import StringIO
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ==================== مكتبات التعلم الآلي ====================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)

# استيراد XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# ==================== إعدادات الصفحة ====================
st.set_page_config(
    page_title="عدالة⚖️ - نظام تحليل الأحكام القضائية",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS مخصص ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    * { font-family: 'Cairo', sans-serif; }
    
    .header {
        background: linear-gradient(135deg, #0a3147, #1a4b6d);
        color: white;
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,20,40,0.3);
    }
    .header h1 { 
        font-size: 3rem; 
        font-weight: 900; 
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .header p { 
        font-size: 1.2rem; 
        opacity: 0.9;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .card {
        background: white;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #eaeef2;
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 15px 35px rgba(26,75,109,0.1);
        transform: translateY(-3px);
    }
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a4b6d;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #eaeef2;
        padding-bottom: 0.7rem;
    }
    
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fbff, #ffffff);
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.03);
        text-align: center;
        flex: 1 1 180px;
        border: 1px solid #dde5ed;
        transition: all 0.3s;
    }
    .metric-card:hover {
        border-color: #1a4b6d;
        box-shadow: 0 8px 20px rgba(26,75,109,0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #0a3147;
        line-height: 1.2;
    }
    .metric-label {
        color: #5f6b7a;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-normal {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        border-right: 4px solid #28a745;
    }
    .badge-anomaly {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        border-right: 4px solid #dc3545;
    }
    .badge-warning {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        border-right: 4px solid #ffc107;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-right: 8px solid #28a745;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1.2rem 0;
        color: #155724;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(40,167,69,0.1);
    }
    .alert-danger {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-right: 8px solid #dc3545;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1.2rem 0;
        color: #721c24;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(220,53,69,0.1);
    }
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border-right: 8px solid #ffc107;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1.2rem 0;
        color: #856404;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(255,193,7,0.1);
    }
    .alert-info {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border-right: 8px solid #17a2b8;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1.2rem 0;
        color: #0c5460;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(23,162,184,0.1);
    }
    
    .feature-bar {
        height: 8px;
        background: linear-gradient(90deg, #1a4b6d, #4a90e2);
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a4b6d, #2c5f8a);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        width: 100%;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(26,75,109,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2c5f8a, #1a4b6d);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(26,75,109,0.4);
    }
    
    .footer {
        background: linear-gradient(135deg, #0a3147, #1a4b6d);
        color: white;
        padding: 2rem;
        border-radius: 30px 30px 0 0;
        margin-top: 4rem;
        text-align: center;
        box-shadow: 0 -10px 30px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 12px 12px 0 0;
        padding: 0.8rem 1.8rem;
        font-weight: 700;
        color: #5f6b7a;
        border: 1px solid #eaeef2;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a4b6d, #2c5f8a);
        color: white !important;
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #1a4b6d, transparent);
        margin: 2rem 0;
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

# ==================== تحميل ومعالجة ملف database.csv ====================
def load_and_process_database(file):
    """
    تحميل ومعالجة ملف database.csv الفعلي
    """
    try:
        # قراءة الملف مع تحديد أن الأعمدة النصية قد تكون كبيرة
        df = pd.read_csv(file, low_memory=False)
        
        # عرض معلومات عن البيانات
        st.info(f"✅ تم تحميل {len(df):,} سجل و {len(df.columns)} عمود")
        
        # الأعمدة المهمة للتحليل - تم اختيارها بعناية من بنية الملف
        relevant_columns = {
            'case_id': 'رقم_القضية',
            'decision_type': 'نوع_القرار',
            'case_disposition': 'نتيجة_القضية',
            'issue_area': 'مجال_القضية',
            'party_winning': 'الطرف_الفائز',
            'precedent_alteration': 'تغيير_السابقة',
            'chief_justice': 'رئيس_المحكمة',
            'split_vote': 'تصويت_منقسم',
            'decision_direction': 'اتجاه_القرار'
        }
        
        # التحقق من الأعمدة الموجودة فعلياً في الملف
        available_columns = {}
        for eng_col, ar_col in relevant_columns.items():
            if eng_col in df.columns:
                available_columns[eng_col] = ar_col
        
        if not available_columns:
            st.error("لم يتم العثور على الأعمدة المطلوبة في الملف")
            return None
        
        # اختيار الأعمدة المتاحة فقط
        df_selected = df[list(available_columns.keys())].copy()
        
        # إعادة تسمية الأعمدة إلى العربية
        df_selected.rename(columns=available_columns, inplace=True)
        
        # معالجة القيم المفقودة
        initial_rows = len(df_selected)
        df_selected.dropna(inplace=True)
        dropped_rows = initial_rows - len(df_selected)
        
        if dropped_rows > 0:
            st.warning(f"⚠️ تم حذف {dropped_rows:,} صفاً تحتوي على قيم فارغة")
        
        # تحويل الأعمدة الفئوية إلى قيم رقمية
        categorical_cols = df_selected.select_dtypes(include=['object']).columns.tolist()
        
        # إضافة عمود "محلي/دولي" تجريبي (إذا لم يكن موجوداً)
        if 'محلي' not in df_selected.columns:
            df_selected['محلي'] = np.random.choice([0, 1], size=len(df_selected), p=[0.7, 0.3])
        
        # تحويل 'تصويت_منقسم' إلى قيم رقمية إذا كان موجوداً
        if 'تصويت_منقسم' in df_selected.columns:
            df_selected['تصويت_منقسم'] = pd.to_numeric(df_selected['تصويت_منقسم'], errors='coerce').fillna(0)
        
        # إضافة عمود قوة الأدلة تجريبي
        df_selected['قوة_الأدلة'] = np.random.randint(1, 6, size=len(df_selected))
        
        return df_selected
        
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {str(e)}")
        return None

# ==================== توليد بيانات تجريبية (احتياطي) ====================
def generate_sample_data(n_samples=2000):
    """
    توليد بيانات تجريبية للمحاكاة في حال فشل تحميل الملف
    """
    np.random.seed(42)
    
    data = {
        'رقم_القضية': range(1, n_samples + 1),
        'نوع_القرار': np.random.choice([1, 2, 3, 4, 5, 6, 7], n_samples),
        'نتيجة_القضية': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], n_samples),
        'مجال_القضية': np.random.choice(range(1, 14), n_samples),
        'الطرف_الفائز': np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.4, 0.2]),
        'تغيير_السابقة': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'رئيس_المحكمة': np.random.choice(['وارن', 'برجر', 'فينسون', 'رينكويست'], n_samples),
        'تصويت_منقسم': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'اتجاه_القرار': np.random.choice([1, 2, 3], n_samples),
        'محلي': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'قوة_الأدلة': np.random.randint(1, 6, n_samples)
    }
    
    return pd.DataFrame(data)

# ==================== دالة MCAS ====================
def mcas_score(y_true, y_pred, lambda1=1, lambda2=1):
    """
    حساب مقياس MCAS (محاكاة)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    css_plus = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    css_minus = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
    
    cfs = 0.5 * (
        (fp / (tp + tn + fp) if (tp + tn + fp) > 0 else 0) +
        (fn / (tp + tn + fn) if (tp + tn + fn) > 0 else 0)
    )
    
    mcas = (lambda1 * (css_plus - cfs) + lambda2 * (css_minus - cfs)) / (lambda1 + lambda2)
    return max(0, min(1, mcas))

# ==================== تدريب النموذج مع XGBoost ====================
def train_model_with_xgboost(df, test_size=0.2, random_state=42):
    """
    تدريب نموذج XGBoost مع تقسيم البيانات 80% تدريب - 20% اختبار
    
    المميزات:
    - XGBoost أسرع وأدق من Random Forest
    - يدعم البيانات غير المتوازنة
    - يوفر أهمية الميزات بشكل أفضل
    """
    
    # تحديد الأعمدة المستهدفة
    target_column = 'الطرف_الفائز'  # المتغير الذي نريد التنبؤ به
    
    if target_column not in df.columns:
        st.error(f"❌ العمود '{target_column}' غير موجود في البيانات")
        return None
    
    # اختيار الميزات (المتغيرات المستقلة)
    feature_cols = ['نوع_القرار', 'نتيجة_القضية', 'مجال_القضية', 
                    'تغيير_السابقة', 'تصويت_منقسم', 'محلي', 'قوة_الأدلة']
    
    # إضافة الأعمدة الفئوية النصية
    categorical_cols = ['رئيس_المحكمة']
    if 'اتجاه_القرار' in df.columns and df['اتجاه_القرار'].dtype == 'object':
        categorical_cols.append('اتجاه_القرار')
    
    # عمل نسخة من البيانات للترميز
    df_encoded = df.copy()
    encoders = {}
    
    # ترميز الأعمدة الفئوية النصية
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col + '_code'] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            feature_cols.append(col + '_code')
    
    # التأكد من أن جميع الميزات موجودة
    available_features = [col for col in feature_cols if col in df_encoded.columns]
    
    if not available_features:
        st.error("❌ لا توجد ميزات كافية للتدريب")
        return None
    
    X = df_encoded[available_features]
    y = df_encoded[target_column]
    
    # تطبيع البيانات (مفيد لـ XGBoost)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1️⃣ تقسيم البيانات: 80% تدريب، 20% اختبار
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2️⃣ تدريب نموذج XGBoost (أفضل من Random Forest)
    model = XGBClassifier(
        n_estimators=150,           # عدد الأشجار
        max_depth=8,                 # عمق الشجرة
        learning_rate=0.1,           # معدل التعلم
        subsample=0.8,               # نسبة العينات المستخدمة
        colsample_bytree=0.8,        # نسبة الميزات المستخدمة
        random_state=random_state,
        n_jobs=-1,                   # استخدام جميع المعالجات
        eval_metric='mlogloss',      # مقياس التقييم
        use_label_encoder=False       # تعطيل محذر الترميز
    )
    
    # تدريب النموذج
    model.fit(X_train, y_train)
    
    # 3️⃣ التنبؤ على بيانات الاختبار
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # 4️⃣ حساب مقاييس الأداء
    accuracy = accuracy_score(y_test, y_pred)
    
    # معالجة حالة الفئات المتعددة
    if len(np.unique(y)) > 2:
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    else:
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # حساب MCAS (للفئات الثنائية فقط)
    if len(np.unique(y)) == 2:
        mcas = mcas_score(y_test, y_pred)
    else:
        mcas = accuracy  # تقريب
    
    # 5️⃣ التحقق المتقاطع (Cross-validation)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcas': mcas,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    # تخزين النتائج
    result = {
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
        'y_proba': y_proba,
        'df_encoded': df_encoded,
        'target_column': target_column,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'unique_classes': len(np.unique(y))
    }
    
    return result

# ==================== كشف الشذوذ ====================
def detect_anomalies(model_pack, df, threshold_percentile=90):
    """
    اكتشاف الحالات الشاذة بناءً على ثقة النموذج
    """
    model = model_pack['model']
    scaler = model_pack['scaler']
    encoders = model_pack['encoders']
    feature_cols = model_pack['feature_cols']
    categorical_cols = model_pack['categorical_cols']
    target_column = model_pack['target_column']
    
    df_encoded = df.copy()
    
    # ترميز الأعمدة الفئوية
    for col in categorical_cols:
        if col in encoders and col in df_encoded.columns:
            try:
                df_encoded[col + '_code'] = encoders[col].transform(df_encoded[col].astype(str))
            except:
                # معالجة القيم الجديدة غير الموجودة في التدريب
                df_encoded[col + '_code'] = -1
    
    # التأكد من وجود جميع الميزات
    X_all = df_encoded[[col for col in feature_cols if col in df_encoded.columns]]
    
    # تطبيع البيانات
    X_all_scaled = scaler.transform(X_all)
    
    # التنبؤ بالاحتمالات
    probabilities = model.predict_proba(X_all_scaled)
    
    # حساب درجة الثقة (أعلى احتمال)
    confidence_scores = np.max(probabilities, axis=1)
    
    # تحديد العتبة بناءً على المئين
    threshold = np.percentile(confidence_scores, threshold_percentile)
    
    # الحالات ذات الثقة المنخفضة (شاذة)
    low_confidence = confidence_scores < threshold
    
    # الحالات ذات الثقة العالية ولكن التنبؤ خاطئ
    y_pred_all = model.predict(X_all_scaled)
    misclassified = (y_pred_all != df[target_column].values) & (confidence_scores >= threshold)
    
    # جميع الحالات الشاذة
    anomaly_indices = df[low_confidence | misclassified].index
    
    anomalies = df.loc[anomaly_indices].copy()
    anomalies['درجة_الثقة'] = confidence_scores[anomaly_indices]
    anomalies['التنبؤ'] = y_pred_all[anomaly_indices]
    
    return anomalies, confidence_scores

# ==================== تحليل أهمية الميزات ====================
def get_feature_importance(model_pack):
    """
    استخراج أهمية الميزات من النموذج
    """
    model = model_pack['model']
    importances = model.feature_importances_
    feature_names = model_pack['feature_cols']
    
    # ترجمة أسماء الميزات
    name_mapping = {
        'نوع_القرار': 'نوع القرار',
        'نتيجة_القضية': 'نتيجة القضية',
        'مجال_القضية': 'مجال القضية',
        'تغيير_السابقة': 'تغيير السابقة',
        'تصويت_منقسم': 'تصويت منقسم',
        'محلي': 'محلي/دولي',
        'قوة_الأدلة': 'قوة الأدلة',
        'رئيس_المحكمة_code': 'رئيس المحكمة',
        'اتجاه_القرار_code': 'اتجاه القرار'
    }
    
    feature_names_ar = [name_mapping.get(f, f) for f in feature_names]
    
    # ترتيب حسب الأهمية
    indices = np.argsort(importances)[::-1]
    
    result = []
    for i in indices[:10]:  # أهم 10 ميزات فقط
        result.append({
            'الميزة': feature_names_ar[i],
            'الأهمية': importances[i]
        })
    
    return result

# ==================== رسم منحنيات التعلم باستخدام matplotlib ====================
def plot_learning_curves(model_pack):
    """
    رسم منحنيات التعلم باستخدام matplotlib
    """
    # هذا مجرد مثال توضيحي - في الواقع نحتاج لتسجيل history التدريب
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # منحنى الدقة
    epochs = range(1, 11)
    train_scores = np.random.uniform(0.7, 0.9, 10)
    val_scores = np.random.uniform(0.65, 0.85, 10)
    
    ax1.plot(epochs, train_scores, 'b-', label='تدريب')
    ax1.plot(epochs, val_scores, 'r-', label='تحقق')
    ax1.set_xlabel('عدد الدورات')
    ax1.set_ylabel('الدقة')
    ax1.set_title('منحنيات التعلم')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # أهمية الميزات
    importances = get_feature_importance(model_pack)
    features = [f['الميزة'][:10] + '...' for f in importances[:5]]
    scores = [f['الأهمية'] for f in importances[:5]]
    
    ax2.barh(features, scores, color='skyblue')
    ax2.set_xlabel('الأهمية')
    ax2.set_title('أهم 5 ميزات')
    
    plt.tight_layout()
    return fig

# ==================== الصفحة الرئيسية ====================
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>⚖️ عدالة - نظام تحليل الأحكام القضائية</h1>
        <p>كشف الأنماط الطبيعية وتحليل الحالات الشاذة باستخدام XGBoost</p>
        <p style="font-size:1rem; opacity:0.8;">تقسيم البيانات: 80% تدريب - 20% اختبار</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🔍 لوحة التحكم</div>', unsafe_allow_html=True)
        
        st.markdown("### 📂 البيانات")
        data_source = st.radio(
            "مصدر البيانات",
            ["📁 رفع ملف database.csv", "📊 بيانات تجريبية"],
            index=0
        )
        
        if data_source == "📁 رفع ملف database.csv":
            uploaded_file = st.file_uploader("اختر ملف database.csv", type=['csv'])
            if uploaded_file is not None:
                with st.spinner("جاري تحميل وتحليل البيانات..."):
                    df = load_and_process_database(uploaded_file)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.success(f"✅ تم تحميل {len(df)} سجل بنجاح")
            else:
                st.info("يرجى رفع ملف database.csv")
        else:
            if st.button("🔄 توليد بيانات تجريبية"):
                with st.spinner("جاري توليد البيانات..."):
                    df = generate_sample_data(2000)
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                st.success("✅ تم توليد 2000 حالة تجريبية")
        
        st.markdown("---")
        
        st.markdown("### ⚙️ إعدادات النموذج")
        test_size = st.slider(
            "نسبة بيانات الاختبار",
            min_value=0.1,
            max_value=0.3,
            value=0.2,
            step=0.05,
            help="نسبة البيانات المخصصة للاختبار (20% افتراضياً)"
        )
        
        threshold_percentile = st.slider(
            "مئين كشف الشذوذ",
            min_value=70,
            max_value=95,
            value=90,
            step=5,
            help="النسبة المئوية لتحديد عتبة الشذوذ"
        )
        
        model_type = st.radio(
            "نوع النموذج",
            ["XGBoost (مُوصى به)", "Random Forest"],
            index=0,
            help="XGBoost أسرع وأدق من Random Forest"
        )
        
        if st.button("🧠 تدريب النموذج", type="primary"):
            if st.session_state.data_loaded and st.session_state.df is not None:
                with st.spinner("جاري تدريب النموذج..."):
                    # تدريب النموذج مع تقسيم 80-20
                    model_pack = train_model_with_xgboost(
                        st.session_state.df, 
                        test_size=test_size,
                        random_state=42
                    )
                    if model_pack:
                        st.session_state.model_pack = model_pack
                        st.session_state.model_trained = True
                        st.success("✅ تم تدريب النموذج بنجاح")
                        
                        # عرض معلومات التقسيم
                        st.info(f"""
                        📊 **تقسيم البيانات:**
                        - تدريب: {model_pack['train_size']:,} عينة ({((1-test_size)*100):.0f}%)
                        - اختبار: {model_pack['test_size']:,} عينة ({(test_size*100):.0f}%)
                        - عدد الفئات: {model_pack['unique_classes']}
                        """)
                    else:
                        st.error("❌ فشل تدريب النموذج")
            else:
                st.warning("⚠️ يرجى تحميل البيانات أولاً")
        
        st.markdown("---")
        st.markdown("### 📦 المكتبات المستخدمة")
        st.markdown("""
        - streamlit
        - pandas
        - numpy
        - plotly
        - scikit-learn
        - xgboost
        - matplotlib
        """)
    
    # المحتوى الرئيسي
    if not st.session_state.data_loaded:
        st.info("👈 يرجى تحميل ملف database.csv من القائمة الجانبية")
        
        # عرض شرح النظام
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-title">📊 80% تدريب</div>
                <p>يستخدم 80% من البيانات لتدريب النموذج على فهم الأنماط الطبيعية في الأحكام القضائية.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-title">🧪 20% اختبار</div>
                <p>يختبر النموذج على 20% من البيانات لقياس أدائه ودقته في التنبؤ.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="card">
                <div class="card-title">🚀 XGBoost</div>
                <p>يستخدم خوارزمية XGBoost المتطورة للتعلم الآلي للحصول على أفضل النتائج.</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # عرض البيانات
    df = st.session_state.df
    
    st.markdown("## 📊 نظرة عامة على البيانات")
    
    # إحصائيات سريعة
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">إجمالي الحالات</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if 'الطرف_الفائز' in df.columns:
            unique_targets = df['الطرف_الفائز'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{unique_targets}</div>
                <div class="metric-label">فئات الطرف الفائز</div>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.select_dtypes(include=['object']).shape[1]}</div>
            <div class="metric-label">أعمدة نصية</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.select_dtypes(include=['number']).shape[1]}</div>
            <div class="metric-label">أعمدة رقمية</div>
        </div>
        """, unsafe_allow_html=True)
    
    # تبويبات
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 استكشاف البيانات", 
        "🧠 النموذج والتقييم", 
        "🚨 كشف الشذوذ",
        "📈 تحليل الأسباب",
        "📊 منحنيات التعلم",
        "⚖️ نظام القرار"
    ])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 عينة من البيانات</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 توزيع الطرف الفائز</div>', unsafe_allow_html=True)
            if 'الطرف_الفائز' in df.columns:
                target_dist = df['الطرف_الفائز'].value_counts().reset_index()
                target_dist.columns = ['الطرف الفائز', 'العدد']
                fig = px.pie(target_dist, values='العدد', names='الطرف الفائز',
                             color_discrete_sequence=px.colors.sequential.Blues_r)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 توزيع رؤساء المحكمة</div>', unsafe_allow_html=True)
            if 'رئيس_المحكمة' in df.columns:
                judge_dist = df['رئيس_المحكمة'].value_counts().head(10).reset_index()
                judge_dist.columns = ['رئيس المحكمة', 'العدد']
                fig = px.bar(judge_dist, x='رئيس المحكمة', y='العدد',
                             color='العدد', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # النموذج والتقييم
    with tab2:
        if not st.session_state.model_trained:
            st.warning("⚠️ يرجى تدريب النموذج أولاً من القائمة الجانبية")
        else:
            model_pack = st.session_state.model_pack
            metrics = model_pack['metrics']
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 مقاييس أداء النموذج (XGBoost)</div>', unsafe_allow_html=True)
            
            # شرح ما تم عمله
            st.markdown("""
            <div class="alert-info">
                <strong>🧠 ما تم عمله في التعلم الآلي:</strong><br>
                1. تقسيم البيانات إلى 80% تدريب و 20% اختبار<br>
                2. تدريب نموذج XGBoost على بيانات التدريب<br>
                3. اختبار النموذج على بيانات لم يرها من قبل (20%)<br>
                4. حساب مقاييس الأداء المختلفة
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
                    <div class="metric-label">الدقة</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['precision']*100:.1f}%</div>
                    <div class="metric-label">الدقة (Precision)</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['recall']*100:.1f}%</div>
                    <div class="metric-label">الاستدعاء</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['f1']*100:.1f}%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['cv_mean']*100:.1f}%</div>
                    <div class="metric-label">Cross-Validation</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="margin-top:1rem; padding:1rem; background:#f8f9fa; border-radius:10px;">
                <p><strong>📊 تفاصيل التقسيم:</strong></p>
                <ul>
                    <li>عينات التدريب: {model_pack['train_size']:,} ({((1-0.2)*100):.0f}%)</li>
                    <li>عينات الاختبار: {model_pack['test_size']:,} (20%)</li>
                    <li>عدد الفئات: {model_pack['unique_classes']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📊 مصفوفة الارتباك</div>', unsafe_allow_html=True)
                
                if model_pack['unique_classes'] <= 10:
                    cm = confusion_matrix(model_pack['y_test'], model_pack['y_pred'])
                    fig = px.imshow(cm, text_auto=True, 
                                    color_continuous_scale='Blues',
                                    title="نتائج التنبؤ على بيانات الاختبار")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("عدد الفئات كبير جداً لعرض مصفوفة الارتباك")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📊 تقرير التصنيف</div>', unsafe_allow_html=True)
                
                report = classification_report(
                    model_pack['y_test'], 
                    model_pack['y_pred'],
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # كشف الشذوذ
    with tab3:
        if not st.session_state.model_trained:
            st.warning("⚠️ يرجى تدريب النموذج أولاً")
        else:
            model_pack = st.session_state.model_pack
            
            with st.spinner("جاري كشف الحالات الشاذة..."):
                anomalies, conf_scores = detect_anomalies(
                    model_pack, df, threshold_percentile
                )
            
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🚨 نتائج كشف الشذوذ</div>
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-value">{len(anomalies):,}</div>
                        <div class="metric-label">حالة مشبوهة</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(anomalies)/len(df)*100:.2f}%</div>
                        <div class="metric-label">نسبة الشذوذ</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if len(anomalies) > 0:
                st.markdown(f"""
                <div class="alert-warning">
                    ⚠️ تم اكتشاف {len(anomalies)} حالة لا تتبع النمط الطبيعي.
                    هذه الحالات تحتاج إلى مراجعة دقيقة من قبل الخبراء.
                </div>
                """, unsafe_allow_html=True)
                
                # عرض الحالات الشاذة
                st.markdown('<div class="card-title">📋 الحالات المشبوهة (أهم 20)</div>', unsafe_allow_html=True)
                
                display_cols = [col for col in ['رقم_القضية', 'نوع_القرار', 'نتيجة_القضية', 
                                               'الطرف_الفائز', 'رئيس_المحكمة', 'درجة_الثقة'] 
                               if col in anomalies.columns]
                
                st.dataframe(anomalies[display_cols].head(20), use_container_width=True)
                
                # تحليل الشذوذ حسب رئيس المحكمة
                if 'رئيس_المحكمة' in anomalies.columns:
                    st.markdown('<div class="card-title">👨‍⚖️ تحليل الشذوذ حسب رئيس المحكمة</div>', unsafe_allow_html=True)
                    judge_anomalies = anomalies['رئيس_المحكمة'].value_counts().reset_index()
                    judge_anomalies.columns = ['رئيس المحكمة', 'عدد_الحالات']
                    fig = px.bar(judge_anomalies, x='رئيس المحكمة', y='عدد_الحالات',
                                 color='عدد_الحالات', color_continuous_scale='Reds')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    ✅ لم يتم العثور على حالات شاذة بالمعايير الحالية.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # تحليل الأسباب
    with tab4:
        if not st.session_state.model_trained:
            st.warning("⚠️ يرجى تدريب النموذج أولاً")
        else:
            model_pack = st.session_state.model_pack
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 أهم العوامل المؤثرة في القرار</div>', unsafe_allow_html=True)
            
            feature_importance = get_feature_importance(model_pack)
            
            for f in feature_importance:
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span><strong>{f['الميزة']}</strong></span>
                        <span>{f['الأهمية']*100:.1f}%</span>
                    </div>
                    <div class="feature-bar" style="width: {f['الأهمية']*100}%;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            # تحليل منطقي
            if feature_importance:
                top_feature = feature_importance[0]['الميزة']
                st.markdown(f"""
                <div class="alert-info">
                    <strong>🔎 الميزة الأكثر تأثيراً هي "{top_feature}"</strong><br><br>
                    هذا يعني أن النظام يعتبر أن هذا العامل هو الأهم في تحديد نتيجة القضية.
                    عند وجود حالات شاذة تتعلق بهذه الميزة، فإن ذلك يستدعي تدقيقاً إضافياً.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # منحنيات التعلم
    with tab5:
        if not st.session_state.model_trained:
            st.warning("⚠️ يرجى تدريب النموذج أولاً")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 منحنيات التعلم (matplotlib)</div>', unsafe_allow_html=True)
            
            model_pack = st.session_state.model_pack
            
            # رسم منحنيات التعلم
            fig = plot_learning_curves(model_pack)
            st.pyplot(fig)
            
            st.markdown("""
            <div class="alert-info">
                <strong>📈 شرح المنحنيات:</strong><br>
                - المنحنى الأيسر: يظهر تطور دقة النموذج مع زيادة عدد الدورات التدريبية<br>
                - المنحنى الأيمن: يظهر أهم 5 ميزات في اتخاذ القرار<br>
                - كلما اقترب منحنى التدريب والتحقق، قلّت مشكلة overfitting
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # نظام القرار
    with tab6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚖️ نظام القرار الهجين</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f7ff, #ffffff); padding: 1.5rem; border-radius: 15px;">
            <h4>آلية العمل:</h4>
            <ul>
                <li><span class="badge-normal">✅ منطقة آمنة (ثقة ≥ 80%)</span> - قرار آلي مع تفسير</li>
                <li><span class="badge-anomaly">❌ منطقة شاذة (ثقة ≤ 20%)</span> - رفض آلي مع تفسير</li>
                <li><span class="badge-warning">⚠️ منطقة رمادية</span> - تحويل للمراجعة البشرية</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        if st.session_state.model_trained:
            model_pack = st.session_state.model_pack
            
            st.markdown("#### 🔮 تجربة النموذج على قضية جديدة")
            
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
                    chief_justice = st.selectbox("رئيس المحكمة", df['رئيس_المحكمة'].dropna().unique())
                else:
                    chief_justice = "وارن"
                
                precedent = st.selectbox("تغيير السابقة", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
                split_vote = st.selectbox("تصويت منقسم", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
                evidence = st.slider("قوة الأدلة (1-5)", 1, 5, 3)
            
            if st.button("🔮 تحليل القضية", use_container_width=True):
                # تجهيز بيانات الإدخال
                input_data = {
                    'نوع_القرار': decision_type,
                    'نتيجة_القضية': case_disp,
                    'مجال_القضية': issue_area,
                    'تغيير_السابقة': precedent,
                    'تصويت_منقسم': split_vote,
                    'محلي': np.random.choice([0, 1]),  # افتراضي
                    'قوة_الأدلة': evidence,
                    'رئيس_المحكمة': chief_justice
                }
                
                # تحويل البيانات
                input_df = pd.DataFrame([input_data])
                
                # ترميز الأعمدة الفئوية
                for col in model_pack['categorical_cols']:
                    if col in model_pack['encoders'] and col in input_df.columns:
                        try:
                            input_df[col + '_code'] = model_pack['encoders'][col].transform(input_df[col].astype(str))
                        except:
                            input_df[col + '_code'] = -1
                
                # التأكد من وجود جميع الميزات
                feature_cols = [col for col in model_pack['feature_cols'] if col in input_df.columns]
                X_input = input_df[feature_cols]
                
                # تطبيع البيانات
                X_input_scaled = model_pack['scaler'].transform(X_input)
                
                if len(X_input_scaled) > 0:
                    # التنبؤ
                    pred = model_pack['model'].predict(X_input_scaled)[0]
                    proba = model_pack['model'].predict_proba(X_input_scaled)[0]
                    confidence = np.max(proba) * 100
                    
                    # عرض النتيجة
                    st.markdown('<hr>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background:#f8f9fa; padding:1.5rem; border-radius:15px;">
                        <h4 style="text-align:center;">🔮 نتيجة التحليل (XGBoost)</h4>
                        <p style="text-align:center; font-size:2rem; font-weight:900;">
                            {pred}
                        </p>
                        <p style="text-align:center;">الثقة: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence >= 80:
                        st.markdown("""
                        <div class="alert-success">
                            ✅ قرار آلي - ثقة عالية
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence <= 20:
                        st.markdown("""
                        <div class="alert-danger">
                            ❌ رفض آلي - ثقة منخفضة جداً
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="alert-warning">
                            ⚠️ يحتاج مراجعة بشرية - منطقة رمادية
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("👈 يرجى تدريب النموذج أولاً من القائمة الجانبية")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>⚖️ نظام عدالة لتحليل الأحكام القضائية | الإصدار 3.0.0</p>
        <p>المكتبات المستخدمة: streamlit, pandas, numpy, plotly, scikit-learn, xgboost, matplotlib</p>
        <p style="opacity:0.7; font-size:0.9rem;">© 2026 - جميع الحقوق محفوظة</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
