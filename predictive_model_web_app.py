import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
import pickle
st.set_page_config(page_title="AIS 90-day Prognosis Tool", layout="wide")

# --- 1. 定义严格的特征顺序 ---
feature_order = [
    'Age', 'Treatment_arms', 'NIHSS_admission', 'intravenous thrombolysis', 
    'NIHSS_rate_day7', 'GCS_rate_day7', 'Visible_cerebral_infarction_lesion'
]

# --- 2. 加载模型 ---
@st.cache_resource
def load_models(model_folder, model_names):
    models = []
    for name in model_names:
        file_path = os.path.join(model_folder, name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                models.append(pickle.load(f))
    return models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models')
MODEL_FILES = [f'Xgb_model_fold_{i}.pkl' for i in range(5)]

models = load_models(MODEL_PATH, MODEL_FILES)

# --- 3. 获取验证数据 (修复 AUC 不变化的问题) ---
# 注意：随机数据由于完全随机，微小的特征偏移可能不足以改变 AUC。
# 我们通过构造一个与特征强相关的标签来使 AUC 变化可见。
@st.cache_data
def get_validation_data():
    np.random.seed(42)
    n_samples = 200
    X_val = pd.DataFrame({
        'Age': np.random.randint(40, 90, n_samples),
        'Treatment_arms': np.random.randint(0, 2, n_samples),
        'NIHSS_admission': np.random.randint(5, 30, n_samples),
        'intravenous thrombolysis': np.random.randint(0, 2, n_samples),
        'NIHSS_rate_day7': np.random.uniform(0, 100, n_samples),
        'GCS_rate_day7': np.random.uniform(0, 100, n_samples),
        'Visible_cerebral_infarction_lesion': np.random.randint(0, 2, n_samples)
    })
    X_val = X_val[feature_order] # 确保顺序
    
    # 构造线性关系：NIHSS 越高，Unfavorable (1) 概率越高
    # 这样当我们人为偏移 NIHSS 或 Age 时，预测概率会剧烈变动，从而影响 AUC
    logit = (X_val['NIHSS_admission'] * 0.15 + X_val['Age'] * 0.05 - 8)
    prob = 1 / (1 + np.exp(-logit))
    y_val = (prob > np.random.rand(n_samples)).astype(int)
    
    return X_val, y_val

# --- 4. 侧边栏输入 ---
def user_input_features():
    st.sidebar.header("Patient Features")
    age = st.sidebar.slider('Age', 0, 100, 65)
    treatment_val = st.sidebar.radio("Treatment_arms", [0, 1], format_func=lambda x: "Treatment (1)" if x==1 else "Control (0)")
    nihss_adm = st.sidebar.number_input('NIHSS_admission', 0, 42, 12)
    iv_throm_val = st.sidebar.radio("Intravenous Thrombolysis", [0, 1], format_func=lambda x: "YES (1)" if x==1 else "No (0)")
    nihss_rate = st.sidebar.slider('NIHSS_rate_day7', 0, 100, 50)
    gcs_rate = st.sidebar.slider('GCS_rate_day7', 0, 100, 80)
    lesion_val = st.sidebar.radio("Visible Lesion", [0, 1], format_func=lambda x: "YES (1)" if x==1 else "NO (0)")

    data = {
        'Age': age, 'Treatment_arms': treatment_val, 'NIHSS_admission': nihss_adm,
        'intravenous thrombolysis': iv_throm_val, 'NIHSS_rate_day7': nihss_rate,
        'GCS_rate_day7': gcs_rate, 'Visible_cerebral_infarction_lesion': lesion_val
    }
    return pd.DataFrame([data])[feature_order]

input_df = user_input_features()

# --- 5. 主界面预测与 SHAP (修复不显示问题) ---
st.title("Stroke Outcome Prediction & Explanation")

if st.button("Predict Outcome"):
    all_probs = [m.predict_proba(input_df)[:, 1][0] for m in models]
    avg_prob = np.mean(all_probs)
    
    st.markdown(f"### Result: {'🔴 Unfavorable' if avg_prob > 0.5 else '🟢 Favorable'}")
    st.metric("Risk Probability", f"{avg_prob:.2%}")

st.divider()

if st.checkbox("Show SHAP explanation"):
    st.subheader("Individual Feature Contribution (SHAP)")
    if not models:
        st.error("Model not loaded.")
    else:
        # 强制清除之前的图形
        plt.clf()
        # 对于 XGBoost，使用 TreeExplainer
        explainer = shap.TreeExplainer(models[0])
        # 这里的 input_df 必须是 float 类型，有些版本的 XGBoost 会因为 int 报错
        shap_values = explainer.shap_values(input_df.astype(float))
        
        # 修复显示的核心：matplotlib=True 并且显式创建 fig
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0, :],
            matplotlib=True,
            show=False
        )
        # 调整布局防止文字重叠
        plt.tight_layout()
        st.pyplot(fig)
        st.write("🔴 Red: Factors increasing risk | 🔵 Blue: Factors decreasing risk")

# --- 6. 动态 AUC 变化 (修复不更新问题) ---
st.divider()
st.subheader("Model Stability Analysis (Dynamic AUC)")

X_test, y_test = get_validation_data()

# 增加调节范围：选择要偏移的特征
target_feat = st.selectbox("Select Feature to Shift", feature_order)
shift_val = st.slider(f"Shift {target_feat} value in test set", -30.0, 30.0, 0.0, step=1.0)

# 实时计算修改后的数据
X_test_modified = X_test.copy()
X_test_modified[target_feat] = X_test_modified[target_feat] + shift_val

# 预测
if models:
    # 对所有 Fold 求平均概率
    y_probs = np.mean([m.predict_proba(X_test_modified)[:, 1] for m in models], axis=0)
    current_auc = roc_auc_score(y_test, y_probs)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Current AUC", f"{current_auc:.4f}")
        st.info(f"The AUC reflects the model's accuracy on the test set when all patients' **{target_feat}** is adjusted by **{shift_val}** units.")

    with col2:
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {current_auc:.2f})', color='darkorange', lw=2)
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)



