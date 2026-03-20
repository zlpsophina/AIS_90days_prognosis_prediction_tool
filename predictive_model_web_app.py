import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
import pickle
st.set_page_config(page_title="AIS 90-day Prognosis Tool", layout="centered")

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
        else:
            st.warning(f"Missing model file: {name}")
    return models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models')
MODEL_FILES = [f'Xgb_model_fold_{i}.pkl' for i in range(5)]

try:
    models = load_models(MODEL_PATH, MODEL_FILES)
    if not models:
        st.error("No models were loaded. Please check the 'models' folder.")
except Exception as e:
    st.error(f"Error loading models: {e}")

# --- 3. 获取验证数据 (修正列名问题) ---
@st.cache_data
def get_validation_data():
    # 实际部署时请替换为: return pd.read_csv('your_test_data.csv')
    np.random.seed(42)
    # 关键修改：确保列名与 feature_order 一致
    X_val = pd.DataFrame(np.random.rand(100, 7), columns=feature_order)
    # 模拟真实的 Age (0-100) 和 NIHSS (0-42) 范围，否则 AUC 计算没意义
    X_val['Age'] = X_val['Age'] * 100
    X_val['NIHSS_admission'] = X_val['NIHSS_admission'] * 42
    
    # 模拟一个真实的标签 (y)
    y_val = (X_val['NIHSS_admission'] * 0.1 + np.random.normal(0, 1, 100) > 2).astype(int)
    return X_val, y_val

# --- 4. 侧边栏输入 ---
def user_input_features():
    st.sidebar.header("Patient Features")
    st.sidebar.markdown("---")
    age = st.sidebar.slider('Age', 0, 100, 65)
    treatment_map = {"Control (0)": 0, "Treatment (1)": 1}
    treatment_arms = st.sidebar.radio("Treatment_arms", list(treatment_map.keys()))
    nihss_adm = st.sidebar.number_input('NIHSS_admission', 0, 42, 12)
    throm_map = {"No (0)": 0, "YES (1)": 1}
    iv_throm = st.sidebar.radio("Intravenous Thrombolysis", list(throm_map.keys()))
    nihss_rate = st.sidebar.slider('NIHSS_rate_day7', 0, 100, 50)
    gcs_rate = st.sidebar.slider('GCS_rate_day7', 0, 100, 80)
    lesion_map = {"NO (0)": 0, "YES (1)": 1}
    v_lesion = st.sidebar.radio("Visible Cerebral Infarction Lesion", list(lesion_map.keys()))

    data = {
        'Age': age,
        'Treatment_arms': treatment_map[treatment_arms],
        'NIHSS_admission': nihss_adm,
        'intravenous thrombolysis': throm_map[iv_throm],
        'NIHSS_rate_day7': nihss_rate,
        'GCS_rate_day7': gcs_rate,
        'Visible_cerebral_infarction_lesion': lesion_map[v_lesion]
    }
    return pd.DataFrame([data])[feature_order]

input_df = user_input_features()

# --- 5. 主界面预测与 SHAP ---
st.title("Stroke Outcome Prediction & Explanation")

if st.button("Predict Outcome"):
    # 计算 5 个模型的平均概率
    all_probs = [m.predict_proba(input_df)[:, 1][0] for m in models]
    avg_prob = np.mean(all_probs)
    
    st.subheader(f"Prediction: {'Unfavorable' if avg_prob > 0.5 else 'Favorable'}")
    st.metric("Risk Probability", f"{avg_prob:.2%}")

st.divider()

if st.checkbox("Show SHAP explanation"):
    # 移除嵌套的 button 以防状态丢失
    with st.spinner("Calculating SHAP values..."):
        explainer = shap.TreeExplainer(models[0])
        shap_values = explainer.shap_values(input_df)
        plt.close() 
        fig = plt.figure(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0, :],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig, clear_figure=True)
        st.write("🔴 Red = Predict Unfavorable | 🔵 Blue = Predict Favorable")

# --- 6. 动态 AUC 变化 (修复 KeyError) ---
st.divider()
st.subheader("Dynamic AUC Analysis")

X_test, y_test = get_validation_data()
shift_val = st.slider("Shift Age value in test set to observe AUC impact", -20.0, 20.0, 0.0)

X_test_modified = X_test.copy()
# 关键修复：这里的 feature_order[0] 现在在 X_test 中存在了
X_test_modified[feature_order[0]] += shift_val 

# 计算 5 折平均 AUC
y_probs = np.mean([m.predict_proba(X_test_modified)[:, 1] for m in models], axis=0)
current_auc = roc_auc_score(y_test, y_probs)

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Current AUC", f"{current_auc:.3f}")
    st.write(f"When **{feature_order[0]}** is shifted by {shift_val}, the model's discriminative power changes.")

with col2:
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label=f'AUC = {current_auc:.2f}', color='darkorange')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    st.pyplot(fig_roc)



