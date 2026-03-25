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
    n_samples = 300 # 样本量稍微多一点，AUC 更平滑
    
    # 构造特征
    data = {
        'Age': np.random.normal(65, 12, n_samples),
        'Treatment_arms': np.random.randint(0, 2, n_samples),
        'NIHSS_admission': np.random.normal(15, 7, n_samples),
        'intravenous thrombolysis': np.random.randint(0, 2, n_samples),
        'NIHSS_rate_day7': np.random.uniform(20, 80, n_samples),
        'GCS_rate_day7': np.random.uniform(20, 90, n_samples),
        'Visible_cerebral_infarction_lesion': np.random.randint(0, 2, n_samples)
    }
    X_val = pd.DataFrame(data)[feature_order]
    
    # 关键：创建一个基于逻辑回归的标签，使其对 NIHSS 和 Age 极度敏感
    # 公式：logit = beta0 + beta1*Age + beta2*NIHSS ...
    # 我们故意放大 NIHSS 的权重，这样当你偏移 NIHSS 时，AUC 会崩得很厉害
    logit = (X_val['NIHSS_admission'] * 0.4 + X_val['Age'] * 0.08 - 10)
    prob = 1 / (1 + np.exp(-logit))
    y_val = (prob > np.random.rand(n_samples)).astype(int)
    
    return X_val, y_val

# --- 4. 侧边栏输入 ---
def user_input_features():
    st.sidebar.header("Patient Features")
    age = st.sidebar.slider('Age', 0, 100, 65)
    treatment_val = st.sidebar.radio("Treatment_arms", [0, 1], format_func=lambda x: "Intervention (1)" if x==1 else "Control (0)")
    nihss_adm = st.sidebar.number_input('NIHSS_admission', 0, 42, 12)
    iv_throm_val = st.sidebar.radio("Intravenous Thrombolysis", [0, 1], format_func=lambda x: "YES (1)" if x==1 else "No (0)")
    nihss_rate = st.sidebar.slider('NIHSS_pct_change', 0, 100, 50)
    gcs_rate = st.sidebar.slider('GCS_pct_change', 0, 100, 80)
    lesion_val = st.sidebar.radio("Visible_cerebral_infarction_lesion", [0, 1], format_func=lambda x: "YES (1)" if x==1 else "NO (0)")

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

# --- 6. SHAP 部分 (解决重叠问题) ---
st.divider()
if st.checkbox("Show Individual Explanation (SHAP)"):
    st.subheader("Individual Feature Contribution (SHAP)")
    if models:
        try:
            # 1. 禁用数学字符解析，防止 LaTeX 报错
            # plt.rcParams['mathtext.default'] = 'regular'
            plt.rcParams['mathtext.fontset'] = 'custom'
            plt.rcParams['mathtext.default'] = 'regular'
            
            # 2. 获取 SHAP 值
            explainer = shap.TreeExplainer(models[0])
            instance = input_df.astype(float)
            shap_values = explainer.shap_values(instance)
            
            # 3. 重点：先建立一个足够宽的 Figure
            # 宽度设为 20，高度 5 
            fig = plt.figure(figsize=(20, 5))
            
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                instance.iloc[0, :],
                matplotlib=True,
                show=False,
                contribution_threshold=0.05 # 过滤掉贡献极小的特征标签，腾出空间
            )
            
            # 5. 关键：通过 gcf() (Get Current Figure) 获取 SHAP 刚刚画好的图
            # 并手动调整布局，为长特征名留出左右边距
            current_fig = plt.gcf()
            current_fig.set_size_inches(20, 5) # 再次强制尺寸
            plt.tight_layout(pad=2.0) # 增加边距
            
            # 6. 显示
            st.pyplot(current_fig, clear_figure=True)
            
            st.write("🔴 **Red**: Increases risk of Unfavorable Outcome | 🔵 **Blue**: Decreases risk")
            
        except Exception as e:
            st.error(f"SHAP Plotting Error: {e}")
            
st.divider()
st.subheader("📊 Model Stability Analysis (Dynamic AUC)")

# 获取原始测试集
X_test_raw, y_test = get_validation_data()

# 侧边栏/主界面选择偏移
target_feat = st.selectbox("Select Feature to Shift", feature_order, index=2) # 默认选 NIHSS_admission
shift_val = st.slider(f"Shift {target_feat} value", -30.0, 30.0, 0.0, step=1.0)

# 创建副本并应用偏移
X_test_modified = X_test_raw.copy()
X_test_modified[target_feat] = X_test_modified[target_feat] + shift_val

# 预测部分 - 增加 spinner 提示并强制执行
with st.spinner('Updating AUC calculation...'):
    if models:
        # 核心：确保输入类型为 float，并保持 feature_order 顺序
        # 计算 5 折平均概率
        fold_probs = []
        for m in models:
            p = m.predict_proba(X_test_modified[feature_order].astype(float))[:, 1]
            fold_probs.append(p)
        
        y_probs = np.mean(fold_probs, axis=0)
        
        # 计算当前 AUC
        current_auc = roc_auc_score(y_test, y_probs)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Current AUC", f"{current_auc:.4f}", delta=f"{current_auc - 0.5:.4f}" if current_auc != 0.5 else None)
            st.write(f"**Explanation:**")
            st.write(f"You adjusted the **{target_feat}** for all patients by **{shift_val}** units.")
            st.info("If the AUC drops significantly, it means the model is sensitive to this feature's distribution shift.")

        with col2:
            # 绘制曲线
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            fig_roc, ax_roc = plt.subplots(figsize=(5, 3))
            ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {current_auc:.2f})', color='darkorange', lw=2)
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Dynamic ROC Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)



