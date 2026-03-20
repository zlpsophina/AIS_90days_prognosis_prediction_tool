import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
import pickle
# --- 1. 模拟验证集数据 (用于计算 AUC) ---
# 在实际部署中，建议加载你训练时预留的独立测试集 (test_df)
# --- 页面配置 ---
st.set_page_config(page_title="The 90 day prognosis outcome prediction tool for AIS patients who underwent successful reperfusion with endovascular thrombectomy", layout="centered")

# --- 加载模型函数 ---
@st.cache_resource # 使用缓存避免重复加载
def load_models(model_folder, model_names):
    models = []
    for name in model_names:
        with open(os.path.join(model_folder, name), 'rb') as f:
            models.append(pickle.load(f))
    return models

# 假设你的模型文件放在 'models' 文件夹下
# model_path='/home1/HWGroup/daiyx/zhenglp/model/EN_LCA/AIS_biomaker/result_v1.24/Step4_train_best_param/XGBoost/Model3_SFS_final_no_Vmsseff/save_fold_models'
# model_files = [f'Xgb_model_fold_{i}.pkl' for i in range(5)]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 假设模型放在仓库的 models 文件夹下
MODEL_PATH = os.path.join(BASE_DIR, 'models')
MODEL_FILES = [f'Xgb_model_fold_{i}.pkl' for i in range(5)]
# 这里的 path 需要根据你实际存放 pkl 的位置修改
try:
    models = load_models(MODEL_PATH, MODEL_FILES)
except:
    st.error("未找到模型文件，请检查路径。")

@st.cache_data
def get_validation_data():
    # 这里仅为演示，实际应 load_csv('test_set.csv')
    np.random.seed(42)
    X_val = pd.DataFrame(np.random.rand(100, 7), columns=[f'Feature_{i}' for i in range(7)])
    y_val = np.random.randint(0, 2, 100)
    return X_val, y_val


feature_order = ['Age', 'Treatment_arms', 'NIHSS_admission', 'intravenous thrombolysis', 'NIHSS_rate_day7','GCS_rate_day7', 'Visible_cerebral_infarction_lesion']



def user_input_features():
    st.sidebar.header("Adjustabel biomaker features for AIS patients")
    st.sidebar.markdown("---")

    # 1. Age (0-100 连续值)
    age = st.sidebar.slider('Age ', 0, 100, 65)

    # 2. Treatment_arms (0 或 1)
    # 使用 map 将显示文字转为数值
    treatment_map = {"Control (0)": 0, "Treatment (1)": 1}
    treatment_arms = st.sidebar.radio("Treatment_arms", list(treatment_map.keys()))
    treatment_val = treatment_map[treatment_arms]

    # 3. NIHSS_admission (0-42 连续值)
    nihss_adm = st.sidebar.number_input('NIHSS_admission ', 0, 42, 12)

    # 4. intravenous thrombolysis (0 或 1)
    throm_map = {"No (0)": 0, "YES (1)": 1}
    iv_throm = st.sidebar.radio("Intravenous Thrombolysis", list(throm_map.keys()))
    iv_throm_val = throm_map[iv_throm]

    # 5. NIHSS_rate_day7 (0-100 百分比)
    nihss_rate = st.sidebar.slider('NIHSS_rate_day7 ', 0, 100, 50)

    # 6. GCS_rate_day7 (连续值)
    # 注：若GCS改善率也是百分比，范围可设为0-100；若不是请调整范围
    gcs_rate = st.sidebar.slider('GCS_rate_day7', 0, 100, 80)

    # 7. Visible_cerebral_infarction_lesion (0 或 1)
    lesion_map = {"NO (0)": 0, "YES (1)": 1}
    lesion = st.sidebar.radio("Visible Cerebral Infarction Lesion", list(lesion_map.keys()))
    lesion_val = lesion_map[lesion]

    # 构造 DataFrame，确保列名与模型训练时完全一致
    data = {
        'Age': age,
        'Treatment_arms': treatment_val,
        'NIHSS_admission': nihss_adm,
        'intravenous thrombolysis': iv_throm_val,
        'NIHSS_rate_day7': nihss_rate,
        'GCS_rate_day7': gcs_rate,
        'Visible_cerebral_infarction_lesion': lesion_val
    }
    # 强制按 feature_order 排序
    return pd.DataFrame([data])[feature_order]




# 调用函数获取用户输入
input_df = user_input_features()

# --- 3. 模型推理与 SHAP 展示 ---
st.title("Stroke Outcome explainable prediction tool")

# 加载模型 (假设已加载到 models 列表)
# model = models[0]

if st.checkbox("Show SHAP explanation"):
    if st.button("Calculate SHAP"):
        with st.spinner("Calculating..."):
            # 使用第一折模型进行解释
            explainer = shap.TreeExplainer(models[0])
            # 注意：input_df 必须是 DataFrame
            shap_values = explainer.shap_values(input_df)

            # 这种写法在远程服务器部署时最稳定
            plt.close() # 清除之前的绘图
            fig = plt.figure(figsize=(10, 3))
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                input_df.iloc[0, :],
                matplotlib=True,
                show=False,
                contribution_threshold=0.05
            )
            st.pyplot(plt.gcf(), clear_figure=True)
            st.write(" Red = Predict Unfavorable | Blue = Predict Favorable")

    # explainer = shap.TreeExplainer(models[0])
    # shap_values = explainer.shap_values(input_df)

    # 绘制 Force Plot
    # Matplotlib 模式在 Streamlit 中更易兼容
    # fig, ax = plt.subplots(figsize=(10, 3))
    # shap.force_plot(
    #     explainer.expected_value,
    #     shap_values[0],
    #     input_df.iloc[0, :],
    #     matplotlib=True,
    #     show=False,
    #     contribution_threshold=0.05
    # )
    # st.pyplot(plt.gcf())
    # # plt.clf()
    #     st.write("notes：red presents the factor related to Unfavorable, blud presents the factor related to Favorable 。")

# --- 4. 动态 AUC 变化观察 ---
st.divider()
st.subheader("AUC")

X_test, y_test = get_validation_data()

# 逻辑：当用户在侧边栏调节特征时，我们观察“如果测试集中该特征发生偏移，AUC会如何变化”
# 或者更常见的做法：根据用户选择的特征范围筛选测试集，查看子组 AUC
shift_val = st.slider("Simulating the Impact of Feature Shift on AUC", -20, 20, 0.0)

# 简单的动态模拟：对测试集增加偏置
X_test_modified = X_test.copy()
X_test_modified[feature_order[0]] += shift_val  # 以第一个特征为例

# 计算 5 折平均预测
y_probs = np.mean([m.predict_proba(X_test_modified)[:, 1] for m in models], axis=0)
current_auc = roc_auc_score(y_test, y_probs)

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Current AUC", f"{current_auc:.3f}")
    st.write(f"The performance of model prediction,when {feature_order[0]} increasing {shift_val}。")

with col2:
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f'ROC curve (area = {current_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

st.subheader("亚组人群模型性能 (Dynamic AUC)")



