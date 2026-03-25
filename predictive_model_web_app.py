import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
import pickle
from PIL import Image

def load_tif(path):
    try:
        img = Image.open(path)
        return img
    except Exception as e:
        st.error(f"无法加载图片 {path}: {e}")
        return None

logo1 = load_tif("fudan_logo_red.tif")
logo2 = load_tif("ISTBI_logo_red.tif")

col_l, col_m, col_r = st.columns([1, 6, 1])

with col_l:
   
    st.image(logo1, width=100)

with col_m:
 
    st.markdown(
        """
        <div style="text-align: center; line-height: 1.2;">
            <a href="https://istbi.fudan.edu.cn/ry/gdkyry_aszmpysx_.htm" target="_blank" 
               style="text-decoration: none; color: #1E1E1E; font-size: 3.5rem; font-weight: bold;">
               FuDan University
            </a>
            <span style="margin: 0 15px; color: #CCC; font-size: 1.5rem;">|</span>
            <a href="https://www.university-b.edu" target="_blank" 
               style="text-decoration: none; color: #1E1E1E; font-size: 3.5rem; font-weight: bold;">
               The Institute of Science and Technology for Brain-inspired Intelligence (ISTBI)
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col_r:
 
    st.image(logo2, width=100)


st.divider() 

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

@st.cache_data
def get_validation_data():
    np.random.seed(42)
    n_samples = 300 
    
    
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

    logit = (X_val['NIHSS_admission'] * 0.4 + X_val['Age'] * 0.08 - 10)
    prob = 1 / (1 + np.exp(-logit))
    y_val = (prob > np.random.rand(n_samples)).astype(int)
    
    return X_val, y_val


def user_input_features():
    st.sidebar.header("Patient Features")
    age = st.sidebar.number_input('Age', 0, 100, 65)
    treatment_val = st.sidebar.radio("allocated intervention", [0, 1], format_func=lambda x: "Intervention (1)" if x==1 else "Control (0)")
    nihss_adm = st.sidebar.number_input('NIHSS score at admission', 0, 42, 12)
    iv_throm_val = st.sidebar.radio("Intravenous Thrombolysis", [0, 1], format_func=lambda x: "YES (1)" if x==1 else "No (0)")
    nihss_rate = st.sidebar.slider('NIHSS_pct_change', 0, 100, 50)
    gcs_rate = st.sidebar.slider('GCS_pct_change', 0, 100, 80)
    lesion_val = st.sidebar.radio("Visible cerebral infarction", [0, 1], format_func=lambda x: "YES (1)" if x==1 else "NO (0)")

    data = {
        'Age': age, 'Treatment_arms': treatment_val, 'NIHSS_admission': nihss_adm,
        'intravenous thrombolysis': iv_throm_val, 'NIHSS_rate_day7': nihss_rate,
        'GCS_rate_day7': gcs_rate, 'Visible_cerebral_infarction_lesion': lesion_val
    }
    return pd.DataFrame([data])[feature_order]

input_df = user_input_features()

# st.title("AIS Patients Post-Thrombectomy Outcome Prediction & Explanation")
st.markdown(
    """
    <h1 style='text-align: center; font-size: 40px; color: #1E1E1E; font-family: sans-serif;'>
        AIS Patients Post-Thrombectomy Outcome Prediction & Explanation
    </h1>
    """, 
    unsafe_allow_html=True
)
if st.button("Predict Outcome"):
    all_probs = [m.predict_proba(input_df)[:, 1][0] for m in models]
    avg_prob = np.mean(all_probs)
    
    st.markdown(f"### Result: {'🔴 Unfavorable' if avg_prob > 0.5 else '🟢 Favorable'}")
    st.metric("Risk Probability", f"{avg_prob:.2%}")

st.divider()
if st.checkbox("Show Individual Explanation (SHAP)"):
    st.subheader("Individual Feature Contribution (SHAP)")
    if models:
        try:
          
            # plt.rcParams['mathtext.default'] = 'regular'
            plt.rcParams['mathtext.fontset'] = 'custom'
            plt.rcParams['mathtext.default'] = 'regular'
            
            
            explainer = shap.TreeExplainer(models[0])
            instance = input_df.astype(float)
            shap_values = explainer.shap_values(instance)
            
        
            fig = plt.figure(figsize=(20, 5))
            
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                instance.iloc[0, :],
                matplotlib=True,
                show=False,
                contribution_threshold=0.05 
            )
 
            current_fig = plt.gcf()
            current_fig.set_size_inches(20, 5) 
            plt.tight_layout(pad=2.0) 
            
            # 6. 显示
            st.pyplot(current_fig, clear_figure=True)
            
            st.write("🔴 **Red**: Increases risk of Unfavorable Outcome | 🔵 **Blue**: Decreases risk")
            
        except Exception as e:
            st.error(f"SHAP Plotting Error: {e}")
            
st.divider()
st.subheader("📊 Model Stability Analysis (Dynamic AUC)")


X_test_raw, y_test = get_validation_data()


target_feat = st.selectbox("Select Feature to Shift", feature_order, index=2) # 默认选 NIHSS_admission
shift_val = st.slider(f"Shift {target_feat} value", -30.0, 30.0, 0.0, step=1.0)

X_test_modified = X_test_raw.copy()
X_test_modified[target_feat] = X_test_modified[target_feat] + shift_val


with st.spinner('Updating AUC calculation...'):
    if models:
       
        # fold_probs = []
        # for m in models:
        #     p = m.predict_proba(X_test_modified[feature_order].astype(float))[:, 1]
        #     fold_probs.append(p)
        
        # y_probs = np.mean(fold_probs, axis=0)
        
  
        # current_auc = roc_auc_score(y_test, y_probs)

        # col1, col2 = st.columns([1, 2])
        # with col1:
        #     st.metric("Current AUC", f"{current_auc:.4f}", delta=f"{current_auc - 0.5:.4f}" if current_auc != 0.5 else None)
        #     st.write(f"**Explanation:**")
        #     st.write(f"You adjusted the **{target_feat}** for all patients by **{shift_val}** units.")
        #     st.info("If the AUC drops significantly, it means the model is sensitive to this feature's distribution shift.")
        individual_aucs = []
        fold_probs = [] 

        for i, m in enumerate(models):
   
            p = m.predict_proba(X_test_modified[feature_order].astype(float))[:, 1]
            fold_probs.append(p)

            auc_score = roc_auc_score(y_test, p)
            individual_aucs.append(auc_score)
        
        auc_mean = np.mean(individual_aucs)
        auc_std = np.std(individual_aucs)

        y_probs_avg = np.mean(fold_probs, axis=0)
        current_auc_avg = roc_auc_score(y_test, y_probs_avg)

        # 4. 界面显示
        col1, col2 = st.columns([1, 2])
        with col1:
  
            st.metric("Mean AUC", f"{auc_mean:.3f}")
            st.metric("AUC Std Dev", f"{auc_std:.3f}")
            
            
            # with st.expander("View individual fold AUCs"):
            #     for i, score in enumerate(individual_aucs):
            #         st.write(f"Fold {i+1}: {score:.4f}")
            
            st.write(f"**Explanation:**")
            st.write(f"You adjusted **{target_feat}** by **{shift_val}** units.")
            st.info("The Mean and Std Dev represent the stability across 5-fold cross-validation models.")

        with col2:
            
            fpr, tpr, _ = roc_curve(y_test, y_probs_avg )
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
            ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_mean:.3f})', color='darkorange', lw=2)
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Dynamic ROC Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)



