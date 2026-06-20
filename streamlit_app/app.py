 # ============================================================
# Customer Churn Prediction - Streamlit Web App (Cloud Ready)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# ── Load models using relative paths ─────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(base, '..')
    lr     = joblib.load(os.path.join(root, 'models', 'logistic_regression.pkl'))
    rf     = joblib.load(os.path.join(root, 'models', 'random_forest.pkl'))
    scaler = joblib.load(os.path.join(root, 'models', 'scaler.pkl'))
    return lr, rf, scaler

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(base, '..')
    return pd.read_csv(os.path.join(root, 'data', 'processed', 'churn_cleaned.csv'))

lr_model, rf_model, scaler = load_models()
train_df = load_data()

# ── Title ─────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction System")
st.markdown("**Predict which customers are at risk of leaving**")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
st.sidebar.markdown("✅ Logistic Regression: 80.70%")
st.sidebar.markdown("✅ Random Forest: 80.34%")
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Info**")
st.sidebar.markdown("📊 Dataset: IBM Telco Churn")
st.sidebar.markdown("👥 Customers: 7,043")
st.sidebar.markdown("📈 Churn Rate: 26.54%")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender           = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen   = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner          = st.selectbox("Partner", ["Yes", "No"])
        dependents       = st.selectbox("Dependents", ["Yes", "No"])
        tenure           = st.slider("Tenure (months)", 0, 72, 12)
        phone_service    = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines   = st.selectbox("Multiple Lines",
                            ["No", "Yes", "No phone service"])
        internet         = st.selectbox("Internet Service",
                            ["DSL", "Fiber optic", "No"])
        online_security  = st.selectbox("Online Security",
                            ["Yes", "No", "No internet service"])
        online_backup    = st.selectbox("Online Backup",
                            ["Yes", "No", "No internet service"])
        device_protect   = st.selectbox("Device Protection",
                            ["Yes", "No", "No internet service"])
        tech_support     = st.selectbox("Tech Support",
                            ["Yes", "No", "No internet service"])

    with col3:
        streaming_tv     = st.selectbox("Streaming TV",
                            ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies",
                            ["Yes", "No", "No internet service"])
        contract         = st.selectbox("Contract",
                            ["Month-to-month", "One year", "Two year"])
        paperless        = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment          = st.selectbox("Payment Method", [
                            "Electronic check",
                            "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)"
                          ])
        monthly_charges  = st.slider("Monthly Charges ($)", 0.0, 120.0, 65.0)
        total_charges    = st.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)

    if st.button("🔮 Predict Churn", type="primary"):

        input_data = pd.DataFrame([{
            'gender'           : gender,
            'SeniorCitizen'    : senior_citizen,
            'Partner'          : partner,
            'Dependents'       : dependents,
            'tenure'           : tenure,
            'PhoneService'     : phone_service,
            'MultipleLines'    : multiple_lines,
            'InternetService'  : internet,
            'OnlineSecurity'   : online_security,
            'OnlineBackup'     : online_backup,
            'DeviceProtection' : device_protect,
            'TechSupport'      : tech_support,
            'StreamingTV'      : streaming_tv,
            'StreamingMovies'  : streaming_movies,
            'Contract'         : contract,
            'PaperlessBilling' : paperless,
            'PaymentMethod'    : payment,
            'MonthlyCharges'   : monthly_charges,
            'TotalCharges'     : total_charges
        }])

        # Encode
        input_encoded  = pd.get_dummies(input_data)
        train_encoded  = pd.get_dummies(
            train_df.drop('Churn', axis=1), drop_first=False
        )
        input_encoded  = input_encoded.reindex(
            columns=train_encoded.columns, fill_value=0
        )
        input_scaled   = scaler.transform(input_encoded)

        model = lr_model if model_choice == "Logistic Regression" else rf_model
        pred  = model.predict(input_scaled)[0]
        prob  = model.predict_proba(input_scaled)[0][1]
        risk_score = int(prob * 100)

        st.divider()
        st.subheader("🎯 Prediction Result")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if pred == 1:
                st.error("⚠️ HIGH CHURN RISK")
            else:
                st.success("✅ LOW CHURN RISK")
        with col_b:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
        with col_c:
            st.metric("Risk Score", f"{risk_score}/100")

        # Risk bar
        fig, ax = plt.subplots(figsize=(6, 1.5))
        color = '#e74c3c' if risk_score > 50 else '#2ecc71'
        ax.barh(['Risk'], [risk_score], color=color, height=0.4)
        ax.barh(['Risk'], [100 - risk_score],
                left=risk_score, color='#ecf0f1', height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Churn Risk Score')
        ax.set_title(f'Risk Score: {risk_score}/100')
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        # Recommendations
        st.subheader("💡 Retention Recommendations")
        if pred == 1:
            if contract == "Month-to-month":
                st.warning("📋 Offer a discounted annual contract")
            if internet == "Fiber optic":
                st.warning("🌐 Investigate service quality issues")
            if payment == "Electronic check":
                st.warning("💳 Encourage switch to auto-pay")
            if tenure < 12:
                st.warning("🎁 Enroll in new customer loyalty program")
            if online_security == "No":
                st.warning("🔒 Offer free online security bundle")
        else:
            st.success("🌟 This customer is likely to stay!")
            st.info("💡 Upsell premium services to increase revenue")

# ════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📁 Upload Customer CSV File")
    st.info("Upload a CSV with multiple customers to predict churn for all")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"✅ Loaded {len(batch_df)} customers")
        st.dataframe(batch_df.head())

        if st.button("🔮 Predict All", type="primary"):
            if 'Churn' in batch_df.columns:
                batch_df = batch_df.drop('Churn', axis=1)

            batch_encoded = pd.get_dummies(batch_df)
            train_encoded = pd.get_dummies(
                train_df.drop('Churn', axis=1), drop_first=False
            )
            batch_encoded = batch_encoded.reindex(
                columns=train_encoded.columns, fill_value=0
            )
            batch_scaled  = scaler.transform(batch_encoded)

            model = lr_model if model_choice == "Logistic Regression" else rf_model
            preds = model.predict(batch_scaled)
            probs = model.predict_proba(batch_scaled)[:, 1]

            batch_df['Churn Prediction']  = ['Yes' if p == 1 else 'No' for p in preds]
            batch_df['Churn Probability'] = (probs * 100).round(1)
            batch_df['Risk Score']        = (probs * 100).astype(int)

            st.divider()
            st.subheader("📊 Batch Results")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Customers", len(batch_df))
            m2.metric("At Risk", len(batch_df[batch_df['Churn Prediction'] == 'Yes']))
            m3.metric("Churn Rate", f"{(preds.sum()/len(preds)*100):.1f}%")

            st.dataframe(batch_df)

            csv = batch_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Results",
                csv,
                "churn_predictions.csv",
                "text/csv"
            )
            