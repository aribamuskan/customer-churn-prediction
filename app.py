import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("customer_churn_model.pkl")
feature_columns = joblib.load("customer_churn_features.pkl")


# ==========================================
# TITLE
# ==========================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict whether "
    "the customer is likely to churn."
)

st.divider()


# ==========================================
# CUSTOMER INFORMATION
# ==========================================

st.subheader("👤 Customer Information")


gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)


tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12,
    step=1
)


monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=1000.0,
    value=85.50,
    step=0.50
)


contract_type = st.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


support_calls = st.number_input(
    "Support Calls",
    min_value=0,
    max_value=20,
    value=5,
    step=1
)


payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Credit card",
        "Bank transfer"
    ]
)


st.divider()


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔮 Predict Customer Churn", use_container_width=True):

    # Create customer dataframe
    customer_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "tenure": [tenure],
        "monthly_charges": [monthly_charges],
        "contract_type": [contract_type],
        "internet_service": [internet_service],
        "support_calls": [support_calls],
        "payment_method": [payment_method]
    })


    # ======================================
    # ONE-HOT ENCODING
    # ======================================

    categorical_columns = [
        "gender",
        "contract_type",
        "internet_service",
        "payment_method"
    ]

    customer_encoded = pd.get_dummies(
        customer_data,
        columns=categorical_columns,
        drop_first=True
    )


    # Make sure columns are exactly
    # the same as training data
    customer_encoded = customer_encoded.reindex(
        columns=feature_columns,
        fill_value=False
    )


    # ======================================
    # PREDICTION
    # ======================================

    prediction = model.predict(customer_encoded)[0]

    probability = model.predict_proba(
        customer_encoded
    )[0][1]


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.subheader("📌 Prediction Result")


    if prediction == 1:

        st.error("⚠️ Customer is likely to CHURN")

        st.write(
            f"Churn Probability: **{probability * 100:.2f}%**"
        )

    else:

        st.success("✅ Customer is likely to STAY")

        st.write(
            f"Churn Probability: **{probability * 100:.2f}%**"
        )


    # ======================================
    # PROBABILITY BAR
    # ======================================

    st.write("### Churn Probability")

    st.progress(float(probability))


    # ======================================
    # CUSTOMER DATA
    # ======================================

    st.write("### Customer Information")

    st.dataframe(
        customer_data,
        use_container_width=True
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Customer Churn Prediction • Machine Learning Project"
)