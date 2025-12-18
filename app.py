import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# Load trained pipeline model
# -------------------------------
# IMPORTANT: use the FINAL pipeline model, not the old one
model = joblib.load("models/final_model.pkl")


# -------------------------------
# App Title & Description
# -------------------------------
st.title("🏦 Smart Loan Approval System")
st.write("Predict loan approval using ML with clear, business-aligned decision logic.")

st.info(
    "Note: Credit History is assumed to be fetched from bank / credit bureau records."
)

st.warning(
    "This system uses a combination of ML risk scoring and banking rules "
    "to ensure safe and explainable loan decisions."
)

# -------------------------------
# User Inputs 
# -------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])

credit_history = st.selectbox(
    "Credit History (from credit bureau)",
    options=[1, 0],
    format_func=lambda x: "Good" if x == 1 else "Bad"
)

applicant_income = st.number_input(
    "Applicant Monthly Income",
    min_value=0,
    step=500
)


# -------------------------------
# System-derived / backend values
# -------------------------------
loan_amount = applicant_income * 0.25  # realistic derivation

default_input = {
    "Loan_ID": "LP001",
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": "No",
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": 0,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": 360,
    "Credit_History": credit_history,
    "Property_Area": "Semiurban"
}



# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Loan Status"):

    # Update user-controlled fields
    default_input["Credit_History"] = credit_history
    default_input["ApplicantIncome"] = applicant_income

    input_df = pd.DataFrame([default_input])

    # ML probability
    probability = model.predict_proba(input_df)[0][1]

    # -------------------------------
    # Hybrid decision logic
    # -------------------------------
    if credit_history == 0:
        decision = 0
        reason = "poor credit history"
    elif applicant_income < 3000:
        decision = 0
        reason = "insufficient income"
    elif probability < 0.65:
        decision = 0
        reason = "high risk based on model assessment"
    else:
        decision = 1
        reason = "strong credit profile and low risk score"

    # -------------------------------
    # UI Output
    # -------------------------------
    st.write(f"### Approval Probability: {probability:.2f}")

    if decision == 1:
        st.success("✅ Loan Approved")
        st.write(f"**Reason:** {reason}")
    else:
        st.error("❌ Loan Rejected")
        st.write(f"**Reason:** {reason}")
