import streamlit as st
import joblib
import numpy as np

from src.explain_genai import generate_explanation


# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("models/random_forest_model.pkl")


# -------------------------------
# App Title & Description
# -------------------------------
st.title("🏦 Smart Loan Approval System")
st.write("Predict loan approval and get a human-readable explanation.")

st.info(
    "Note: Credit History is assumed to be fetched from bank/credit bureau records."
)

# -------------------------------
# User Inputs
# -------------------------------
credit_history = st.selectbox(
    "Credit History (as per bank records)",
    options=[1, 0],
    format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)"
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    step=500
)


# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Loan Status"):

    dummy_input = np.zeros(model.n_features_in_)
    dummy_input[0] = credit_history
    dummy_input[1] = applicant_income

    prediction = model.predict([dummy_input])[0]

    explanation = generate_explanation(
        {
            "Credit_History": credit_history,
            "ApplicantIncome": applicant_income
        },
        prediction
    )

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write("### Explanation")
    st.write(explanation)
