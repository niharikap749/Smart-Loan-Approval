from preprocess import preprocess_data
from explain_genai import generate_explanation

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib


# -------------------------------
# Load preprocessed data
# -------------------------------
X_train, X_test, y_train, y_test = preprocess_data("../data/loan_data.csv")


# -------------------------------
# Logistic Regression (Baseline)
# -------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("\nClassification Report:\n", classification_report(y_test, lr_pred))


# -------------------------------
# Random Forest (Improved Model)
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))


# -------------------------------
# Save the final model
# -------------------------------
joblib.dump(rf_model, "../models/random_forest_model.pkl")


# -------------------------------
# GenAI-style Explanation Demo
# -------------------------------
sample_input = {
    "Credit_History": 1,
    "ApplicantIncome": 5000
}

sample_prediction = rf_model.predict(X_test[:1])[0]

explanation = generate_explanation(sample_input, sample_prediction)

print("\nSample Prediction Explanation:")
print(explanation)
