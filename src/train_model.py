from preprocess import get_pipeline_and_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib


preprocessor, X_train, X_test, y_train, y_test = get_pipeline_and_data(
    "../data/loan_data.csv"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "../models/final_model.pkl")
print("Model saved as models/final_model.pkl")
