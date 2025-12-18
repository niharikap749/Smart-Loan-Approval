import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(path):
    """
    Loads the dataset, cleans it, encodes features,
    splits into train/test, and scales numeric values.
    """

    # ---- FIX 1: Robust file path handling ----
    base_dir = os.path.dirname(__file__)        # src/
    full_path = os.path.join(base_dir, path)    # src/../data/loan_data.csv

    df = pd.read_csv(full_path)

    # ---- Encode target variable ----
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # ---- Handle missing values ----
    df.fillna({
        'Gender': df['Gender'].mode()[0],
        'Married': df['Married'].mode()[0],
        'Dependents': df['Dependents'].mode()[0],
        'LoanAmount': df['LoanAmount'].median(),
        'Credit_History': df['Credit_History'].mode()[0]
    }, inplace=True)

    # ---- Convert categorical features to numeric ----
    df = pd.get_dummies(df, drop_first=True)
    # ---- Final safety check: fill any remaining NaNs ----
    df = df.fillna(0)


    # ---- Separate features and target ----
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # ---- Train-test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Feature scaling ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
