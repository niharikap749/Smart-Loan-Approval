import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(path):
    df = pd.read_csv(path)
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    df.fillna({
        'Gender': df['Gender'].mode()[0],
        'Married': df['Married'].mode()[0],
        'Dependents': df['Dependents'].mode()[0],
        'LoanAmount': df['LoanAmount'].median(),
        'Credit_History': df['Credit_History'].mode()[0]
    }, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
