import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_model():
    # Load dataset and define features and target
    data = pd.read_csv('life_insurance_prediction.csv')
    features = ['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type']
    target = 'Prediction_Target'

    # Prepare data for training
    X = data[features].copy()
    y = data[target]

    # Label encode categorical columns
    label_encoders = {}
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Train the model using XGBoost
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X, y)

    # Calculate model accuracy
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Train the premium model using XGBoost
    premium_model = XGBRegressor()
    premium_model.fit(X, data['Premium_Amount'])

    return model, premium_model, label_encoders

def get_user_input():
    # Collect user inputs (excluding premium_amount)
    age = int(input("Enter age: "))
    gender = input("Enter gender (Male/Female): ").capitalize()
    income = float(input("Enter income: "))
    health_status = input("Enter health status (Excellent/Good/Average/Poor): ").capitalize()
    smoking = input("Do you smoke? (Yes/No): ").capitalize()
    family_history = input("Do you have a family history of illness? (Yes/No): ").capitalize()

    return age, gender, income, health_status, smoking, family_history

def predict_insurance():
    # Get user inputs
    age, gender, income, health_status, smoking, family_history = get_user_input()

    # Train the model every time (since we're not using pickle)
    model, premium_model, label_encoders = train_model()

    # Prepare input data for prediction (without premium_amount)
    input_data = pd.DataFrame([[age, gender, income, health_status, smoking, family_history, 'Term']],
                               columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type'])

    # Encode categorical inputs
    for col, le in label_encoders.items():
        input_data[col] = le.transform(input_data[col].astype(str))

    # Make prediction for eligibility
    prediction = model.predict(input_data)

    # Determine eligible policies based on conditions
    if income > 100000 and health_status == 'Excellent':
        eligible_policies = ['Whole', 'Universal', 'Term']
    elif income > 50000 and health_status in ['Good', 'Average']:
        eligible_policies = ['Universal', 'Term']
    else:
        eligible_policies = ['Term']

    # Estimate premiums for eligible policies
    premium_estimates = {}
    for policy in eligible_policies:
        policy_encoded = label_encoders['Policy_Type'].transform([policy])[0]
        input_data['Policy_Type'] = policy_encoded
        premium_estimates[policy] = premium_model.predict(input_data)[0]

    # Return the result
    result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
    return result, eligible_policies, premium_estimates

# Example usage
if __name__ == "__main__":
    result, eligible_policies, premium_estimates = predict_insurance()
    print(f"Eligibility: {result}")
    print(f"Eligible Policies: {eligible_policies}")
    print(f"Premium Estimates: {premium_estimates}")