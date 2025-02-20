import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st

def train_model():
    # Load dataset and define features and target
    data = pd.read_csv('life_insurance_prediction.csv')
    features = ['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type']
    target = 'Prediction_Target'

    # Prepare data for training
    X = data[features].copy()
    y = data[target]

    # Label encode categorical columns
    label_encoders = {}
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Train the model using XGBoost
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X, y)

    # Train the premium model using XGBoost
    premium_model = XGBRegressor()
    premium_model.fit(X, data['Premium_Amount'])

    return model, premium_model, label_encoders

def predict_insurance():
    st.title("🏦 Life Insurance Eligibility & Premium Prediction")

    with st.container():
        age = st.slider("Select Age", 1, 100, 30)
        income = st.number_input("Enter Income", min_value=0.0, step=1000.0)

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.radio("Select Gender", ["Male", "Female"], horizontal=True)
        with col2:
            smoking = st.radio("Do you smoke?", ["Yes", "No"], horizontal=True)

    health_status = st.selectbox("Select Health Status", ["Excellent", "Good", "Average", "Poor"])

    if st.button("Predict Eligibility"):
        model, premium_model, label_encoders = train_model()

        input_data = pd.DataFrame([[age, gender, income, health_status, smoking, 'Term']],
                                   columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type'])

        for col, le in label_encoders.items():
            input_data[col] = le.transform(input_data[col].astype(str))

        if income > 100000 and health_status == 'Excellent':
            eligible_policies = ['Whole', 'Universal', 'Term']
        elif income > 50000 and health_status in ['Good', 'Average']:
            eligible_policies = ['Universal', 'Term']
        elif income > 5000:
            eligible_policies = ['Term']
        else:
            st.error("❌ Not Eligible for Insurance")
            st.write("Reason: Income is below the minimum threshold of 5000")
            return

        premium_estimates = {}
        for policy in eligible_policies:
            policy_encoded = label_encoders['Policy_Type'].transform([policy])[0]
            input_data['Policy_Type'] = policy_encoded
            premium_estimates[policy] = premium_model.predict(input_data)[0]

        st.success("🎉 Eligible for Insurance")
        st.write(f"Eligible Policies: {', '.join(eligible_policies)}")
        st.write("Estimated Premiums:")
        for policy, premium in premium_estimates.items():
            st.write(f"- {policy}: {premium:.2f}")

if __name__ == "__main__":
    predict_insurance()
