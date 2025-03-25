import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st

def train_model():
    data = pd.read_csv('life_insurance_prediction.csv')
    features = ['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type']
    target = 'Prediction_Target'

    X = data[features].copy()
    y = data[target]

    # Encode categorical variables
    label_encoders = {}
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X, y)

    premium_model = XGBRegressor()
    premium_model.fit(X, data['Premium_Amount'])

    return model, premium_model, label_encoders, accuracy_score(y, model.predict(X))

def predict_insurance():
    st.title("\U0001F3E6 Life Insurance Eligibility & Premium Prediction")

    age = st.slider("Select Age", 1, 100, 22)
    income = st.number_input("Enter Income", min_value=0.0, step=1000.0)
    gender = st.radio("Select Gender", ["Male", "Female"], horizontal=True)
    smoking = st.radio("Do you smoke?", ["Yes", "No"], horizontal=True)
    health_status = st.selectbox("Select Health Status", ["Excellent", "Good", "Average", "Poor"])

    if st.button("Predict Eligibility"):
        if age < 18 and smoking == "Yes":
            st.error("❌ Not Eligible for Insurance")
            st.write("Reason: Underage smoking detected.")
            return

        model, premium_model, label_encoders, accuracy = train_model()

        input_data = pd.DataFrame([[age, gender, income, health_status, smoking, 'Term']],
                                    columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type'])

        # Convert categorical values to numerical using trained label encoders
        for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

        # Define policy eligibility rules in a DataFrame
        eligibility_df = pd.DataFrame({
            "Policy_Type": ["Whole", "Universal", "Term"],
            "Min_Income": [100000, 50000, 5000],
            "Allowed_Health": [["Excellent"], ["Good", "Average"], ["Poor", "Good", "Average"]]
        })

        # Filter eligibility based on health status and income
        eligibility_df = eligibility_df.explode("Allowed_Health")
        eligible_policies = eligibility_df[
            (income >= eligibility_df["Min_Income"]) & (eligibility_df["Allowed_Health"] == health_status)
        ]["Policy_Type"].tolist()

        if health_status == "Excellent":
            income_ranges = [100000, 50000, 5000]  # Define income thresholds
            policy_mapping = {100000: ["Whole", "Universal", "Term"],
                              50000: ["Universal", "Term"],
                              5000: ["Term"]}
            eligible_policies = next((p for inc, p in policy_mapping.items() if income >= inc), [])

        if not eligible_policies:
            st.error("❌ Not Eligible for Insurance")
            st.write("Reason: Income below minimum or Health Status not eligible.")
            return

        # Predict premiums
        premium_estimates = {}
        for policy in eligible_policies:
            temp_input = pd.DataFrame([[age, gender, income, health_status, smoking, policy]],
                                        columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type'])
            for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']:
                temp_input[col] = label_encoders[col].transform(temp_input[col].astype(str))
            premium_estimates[policy] = premium_model.predict(temp_input)[0]

        st.success("\U0001F389 Eligible for Insurance")
        st.write(f"Eligible Policies: {', '.join(eligible_policies)}")
        st.write("Estimated Premiums:")
        for policy, premium in premium_estimates.items():
            st.write(f"- {policy}: {premium:.2f}")

        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Define company links for each type of insurance
        company_links = {
            "Whole": ["LIC Jeevan Umang - [LIC](https://www.licindia.in)",
                      "HDFC Life Sanchay Whole Life - [HDFC Life](https://www.hdfclife.com)",
                      "Max Life Whole Life Super - [Max Life](https://www.maxlifeinsurance.com)"],
            "Universal": ["ICICI Pru Lifetime Classic - [ICICI Prudential](https://www.iciciprulife.com)",
                            "SBI Life Smart Privilege - [SBI Life](https://www.sbilife.co.in)",
                            "Tata AIA Smart Sampoorna Raksha - [Tata AIA](https://www.tataaia.com)"],
            "Term": ["LIC Tech Term - [LIC](https://www.licindia.in)",
                     "HDFC Life Click 2 Protect - [HDFC Life](https://www.hdfclife.com)",
                     "ICICI Pru iProtect Smart - [ICICI Prudential](https://www.iciciprulife.com)"]
        }

        # Display company links
        st.write("🔗 **Recommended Insurance Providers:**")
        for policy in eligible_policies:
            st.write(f"**{policy} Insurance:**")
            for link in company_links[policy]:
                st.write(f"- {link}")

if __name__ == "__main__":
    predict_insurance()