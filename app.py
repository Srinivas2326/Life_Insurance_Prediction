import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st
import sqlite3

# Initialize DB
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        name TEXT PRIMARY KEY,
        age INTEGER,
        gender TEXT,
        income REAL,
        health_status TEXT,
        smoking TEXT
    )''')
    conn.commit()
    conn.close()

# Save user data
def save_user_data(name, age, gender, income, health_status, smoking):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("REPLACE INTO users VALUES (?, ?, ?, ?, ?, ?)",
              (name, age, gender, income, health_status, smoking))
    conn.commit()
    conn.close()

# Load user data
def load_user_data(name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE name = ?", (name,))
    result = c.fetchone()
    conn.close()
    return result

# Train models
def train_model():
    data = pd.read_csv('life_insurance_prediction.csv')
    features = ['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type']
    target = 'Prediction_Target'

    X = data[features].copy()
    y = data[target]

    label_encoders = {}
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    model = XGBClassifier(eval_metric='logloss')
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    premium_model = XGBRegressor()
    premium_model.fit(X, data['Premium_Amount'])

    return model, premium_model, label_encoders, accuracy

# Main Streamlit app
def predict_insurance():
    init_db()
    st.title("\U0001F3E6 Life Insurance Eligibility & Premium Prediction")

    name = st.text_input("Enter your full name (First Name and Last Name):")
    if name and len(name.strip().split()) < 2:
        st.warning("⚠️ Please enter your full name with surname (e.g., Gandhi Mahatma).")
        return

    user_data = load_user_data(name) if name else None

    with st.container():
        age = st.slider("Select Age", 1, 100, user_data[1] if user_data else 22)
        income = st.number_input("Enter Income", min_value=0.0, step=1000.0, value=user_data[3] if user_data else 50000.0)

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.radio("Select Gender", ["Male", "Female"], horizontal=True, index=["Male", "Female"].index(user_data[2]) if user_data else 0)
        with col2:
            smoking = st.radio("Do you smoke?", ["Yes", "No"], horizontal=True, index=["Yes", "No"].index(user_data[5]) if user_data else 1)

    health_status = st.selectbox("Select Health Status", ["Excellent", "Good", "Average", "Poor"], index=["Excellent", "Good", "Average", "Poor"].index(user_data[4]) if user_data else 0)

    if st.button("Predict Eligibility"):
        if age < 18 and smoking == "Yes":
            st.error("❌ Not Eligible for Insurance")
            st.write("Reason: Underage smoking detected.")
            st.info("💡 Suggestion: Maintain a healthy lifestyle and avoid smoking to improve eligibility in the future.")
            return

        save_user_data(name or "Guest", age, gender, income, health_status, smoking)
        model, premium_model, label_encoders, accuracy = train_model()

        input_data = pd.DataFrame([[age, gender, income, health_status, smoking, 'Term']],
                                  columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type'])
        for col, le in label_encoders.items():
            input_data[col] = le.transform(input_data[col].astype(str))

        if income > 100000 and health_status == 'Excellent':
            eligible_policies = ['Whole', 'Universal', 'Term']
        elif income > 50000 and health_status in ['Good', 'Average', 'Excellent']:
            eligible_policies = ['Universal', 'Term']
        elif income > 5000:
            eligible_policies = ['Term']
        else:
            st.error("❌ Not Eligible for Insurance")
            st.write("Reason: Income is below the minimum threshold of 5000")
            st.info("💡 Suggestion: Consider increasing your income and improving your financial stability before applying again.")
            return

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

        premium_estimates = {}
        for policy in eligible_policies:
            policy_encoded = label_encoders['Policy_Type'].transform([policy])[0]
            input_data['Policy_Type'] = policy_encoded
            premium_estimates[policy] = premium_model.predict(input_data)[0]

        st.success("\U0001F389 Eligible for Insurance")
        st.write(f"Eligible Policies: {', '.join(eligible_policies)}")
        st.write("Estimated Premiums:")
        for policy, premium in premium_estimates.items():
            st.write(f"- {policy}: {premium:.2f}")

        # st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        st.write("🔗 **Recommended Insurance Providers:**")
        for policy in eligible_policies:
            st.write(f"**{policy} Insurance:**")
            for link in company_links[policy]:
                st.write(f"- {link}")

        st.info("💡 Important Advice: If you miss paying your premium, you may face policy lapses, additional charges, or loss of coverage.")
        st.info("📌 If you ever face financial difficulty, check with your insurer about grace periods, premium holidays, or policy adjustments to maintain coverage.")

if __name__ == "__main__":
    predict_insurance()
