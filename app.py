import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_model():
    # Load dataset and define features and target
    data = pd.read_csv('life_insurance_prediction.csv')
    features = ['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type']
    target = 'Prediction_Target'

    # Standardize the categorical columns to have consistent case
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type']:
        data[col] = data[col].str.capitalize()

    # Label encode categorical columns
    label_encoders = {}
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Define X and y
    X = data[features]
    y = data[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the eligibility model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Train the premium model
    premium_model = RandomForestClassifier(random_state=42)
    premium_model.fit(X_train, data.loc[X_train.index, 'Premium_Amount'])

    return model, premium_model, label_encoders, accuracy

def get_user_input():
    # Collect user inputs
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

    # Train the model
    model, premium_model, label_encoders, accuracy = train_model()

    # Prepare input data for prediction
    input_data = pd.DataFrame([[age, gender, income, health_status, smoking, family_history, 'None']],
                               columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type'])

    # Encode categorical inputs
    for col in label_encoders:
        if col != 'Policy_Type':  # Don't encode 'Policy_Type' yet
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Placeholder for 'Policy_Type'
    input_data['Policy_Type'] = 0  

    # Make eligibility prediction
    prediction = model.predict(input_data)

    # Here '0' means eligible and '1' means not eligible
    if prediction[0] == 0:
        eligible_policies = []
        if income > 100000 and health_status == 'Excellent':
            eligible_policies = ['Whole', 'Universal', 'Term']
        elif income > 50000 and health_status in ['Good', 'Average']:
            eligible_policies = ['Universal', 'Term']
        else:
            eligible_policies = ['Term']

        # Estimate premiums for eligible policies
        premium_estimates = {}
        for policy in eligible_policies:
            input_data['Policy_Type'] = label_encoders['Policy_Type'].transform([policy])[0]
            premium_estimates[policy] = premium_model.predict(input_data)[0]

        return "Eligible", eligible_policies, premium_estimates, accuracy

    else:
        return "Not Eligible", None, None, accuracy

# Example usage
if __name__ == "__main__":
    result, eligible_policies, premium_estimates, accuracy = predict_insurance()
    
    print(f"Eligibility: {result}")

    if result == "Eligible":
        print(f"Eligible Policies: {eligible_policies}")
        print(f"Premium Estimates: {premium_estimates}")

    print(f"Model Accuracy: {accuracy:.2%}")
