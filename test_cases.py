import pandas as pd
from app import train_model

def run_test_cases():
    model, premium_model, label_encoders, accuracy = train_model()
    test_data = [
        {'Age': 25, 'Gender': 'Male', 'Income': 80000, 'Health_Status': 'Good', 'Smoking_Habit': 'No', 'Policy_Type': 'Term'},
        {'Age': 40, 'Gender': 'Female', 'Income': 120000, 'Health_Status': 'Excellent', 'Smoking_Habit': 'No', 'Policy_Type': 'Whole'},
        {'Age': 16, 'Gender': 'Male', 'Income': 20000, 'Health_Status': 'Average', 'Smoking_Habit': 'Yes', 'Policy_Type': 'Term'},
        {'Age': 30, 'Gender': 'Female', 'Income': 4000, 'Health_Status': 'Poor', 'Smoking_Habit': 'No', 'Policy_Type': 'Term'}
    ]

    for idx, test in enumerate(test_data):
        print(f"Test Case {idx + 1}:")
        print(f"Input: {test}")
        
        # Check for underage smoking condition
        if test['Age'] < 18 and test['Smoking_Habit'] == 'Yes':
            print("Not Eligible (Underage smoking detected)\n")
            continue

        input_data = pd.DataFrame([test])
        for col, le in label_encoders.items():
            input_data[col] = le.transform(input_data[col].astype(str))
        
        if test['Income'] > 100000 and test['Health_Status'] == 'Excellent':
            eligible_policies = ['Whole', 'Universal', 'Term']
        elif test['Income'] > 50000 and test['Health_Status'] in ['Good', 'Average']:
            eligible_policies = ['Universal', 'Term']
        elif test['Income'] > 5000:
            eligible_policies = ['Term']
        else:
            print("Not Eligible (Income below 5000)\n")
            continue

        premium_estimates = {}
        for policy in eligible_policies:
            policy_encoded = label_encoders['Policy_Type'].transform([policy])[0]
            input_data['Policy_Type'] = policy_encoded
            premium_estimates[policy] = premium_model.predict(input_data)[0]

        print(f"Eligible Policies: {', '.join(eligible_policies)}")
        print("Estimated Premiums:")
        for policy, premium in premium_estimates.items():
            print(f"- {policy}: {premium:.2f}")
        print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

if __name__ == "__main__":
    run_test_cases()