import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load models and scaler
log_reg = joblib.load('log_reg_model.pkl')
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

def preprocess_input(input_data):
    # Convert input data to DataFrame with the correct columns
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Ensure all columns are present (fill missing columns with 0)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the order used during fitting
    input_df = input_df[feature_names]

    # Standardize the data
    scaled_data = scaler.transform(input_df)
    
    return scaled_data

# Streamlit UI
st.title('Fraud Detection Model')

# User input fields
amount = st.number_input('Transaction Amount', value=0.0)
oldbalanceOrg = st.number_input('Old Balance Origin', value=0.0)
newbalanceOrig = st.number_input('New Balance Origin', value=0.0)
oldbalanceDest = st.number_input('Old Balance Destination', value=0.0)
newbalanceDest = st.number_input('New Balance Destination', value=0.0)
sender_balance_change = st.number_input('Sender Balance Change', value=0.0)
receiver_balance_change = st.number_input('Receiver Balance Change', value=0.0)
transaction_type = st.selectbox('Transaction Type', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
step = st.number_input('Step', value=0)  # Add 'step' input if it was used during training

# Convert transaction type to one-hot encoding
transaction_type_dummies = pd.get_dummies([transaction_type], prefix='type')
transaction_type_dummies = transaction_type_dummies.reindex(columns=[col for col in feature_names if col.startswith('type_')], fill_value=0)

# Combine all input values
user_input = {
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'sender_balance_change': sender_balance_change,
    'receiver_balance_change': receiver_balance_change,
    **transaction_type_dummies.to_dict(orient='records')[0],
    'step': step  # Include 'step' if it was used during training
}

# Preprocess the input
processed_input = preprocess_input(user_input)

# Predict using each model
log_reg_pred = log_reg.predict(processed_input)
rf_pred = rf_model.predict(processed_input)
xgb_pred = xgb_model.predict(processed_input)

# Display predictions
st.subheader('Predictions')
st.write('Logistic Regression Prediction:', 'Fraud' if log_reg_pred[0] else 'Non-Fraud')
st.write('Random Forest Prediction:', 'Fraud' if rf_pred[0] else 'Non-Fraud')
st.write('XGBoost Prediction:', 'Fraud' if xgb_pred[0] else 'Non-Fraud')
