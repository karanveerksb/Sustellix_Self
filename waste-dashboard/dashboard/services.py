import joblib
import pandas as pd
import numpy as np

# Load the saved transformer and model
transformer = joblib.load('transformer.joblib')
model = joblib.load('waste_model.joblib')
column_names = joblib.load('column_names.joblib')

def preprocess_data(data_dict):
    """Convert input dictionary to DataFrame and apply transformer"""
    df = pd.DataFrame([data_dict])
    transformed_data = transformer.transform(df)
    return transformed_data

def predict_waste(data_dict):
    """Make prediction using the trained model"""
    processed_data = preprocess_data(data_dict)
    prediction = model.predict(processed_data)
    return float(prediction[0])

def get_all_predictions():
    """Return pre-computed predictions for all hostels"""
    # Pre-computed predictions - hardcode these based on your notebook results
    return {
        'A': {'total': 45.6, 'per_capita': 0.049},
        'B': {'total': 38.2, 'per_capita': 0.053},
        # Add all hostels here
    }