import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Load the trained model and feature columns
model = joblib.load('xgboost_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Function to preprocess input data
def preprocess_data(df):
    # Select the required columns
    df = df[['City/Locality', 'BHK', 'Property Size (sqft)', 'Furnishing']].reset_index(drop=True)

    # Convert BHK and Property Size to correct types
    df['BHK'] = df['BHK'].astype(int)
    df['Property Size (sqft)'] = df['Property Size (sqft)'].astype(float)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['City/Locality', 'Furnishing'])

    return df

# Route for home
@app.route('/')
def home():
    return "Welcome to the House Price Prediction API!"

# Route for predicting house prices
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request
    data = request.json
    location = data.get('location')
    bhk = data.get('bhk')
    furnishing = data.get('furnishing')
    size = data.get('size')

    # Ensure inputs are correctly typed
    bhk = int(bhk)
    size = float(size)
    
    # Prepare input data for prediction
    input_data = {
        'BHK': [bhk],
        'Property Size (sqft)': [size]
    }
    
    # One-hot encode categorical variables
    for col in feature_columns:
        if 'City/Locality_' in col:
            input_data[col] = [1 if col == f'City/Locality_{location}' else 0]
        elif 'Furnishing_' in col:
            input_data[col] = [1 if col == f'Furnishing_{furnishing}' else 0]
    
    input_df = pd.DataFrame(input_data)

    # Align input_df with feature columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Convert columns to correct data types
    input_df['BHK'] = input_df['BHK'].astype(int)
    input_df['Property Size (sqft)'] = input_df['Property Size (sqft)'].astype(float)

    # Predict the price
    predicted_price = model.predict(input_df)[0]
    
    # Convert predicted_price to standard Python float
    predicted_price = float(predicted_price)
    
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)