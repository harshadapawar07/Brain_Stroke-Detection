from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model, scaler, and encoder
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('brain_stroke.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# One-hot encoding column order from training
encoder_columns = [
    'Gender_Female', 'Gender_Male',
    'Smoking Status_Current', 'Smoking Status_Former', 'Smoking Status_Never',
    'Physical Activity_Active', 'Physical Activity_Moderate', 'Physical Activity_Sedentary'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json

        # Create a DataFrame with the input data
        input_df = pd.DataFrame([{
            'Age': data['Age'],
            'BP': data['BP'],
            'ICP': data['ICP'],
            'CBF': data['CBF'],
            'BT': data['BT'],
            'CPP': data['CPP'],
            'WBC': data['WBC'],
            'Gender': data['Gender'],
            'Smoking Status': data['Smoking Status'],
            'Physical Activity': data['Physical Activity']
        }])

        # One-hot encode the categorical variables
        encoded_df = pd.get_dummies(input_df, columns=['Gender', 'Smoking Status', 'Physical Activity'])

        # Ensure the one-hot encoded DataFrame has all necessary columns (from training)
        for col in encoder_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0  # Add missing columns with a default value of 0

        # Reorder the columns to match the training order
        encoded_df = encoded_df[encoder_columns]

        # Combine with the numeric columns
        numeric_features = input_df[['Age', 'BP', 'ICP', 'CBF', 'BT', 'CPP', 'WBC']]
        final_input = pd.concat([numeric_features, encoded_df], axis=1)

        # Scale the input data
        input_scaled = scaler.transform(final_input)

        # Make a prediction
        prediction = model.predict(input_scaled)[0]

        # Return the prediction
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
