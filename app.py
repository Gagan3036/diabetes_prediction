from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('diabetes_prediction_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        pregnancies = float(data['pregnancies'])
        glucose = float(data['glucose'])
        diastolic = float(data['diastolic'])
        triceps = float(data['triceps'])
        insulin = float(data['insulin'])
        bmi = float(data['bmi'])
        dpf = float(data['dpf'])
        age = float(data['age'])

        # Create input array
        input_data = np.array([[pregnancies, glucose, diastolic, triceps, insulin, bmi, dpf, age]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return prediction as JSON response
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        # Return error message if prediction fails
        return jsonify({'error': str(e)})


