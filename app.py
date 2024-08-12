import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, url_for, render_template

from src.utils import pre_processing, NUMERICAL_VARIBLES, BINARY_VARIBLES, CATEGORICAL_VARIBLES


app = Flask(__name__)

# Load the Models
rf_model = pickle.load(open('models/random_forest.pkl', 'rb'))

# Initializations
input_features = NUMERICAL_VARIBLES+BINARY_VARIBLES+CATEGORICAL_VARIBLES

def loan_status(value):
    if value == 1:
        return 'Defaulter'
    elif value == 0:
        return 'Not a Defaulter'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data={}
    
    for key in request.form.keys():
        data[key] = request.form[key]
    
    x = pd.DataFrame([data])[input_features]
    x = pre_processing(x)
    
    # Predict with Random Forest model
    y_predict_rf = rf_model.predict(x)
    
        
    return render_template('home.html', 
                           prediction = loan_status(y_predict_rf[0]))

if __name__ == '__main__':
    app.run(debug=True)
    
    
    