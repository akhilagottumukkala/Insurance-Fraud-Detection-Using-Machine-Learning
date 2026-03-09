from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model Assets
model = pickle.load(open('fraud_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Create a base dataframe filled with 0s
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0
    
    # 2. Extract Numerical values
    nums = ['total_claim_amount', 'months_as_customer']
    for n in nums:
        input_df.at[0, n] = int(request.form.get(n, 0))
    
    # 3. Categorical Logic (The most critical part)
    cats = ['incident_severity', 'insured_hobbies', 'collision_type', 'authorities_contacted']
    for cat in cats:
        val = request.form.get(cat)
        col_name = f"{cat}_{val}"
        # Ensure the column exists in your model columns before setting to 1
        if col_name in model_columns:
            input_df.at[0, col_name] = 1

    # 4. Predict Proba for Granular Control
    prob = model.predict_proba(input_df)[0][1] # Probability of fraud (Class 1)
    
    # Convert probability to a clean percentage for the UI
    score_percentage = round(prob * 100, 1)

    return render_template('index.html', score=score_percentage)

if __name__ == "__main__":
    app.run(debug=True)