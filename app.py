from flask import Flask, request, render_template
import joblib
import numpy as np
import xgboost as xgb


app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    age = int(request.form['Age'])
    gender_male = int(request.form['Male'])
    gender_female = int(request.form['Female'])
    ec = int(request.form['Electronics And Communication'])
    cs = int(request.form['Computer Science'])
    it = int(request.form['Information Technology'])
    mech = int(request.form['Mechanical'])
    elec = int(request.form['Electrical'])
    civil = int(request.form['Civil'])
    internships = 1 if request.form['Internships'] == 'Yes' else 0
    cgpa = float(request.form['CGPA'])
    hostel = 1 if request.form['Hostel'] == 'Yes' else 0
    backlogs = 1 if request.form['HistoryOfBacklogs'] == 'Yes' else 0

    # Combine features into an array and apply scaling
    features = [
        age, gender_male, gender_female, ec, cs, it, mech, elec, civil,
        internships, cgpa, hostel, backlogs
    ]
    
    features_array = np.array(features).reshape(1, -1)
    
    # Apply scaling
    try:
        scaled_features = scaler.transform(features_array)
    except sklearn.exceptions.NotFittedError as e:
        return render_template('index.html', prediction_text='Error: Scaler not fitted. Please check your setup.')

    # Make prediction using the scaled features
    prediction = model.predict(scaled_features)
    
    output = 'Placed' if prediction[0] == 1 else 'Not Placed'
    probabilities = model.predict_proba(scaled_features)

    # For binary classification, probabilities[:, 1] gives the probability of the positive class
    probability_class_1 = probabilities[0, 1]

    
    return render_template('index.html', prediction_text='Probability of getting placed : {}%'.format(int(probability_class_1 * 100)))

if __name__ == "__main__":
    app.run(debug=True)
