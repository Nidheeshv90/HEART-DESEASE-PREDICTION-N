from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/heart_disease_model.pkl')


# Route: Home / Login Page
@app.route('/')
def login():
    return render_template('login.html')


# Route: Login Form Submission
@app.route('/login', methods=['POST'])
def check_login():
    email = request.form['email']
    password = request.form['password']
    if email == 'admin@example.com' and password == 'password123':
        return redirect(url_for('input_form'))
    return "Invalid Credentials"


# Route: Input Form for Prediction
@app.route('/input')
def input_form():
    return render_template('input.html')


# Route: Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        HighBP = int(request.form['HighBP'])
        HighChol = int(request.form['HighChol'])
        CholCheck = int(request.form['CholCheck'])
        BMI = float(request.form['BMI'])
        Smoker = int(request.form['Smoker'])
        Stroke = int(request.form['Stroke'])
        Diabetes = int(request.form['Diabetes'])
        PhysActivity = int(request.form['PhysActivity'])
        Fruits = int(request.form['Fruits'])
        Veggies = int(request.form['Veggies'])
        MentHlth = float(request.form['MentHlth'])
        PhysHlth = float(request.form['PhysHlth'])
        DiffWalk = int(request.form['DiffWalk'])
        AnyHealthcare = int(request.form['AnyHealthcare'])
        NoDocbcCost = int(request.form['NoDocbcCost'])
        GenHlth = float(request.form['GenHlth'])
        HvyAlcoholConsump = int(request.form['HvyAlcoholConsump'])
        Age = int(request.form['Age'])
        Sex = int(request.form.get('Sex', 0))  # Defaults to 0 if missing

        # Prepare input
        input_df = pd.DataFrame([{
            'HighBP': HighBP,
            'HighChol': HighChol,
            'CholCheck': CholCheck,
            'BMI': BMI,
            'Smoker': Smoker,
            'Stroke': Stroke,
            'Diabetes': Diabetes,
            'PhysActivity': PhysActivity,
            'Fruits': Fruits,
            'Veggies': Veggies,
            'MentHlth': MentHlth,
            'PhysHlth': PhysHlth,
            'DiffWalk': DiffWalk,
            'AnyHealthcare': AnyHealthcare,
            'NoDocbcCost': NoDocbcCost,
            'GenHlth': GenHlth,
            'HvyAlcoholConsump': HvyAlcoholConsump,
            'Age': Age,
            'Sex': Sex,
        }])


        # Make prediction
        prediction = model.predict(input_df)[0]  # 0 or 1

        # Convert to text
        result_text = "Yes" if prediction == 1 else "No"

        return render_template("result.html", result=result_text)

    except Exception as e:
        import traceback
        return f"<h3>Error: {str(e)}<br><pre>{traceback.format_exc()}</pre></h3>"


if __name__ == '__main__':
    app.run(port=5002)