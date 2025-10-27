# HEART-DESEASE-PREDICTION-N
❤️ Heart Disease Prediction Using Machine Learning

Description:
This project focuses on predicting the presence of heart disease in patients using machine learning techniques. The model is implemented in Python and utilizes libraries such as Scikit-learn, NumPy, Pandas, and Joblib for data analysis, model training, and deployment.

The dataset contains various medical attributes such as HighBP', 'HighChol', 'CholCheck', 'BMI',
       'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age  which are key indicators of cardiovascular health. Data preprocessing and feature engineering are performed using Pandas and NumPy to ensure data quality and improve model accuracy.

A Random Forest Classifier is trained using Scikit-learn to predict whether a patient is likely to have heart disease. The trained model is then serialized and saved using Joblib, enabling efficient model reuse without retraining.

Key Steps:

A Flask-based web app serves as the front end, where users can input car details and receive instant  predictions.

Import and explore the dataset using Pandas.

Clean and preprocess data (handle missing values, encode categorical features, normalize data).

Split the data into training and testing sets.

Train a KNeighborsClassifier using Scikit-learn.

Evaluate the model using metrics such as accuracy, confusion matrix, and classification report.

Save the trained model with Joblib for future predictions.

Tools & Libraries Used:

Python

Scikit-learn – model building, evaluation, and data preprocessing

NumPy – numerical computations

Pandas – data manipulation and analysis

Joblib – model saving and loading

KNeighborsClassifier – ensemble algorithm for classification

Outcome:
The project successfully builds a predictive system that identifies patients at risk of heart disease based on medical parameters. The model can be integrated into healthcare applications for early detection and preventive diagnosis.


