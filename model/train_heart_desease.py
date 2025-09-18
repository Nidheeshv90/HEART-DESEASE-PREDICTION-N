import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("C:/project file/heart_disease_health_indicators_BRFSS2015.csv")

# Define features and target
x = df[['HighBP', 'HighChol', 'CholCheck', 'BMI',
        'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Age','Sex']]
y = df['HeartDiseaseorAttack']

# Identify categorical columns
categorical_cols = x.select_dtypes(include='object').columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Define pipeline with SVC (or switch to RandomForestClassifier as needed)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', SVC())
])

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

# Fit the pipeline
pipeline.fit(x_train, y_train)

# Make predictions
y_pred = pipeline.predict(x_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
model_path: str = "heart_disease_model.pkl"
joblib.dump(pipeline, model_path)

print("Model training and saving successful.")
