from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import os

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
DATA_PATH = 'backend/cpdata1.csv'  # Update this path if needed

try:
    data = pd.read_csv(r"C:\Users\Hp\Downloads\crop-recommendation-app\crop-recommendation-app\crop_recommendation-app\backend\cpdata1.csv")
except FileNotFoundError:
    raise Exception(f"Data file not found at {DATA_PATH}. Please check the file path.")

# Define columns for categorical and target column
categorical_columns = ['soil_type', 'crop_type']
target_column = 'label'

# One-hot encode the target column
label_dummies = pd.get_dummies(data[target_column])
label_names = label_dummies.columns
data = pd.concat([data, label_dummies], axis=1)
data.drop(target_column, axis=1, inplace=True)

# Define feature columns for prediction
feature_columns = ['temperature', 'humidity', 'ph', 'rainfall']
X = data[feature_columns].values
y = data[label_dummies.columns].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the DecisionTreeRegressor model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare the input data as a numpy array
        user_input = np.array([[temperature, humidity, ph, rainfall]])

        # Scale the input data
        user_input_scaled = scaler.transform(user_input)

        # Predict the crop type using the trained model
        pred = model.predict(user_input_scaled)

        # Get the predicted label
        predicted_label_index = np.argmax(pred)
        predicted_crop = label_names[predicted_label_index]

        return render_template('result.html', predicted_crop=predicted_crop)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
