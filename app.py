from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd # Import pandas for DataFrame creation
import os

# Import XGBoost for dummy model creation and loading
import xgboost as xgb
from sklearn.preprocessing import StandardScaler # Needed for dummy scaler

app = Flask(__name__)

# Define the path to the saved models
MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
DEFAULT_VALUES_PATH = os.path.join(MODEL_DIR, 'default_feature_values.pkl')

# Load the trained model, scaler, and default feature values
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open(DEFAULT_VALUES_PATH, 'rb') as defaults_file:
        default_feature_values = pickle.load(defaults_file)
    print("Model, Scaler, and Default Feature Values loaded successfully!")
except FileNotFoundError:
    print(f"Error: One or more files not found. Please ensure '{MODEL_PATH}', '{SCALER_PATH}', and '{DEFAULT_VALUES_PATH}' exist.")
    print("Run the Jupyter notebook 'heart_failure_prediction.ipynb' to train and save all necessary files.")
    model = None
    scaler = None
    default_feature_values = {} # Initialize empty to prevent errors

# Define ALL feature names in the order expected by the model
# This order MUST match the columns in your training data (X)
ALL_FEATURE_NAMES = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

# For best accuracy, all features are user-inputted.
# The 'required' attribute in HTML will enforce this.
USER_INPUT_FEATURES = ALL_FEATURE_NAMES

@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if model is None or scaler is None or not default_feature_values:
        return render_template('index.html', prediction_text="Error: Model, Scaler, or Default Values not loaded. Please check server logs.", error=True)

    try:
        # Start with default values for all features (as a fallback, though all should be provided)
        input_data = default_feature_values.copy()

        # Override default values with user inputs
        form_data = request.form
        for feature in USER_INPUT_FEATURES:
            # All fields are required in HTML, so they should always be present and not empty
            value = float(form_data[feature])
            input_data[feature] = value

        # Ensure the input data is in the correct order for the model
        # Create a list of values in the order of ALL_FEATURE_NAMES
        ordered_input_values = [input_data[feature] for feature in ALL_FEATURE_NAMES]

        # Convert to numpy array and reshape for single prediction
        features_array = np.array(ordered_input_values).reshape(1, -1)

        # Scale the input features using the loaded scaler
        scaled_features = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(scaled_features)[0] # Get the first (and only) prediction
        prediction_proba = model.predict_proba(scaled_features)[0] # Get probabilities

        # Interpret the prediction
        if prediction == 1:
            result_text = f"Prediction: High risk of Heart Failure (Probability: {prediction_proba[1]*100:.2f}%)"
            is_death_event = True
        else:
            result_text = f"Prediction: Low risk of Heart Failure (Probability: {prediction_proba[0]*100:.2f}%)"
            is_death_event = False

        return render_template('index.html', prediction_text=result_text, is_death_event=is_death_event)

    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter numerical values for all fields.", error=True)
    except Exception as e:
        return render_template('index.html', prediction_text=f"An error occurred: {e}", error=True)

if __name__ == '__main__':
    # Create 'saved_models' directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")

    # Create 'templates' directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created directory: templates")

    # Dummy model, scaler, and default values creation if they don't exist for initial run/testing
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(DEFAULT_VALUES_PATH):
        print("Creating dummy model, scaler, and default values for initial run. Please train your actual model in the notebook.")
        # Create dummy data matching the expected feature count
        dummy_X = pd.DataFrame(np.random.rand(5, len(ALL_FEATURE_NAMES)), columns=ALL_FEATURE_NAMES)
        dummy_y = np.array([0, 1, 0, 1, 0])

        dummy_scaler = StandardScaler()
        dummy_X_scaled = dummy_scaler.fit_transform(dummy_X)

        dummy_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) # Minimal params for dummy
        dummy_model.fit(dummy_X_scaled, dummy_y)

        dummy_default_values = dummy_X.mean().to_dict() # Simple mean for dummy
        for bf in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']:
            if bf in dummy_default_values:
                dummy_default_values[bf] = dummy_X[bf].mode()[0]


        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(dummy_model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(dummy_scaler, f)
        with open(DEFAULT_VALUES_PATH, 'wb') as f:
            pickle.dump(dummy_default_values, f)
        print("Dummy model, scaler, and default values created.")

    app.run(debug=True) # debug=True allows for automatic reloading on code changes
