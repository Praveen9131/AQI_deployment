from flask import Flask, render_template, request
import pandas as pd
import pickle

loaded_model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

app = Flask(__name__)

# Replace these with the actual feature names used during training
features_used_during_training = ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_names = features_used_during_training  # Use the correct feature names
    feature_values = [request.form.get(feature) for feature in feature_names]

    # Check if any feature value is missing
    if any(value is None for value in feature_values):
        return "Error: Please provide values for all features."

    # Create a DataFrame with the input features
    df = pd.DataFrame([feature_values], columns=feature_names)

    # Perform prediction
    my_prediction = loaded_model.predict(df)
    my_prediction = my_prediction.tolist()

    return render_template('app.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(debug=True)
