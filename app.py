import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

# Load the saved model, KBest, and Scaler
model_filename = 'ann_model.pkl'
kbest_filename = 'kbest_selector.pkl'
scaler_filename = 'scaler.pkl'

with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

with open(kbest_filename, 'rb') as kbest_file:
    kbest_selector = pickle.load(kbest_file)

with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title and Description
st.title("Breast Cancer Prediction App")
st.write("""
This application predicts the likelihood of breast cancer being malignant or benign based on user input features. 
Please enter the following cell measurements to receive a prediction:
""")

# Input fields for the user with help text
radius_mean = st.text_input("Radius Mean", help="Mean of distances from the center to points on the perimeter of the cell nuclei")
perimeter_mean = st.text_input("Perimeter Mean", help="Mean value of the perimeter of the cell nuclei")
area_mean = st.text_input("Area Mean", help="Mean value of the area of the cell nuclei")
concavity_mean = st.text_input("Concavity Mean", help="Mean severity of concave portions of the cell nuclei contour")
concave_points_mean = st.text_input("Concave Points Mean", help="Mean number of concave portions of the cell nuclei contour")
radius_worst = st.text_input("Radius Worst", help="Worst (largest) value for the radius of the cell nuclei")
perimeter_worst = st.text_input("Perimeter Worst", help="Worst (largest) value for the perimeter of the cell nuclei")
area_worst = st.text_input("Area Worst", help="Worst (largest) value for the area of the cell nuclei")
concavity_worst = st.text_input("Concavity Worst", help="Worst (largest) value for the concavity of the cell nuclei")
concave_points_worst = st.text_input("Concave Points Worst", help="Worst (largest) value for the concave points of the cell nuclei")

# Button to trigger prediction
if st.button("Predict"):
    # Function to validate and convert input to float
    def validate_and_convert(value, label):
        try:
            return float(value)
        except ValueError:
            st.error(f"Please enter a valid number for {label}.")
            return None

    # Validate and convert inputs
    radius_mean = validate_and_convert(radius_mean, "Radius Mean")
    perimeter_mean = validate_and_convert(perimeter_mean, "Perimeter Mean")
    area_mean = validate_and_convert(area_mean, "Area Mean")
    concavity_mean = validate_and_convert(concavity_mean, "Concavity Mean")
    concave_points_mean = validate_and_convert(concave_points_mean, "Concave Points Mean")
    radius_worst = validate_and_convert(radius_worst, "Radius Worst")
    perimeter_worst = validate_and_convert(perimeter_worst, "Perimeter Worst")
    area_worst = validate_and_convert(area_worst, "Area Worst")
    concavity_worst = validate_and_convert(concavity_worst, "Concavity Worst")
    concave_points_worst = validate_and_convert(concave_points_worst, "Concave Points Worst")

    # Check if all required inputs are provided and valid
    if None in [radius_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean, radius_worst, perimeter_worst, area_worst, concavity_worst, concave_points_worst]:
        st.error("Please provide valid values for all features.")
    else:
        # Create DataFrame from user inputs
        input_data = {
            'radius_mean': radius_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'concavity_mean': concavity_mean,
            'concave points_mean': concave_points_mean,
            'radius_worst': radius_worst,
            'perimeter_worst': perimeter_worst,
            'area_worst': area_worst,
            'concavity_worst': concavity_worst,
            'concave points_worst': concave_points_worst,
        }
        input_df = pd.DataFrame([input_data])

        # Ensure all columns are present in the input DataFrame
        input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # Scale the Input Data
        scaled_input = scaler.transform(input_df)

        # Apply the Feature Selector to the Scaled Input Data
        input_df_transformed = kbest_selector.transform(scaled_input)

        # Make Predictions
        prediction = model.predict(input_df_transformed)
        prediction_label = 'Malignant' if prediction[0] == 1 else 'Benign'
        st.write(f"The predicted diagnosis is: {prediction_label}")
