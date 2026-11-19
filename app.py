
import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("breast_cancer_scaler.pkl")

# Feature names
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se", "texture_se", "perimeter_se",
    "area_se", "smoothness_se", "compactness_se",
    "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se",

    "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]

# Page setup
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺")

st.title("Breast Cancer Prediction")

st.write(
    "Enter the required medical features to predict the result."
)

st.subheader("Prediction Input")
st.write("Enter all 30 feature values:")

# User inputs
features = []

for name in feature_names:
    value = st.number_input(
        name,
        min_value=0.0,
        value=0.0,
        format="%.6f"
    )

    features.append(value)

# Prediction
if st.button("Predict"):

    # Convert input into array
    input_data = np.array(features).reshape(1, -1)

    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Display result
    if prediction[0] == 0:
        st.success("Prediction: Benign")

    else:
        st.error("Prediction: Malignant")

    st.caption(
        "Educational demonstration only. "
        "This result is not a medical diagnosis."
    )
