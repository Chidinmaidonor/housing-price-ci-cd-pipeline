import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Title and intro
st.title("Machine Learning Model Deployment")
st.write("This is a deployed ML model running on Streamlit Cloud!")

# Example user inputs
st.subheader("Enter your input data")
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Predict button
if st.button("Predict"):
    # Arrange inputs into the same shape your model expects
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
