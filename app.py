import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

# Title
st.title("Housing Price Prediction App")
st.write("Enter property details below to get a predicted price:")

# Collect inputs from user
area = st.number_input("Area (in 1000 sqft)", value=3)
bedrooms = st.number_input("Number of bedrooms", value=3)
bathrooms = st.number_input("Number of bathrooms", value=2)
stories = st.number_input("Number of stories", value=1)
mainroad = st.selectbox("Near main road?", [0, 1])
guestroom = st.selectbox("Has guest room?", [0, 1])
basement = st.selectbox("Has basement?", [0, 1])
hotwaterheating = st.selectbox("Has hot water heating?", [0, 1])
airconditioning = st.selectbox("Has air conditioning?", [0, 1])
parking = st.number_input("Number of parking spaces", value=1)
prefarea = st.selectbox("Preferred area?", [0, 1])
furnishingstatus = st.selectbox("Furnishing status (0=No, 1=Yes)", [0, 1])

# Predict button
if st.button("Predict Price"):
    # Prepare input as 2D array
    input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom,
                            basement, hotwaterheating, airconditioning, parking,
                            prefarea, furnishingstatus]])
    
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Price: {prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"Error: {e}")
