# predict.py
import joblib
import pandas as pd
from preprocessing import preprocess

def predict(new_data):
    # Load model
    model = joblib.load("model.pkl")

    # Convert input data to DataFrame
    df = pd.DataFrame([new_data])

    # Preprocess input
    df = preprocess(df)

    # Predict
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    example_input = {
       
        "area": 1500,
        "bedrooms": 3,
        "bathrooms": "2",
        "stories": "yes",
        "mainroad": "no",
        "guestroom": "yes",
        "basement": "2",
        "hotwaterheating": "yes",
        "airconditioning": "no",
        "parking": "yes",
        "prefarea": "no",
        "furnishingstatus": "furnished",

    }
    result = predict(example_input)
    print(f"Predicted Price: {result}")
