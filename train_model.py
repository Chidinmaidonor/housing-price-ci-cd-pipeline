# train_model.py
import joblib
from sklearn.linear_model import LinearRegression
from preprocessing import load_data, preprocess

def train():
    # Load dataset
    df = load_data("data/housing.csv")

    # Preprocess
    df = preprocess(df)

    # Select features (X) and target (y)
    X = df.drop(columns=["price"])
    y = df["price"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    joblib.dump(model, "model.pkl")
    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    train()
