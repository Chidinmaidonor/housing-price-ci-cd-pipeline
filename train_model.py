# train_model.py
import joblib
from sklearn.linear_model import LinearRegression
from preprocessing import load_data, preprocess

def train():

    df = load_data("data/housing.csv")


    df = preprocess(df)

    
    X = df.drop(columns=["price"])
    y = df["price"]

    
    model = LinearRegression()
    model.fit(X, y)


    joblib.dump(model, "model.pkl")
    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    train()
