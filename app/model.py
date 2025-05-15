import joblib
from sklearn.datasets import load_iris

clf = joblib.load("iris_model.pkl")
iris = load_iris()

def predict_species(features: list[float]) -> str:
    prediction = clf.predict([features])
    return iris.target_names[prediction[0]]

def predict_with_confidence(features: list[float]) -> dict:
    proba = clf.predict_proba([features])[0]
    pred_class = clf.predict([features])[0]
    return {
        "species": iris.target_names[pred_class],
        "confidence": round(proba[pred_class], 3)
    }
