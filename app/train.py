from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

joblib.dump(clf, "iris_model.pkl")
print("✅ Model trained and saved.")
