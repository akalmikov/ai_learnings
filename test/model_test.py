from app.model import predict_species

def test_prediction_output():
    result = predict_species([5.1, 3.5, 1.4, 0.2])
    assert result in ["setosa", "versicolor", "virginica"]
