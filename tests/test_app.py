import pytest
from flask import Flask
from app import create_app
from app.models import ToxicityModel

# Przygotowanie aplikacji testowej
@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def model():
    # Inicjalizacja modelu do testów
    return ToxicityModel('model/best_model_cnn.keras', 'model/tokenizer.pkl')

# Test klasy ToxicityModel
def test_toxicity_model_prediction(model):
    test_comment = "You are an idiot."
    result = model.predict_toxicity(test_comment)
    assert isinstance(result, list)  # Wynik powinien być listą
    assert len(result) == 6  # Model ma przewidywać 6 klas
    for score in result:
        assert 0 <= score <= 1  # Wynik każdej klasy powinien być w zakresie [0, 1]

# Test endpointu /api/check_toxicity
def test_check_toxicity_endpoint(client):
    # Wysłanie prawidłowego żądania
    response = client.post('/api/check_toxicity', json={"comment": "I hate you."})
    assert response.status_code == 200  # Sprawdź poprawny kod HTTP
    data = response.get_json()
    assert "toxicity_score" in data  # Sprawdź, czy wynik zawiera "toxicity_score"
    assert len(data['toxicity_score']) == 6  # Sprawdź, czy są 6 wyniki klas

def test_check_toxicity_invalid_request(client):
    # Wysłanie żądania bez komentarza
    response = client.post('/api/check_toxicity', json={})
    assert response.status_code == 400  # Sprawdź kod błędu
    data = response.get_json()
    assert "error" in data  # Sprawdź, czy zwracana jest informacja o błędzie

def test_check_toxicity_internal_error(client, monkeypatch):
    # Symulacja błędu w predykcji
    def mock_predict_toxicity(*args, **kwargs):
        raise Exception("Mocked error")

    monkeypatch.setattr('app.models.ToxicityModel.predict_toxicity', mock_predict_toxicity)
    response = client.post('/api/check_toxicity', json={"comment": "Some comment."})
    assert response.status_code == 500  # Sprawdź kod błędu serwera
    data = response.get_json()
    assert "error" in data
