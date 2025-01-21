import pytest
from flask import Flask
from app import create_app
from app.models import ToxicityModel

@pytest.fixture
def app():
    from app import create_app
    from app.routes import bp 

    app = create_app()
    app.register_blueprint(bp) 
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def model():
    return ToxicityModel('model/best_model_cnn.keras', 'model/tokenizer.pkl')

def test_toxicity_model_prediction(model):
    test_comment = "You are an idiot."
    result = model.predict_toxicity(test_comment)
    assert isinstance(result, list)
    assert len(result) == 6
    for score in result:
        assert 0 <= score <= 1

def test_check_toxicity_endpoint(client):
    response = client.post('/api/check_toxicity', json={"comment": "I hate you."})
    assert response.status_code == 200
    data = response.get_json()
    assert "toxicity_score" in data
    assert len(data['toxicity_score']) == 6

def test_check_toxicity_invalid_request(client):
    response = client.post('/api/check_toxicity', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_check_toxicity_internal_error(client, monkeypatch):
    def mock_predict_toxicity(*args, **kwargs):
        raise Exception("Mocked error")

    monkeypatch.setattr('app.models.ToxicityModel.predict_toxicity', mock_predict_toxicity)
    response = client.post('/api/check_toxicity', json={"comment": "Some comment."})
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
