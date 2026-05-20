from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"
    assert "environment" in response.json()

def test_generate_fail_no_model():
    response = client.post(
        "/generate",
        json={"instruction": "How do I reset my password?"}
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"
