from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_health():
    # Since the model won't be loaded in a standard test environment, 
    # we expect the model_loaded status to be false or for the startup to fail.
    # We can mock the engine if needed, but a simple health check is a good start.
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

def test_generate_fail_no_model():
    # Test that /generate returns 503 if engine is not initialized
    response = client.post(
        "/generate",
        json={"instruction": "How do I reset my password?"}
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"
