from http import HTTPStatus

from fastapi.testclient import TestClient

from src.project.env_settup.app import app


def test_root():
    client = TestClient(app)  # Arrange

    response = client.get("/")  # Act - SUT

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"message": "hello everyone"}  # Assert
