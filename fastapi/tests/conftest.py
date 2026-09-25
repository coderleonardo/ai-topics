import pytest
from fastapi.testclient import TestClient

from src.project.env_settup.app import app


@pytest.fixture
def client():
    return TestClient(app)
