import sys
import warnings
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    from app.main import create_app
    from mock_services.mock_server import app as mock
    with TestClient(create_app(hr_http=TestClient(mock))) as c:
        yield c


@pytest.fixture(scope="session")
def keyed_client():
    from app.main import create_app
    from mock_services.mock_server import app as mock
    with TestClient(create_app(hr_http=TestClient(mock), api_key="secret-test-key")) as c:
        yield c
