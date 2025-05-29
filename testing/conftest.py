"""
Test configuration and fixtures for NuroPadel testing suite
"""
import pytest
import requests
import time
import json
from pathlib import Path

# Test configuration
TEST_BASE_URL = "http://localhost:8080"
SERVICES = {
    "yolo-combined": {"port": 8001, "endpoints": ["/yolo11/pose", "/yolo11/object", "/yolov8/pose", "/yolov8/object", "/track-ball"]},
    "mmpose": {"port": 8003, "endpoints": ["/mmpose/pose"]},
    "yolo-nas": {"port": 8004, "endpoints": ["/yolo-nas/pose", "/yolo-nas/object"]}
}

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def test_video_url():
    """Provide a test video URL for API testing"""
    return "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"

@pytest.fixture(scope="session")
def test_payload():
    """Standard test payload for API requests"""
    return {
        "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
        "video": False,
        "data": True,
        "confidence": 0.3
    }

@pytest.fixture(scope="session")
def wait_for_services():
    """Wait for all services to be healthy before running tests"""
    max_wait = 60  # seconds
    for service_name, config in SERVICES.items():
        url = f"http://localhost:{config['port']}/healthz"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        print(f"✅ {service_name} is healthy")
                        break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        else:
            pytest.fail(f"Service {service_name} did not become healthy within {max_wait} seconds")

@pytest.fixture
def api_client():
    """HTTP client for API testing"""
    return requests.Session()

@pytest.fixture
def sample_request_data():
    """Sample data for testing API requests"""
    return {
        "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
        "video": False,
        "data": True,
        "confidence": 0.5
    }

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "health: mark test as health check")

def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location"""
    for item in items:
        # Add markers based on file location
        if "test_health" in item.nodeid:
            item.add_marker(pytest.mark.health)
        if "api" in item.fspath.dirname:
            item.add_marker(pytest.mark.api)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)