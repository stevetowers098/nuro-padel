"""
Health check tests for all NuroPadel services
"""
import pytest
import requests

@pytest.mark.health
def test_yolo_combined_health(wait_for_services):
    """Test YOLO Combined service health endpoint"""
    response = requests.get("http://localhost:8001/healthz")
    assert response.status_code == 200
    
    health_data = response.json()
    assert health_data["status"] == "healthy"
    assert "models" in health_data
    
    # Check specific models are loaded
    models = health_data["models"]
    assert models.get("yolo11_pose") is True
    assert models.get("yolov8_object") is True
    assert models.get("yolov8_pose") is True

@pytest.mark.health
def test_mmpose_health(wait_for_services):
    """Test MMPose service health endpoint"""
    response = requests.get("http://localhost:8003/healthz")
    assert response.status_code == 200
    
    health_data = response.json()
    assert health_data["status"] == "healthy"
    assert health_data["mmpose_available"] is True

@pytest.mark.health
def test_yolo_nas_health(wait_for_services):
    """Test YOLO-NAS service health endpoint"""
    response = requests.get("http://localhost:8004/healthz")
    assert response.status_code == 200
    
    health_data = response.json()
    assert health_data["status"] == "healthy"
    assert "models" in health_data
    
    models = health_data["models"]
    assert models.get("super_gradients_available") is True

@pytest.mark.health
def test_nginx_load_balancer_health():
    """Test Nginx load balancer health routing"""
    # Test global health endpoint
    response = requests.get("http://localhost:8080/healthz")
    assert response.status_code == 200
    
    # Test service routing through load balancer
    services = ["yolo-combined", "mmpose", "yolo-nas"]
    for service in services:
        response = requests.get(f"http://localhost:8080/{service}/healthz")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"

@pytest.mark.health
def test_all_services_responding():
    """Test that all critical services are responding"""
    critical_endpoints = [
        "http://localhost:8001/healthz",  # YOLO Combined
        "http://localhost:8003/healthz",  # MMPose
        "http://localhost:8004/healthz",  # YOLO-NAS
        "http://localhost:8080/healthz"   # Load Balancer
    ]
    
    for endpoint in critical_endpoints:
        response = requests.get(endpoint, timeout=10)
        assert response.status_code == 200, f"Service at {endpoint} is not responding"