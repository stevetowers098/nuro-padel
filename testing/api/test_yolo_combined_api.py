"""
API tests for YOLO Combined service endpoints
"""
import pytest
import requests
import json

@pytest.mark.api
def test_yolo11_pose_endpoint(sample_request_data, wait_for_services):
    """Test YOLO11 pose detection endpoint"""
    response = requests.post(
        "http://localhost:8001/yolo11/pose",
        json=sample_request_data,
        timeout=30
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "data" in data
    assert "poses_per_frame" in data["data"]
    assert isinstance(data["data"]["poses_per_frame"], list)

@pytest.mark.api
def test_yolo11_object_endpoint(sample_request_data, wait_for_services):
    """Test YOLO11 object detection endpoint"""
    response = requests.post(
        "http://localhost:8001/yolo11/object",
        json=sample_request_data,
        timeout=30
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "data" in data
    assert "objects_per_frame" in data["data"]
    assert isinstance(data["data"]["objects_per_frame"], list)

@pytest.mark.api
def test_yolov8_pose_endpoint(sample_request_data, wait_for_services):
    """Test YOLOv8 pose detection endpoint"""
    response = requests.post(
        "http://localhost:8001/yolov8/pose",
        json=sample_request_data,
        timeout=30
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "data" in data
    assert "poses_per_frame" in data["data"]

@pytest.mark.api
def test_yolov8_object_endpoint(sample_request_data, wait_for_services):
    """Test YOLOv8 object detection endpoint"""
    response = requests.post(
        "http://localhost:8001/yolov8/object",
        json=sample_request_data,
        timeout=30
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "data" in data
    assert "objects_per_frame" in data["data"]

@pytest.mark.api
def test_enhanced_ball_tracking(sample_request_data, wait_for_services):
    """Test enhanced ball tracking endpoint"""
    response = requests.post(
        "http://localhost:8001/track-ball",
        json=sample_request_data,
        timeout=30
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "data" in data
    assert "objects_per_frame" in data["data"]

@pytest.mark.api
def test_invalid_video_url():
    """Test API response with invalid video URL"""
    invalid_data = {
        "video_url": "https://invalid-url.com/nonexistent.mp4",
        "video": False,
        "data": True
    }
    
    response = requests.post(
        "http://localhost:8001/yolo11/pose",
        json=invalid_data,
        timeout=30
    )
    
    # Should return error for invalid URL
    assert response.status_code in [400, 422, 500]

@pytest.mark.api
def test_confidence_parameter():
    """Test confidence parameter affects results"""
    high_confidence_data = {
        "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
        "video": False,
        "data": True,
        "confidence": 0.9  # High confidence
    }
    
    low_confidence_data = {
        "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
        "video": False,
        "data": True,
        "confidence": 0.1  # Low confidence
    }
    
    # Test with high confidence
    high_response = requests.post(
        "http://localhost:8001/yolo11/object",
        json=high_confidence_data,
        timeout=30
    )
    
    # Test with low confidence
    low_response = requests.post(
        "http://localhost:8001/yolo11/object",
        json=low_confidence_data,
        timeout=30
    )
    
    assert high_response.status_code == 200
    assert low_response.status_code == 200
    
    # Low confidence should detect more objects (or equal)
    high_data = high_response.json()
    low_data = low_response.json()
    
    high_objects = sum(len(frame) for frame in high_data["data"]["objects_per_frame"])
    low_objects = sum(len(frame) for frame in low_data["data"]["objects_per_frame"])
    
    assert low_objects >= high_objects