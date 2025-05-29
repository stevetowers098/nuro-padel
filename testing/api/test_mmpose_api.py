"""
API tests for MMPose biomechanical analysis service
"""
import pytest
import requests

@pytest.mark.api
@pytest.mark.slow
def test_mmpose_pose_analysis(sample_request_data, wait_for_services):
    """Test MMPose biomechanical analysis endpoint"""
    response = requests.post(
        "http://localhost:8003/mmpose/pose",
        json=sample_request_data,
        timeout=60  # MMPose can be slower
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "data" in data
    assert "biomechanics_per_frame" in data["data"]
    assert isinstance(data["data"]["biomechanics_per_frame"], list)
    
    # Check model info
    assert "model_info" in data["data"]
    assert data["data"]["model_info"]["name"] in ["rtmpose-m", "hrnet-w48"]

@pytest.mark.api
def test_mmpose_biomechanical_metrics(sample_request_data, wait_for_services):
    """Test that biomechanical metrics are calculated"""
    response = requests.post(
        "http://localhost:8003/mmpose/pose",
        json=sample_request_data,
        timeout=60
    )
    
    assert response.status_code == 200
    data = response.json()
    
    biomechanics = data["data"]["biomechanics_per_frame"]
    if len(biomechanics) > 0:
        frame_data = biomechanics[0]
        
        # Check for biomechanical metrics
        if "biomechanical_metrics" in frame_data:
            metrics = frame_data["biomechanical_metrics"]
            
            # Should have key metrics
            expected_metrics = ["posture_score", "balance_score", "movement_efficiency", "power_potential"]
            for metric in expected_metrics:
                if metric in metrics:
                    assert isinstance(metrics[metric], (int, float))
                    assert 0 <= metrics[metric] <= 100  # Scores should be 0-100

@pytest.mark.api
def test_mmpose_keypoint_structure(sample_request_data, wait_for_services):
    """Test MMPose keypoint data structure"""
    response = requests.post(
        "http://localhost:8003/mmpose/pose",
        json=sample_request_data,
        timeout=60
    )
    
    assert response.status_code == 200
    data = response.json()
    
    biomechanics = data["data"]["biomechanics_per_frame"]
    if len(biomechanics) > 0:
        frame_data = biomechanics[0]
        
        if "keypoints" in frame_data and frame_data["keypoints"]:
            keypoints = frame_data["keypoints"]
            
            # Check keypoint structure
            expected_keypoints = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
            
            for keypoint_name, keypoint_data in keypoints.items():
                assert keypoint_name in expected_keypoints
                assert "x" in keypoint_data
                assert "y" in keypoint_data
                assert "confidence" in keypoint_data
                assert isinstance(keypoint_data["confidence"], float)
                assert 0 <= keypoint_data["confidence"] <= 1

@pytest.mark.api
def test_mmpose_video_generation():
    """Test MMPose video generation capability"""
    video_request = {
        "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
        "video": True,  # Request video output
        "data": True
    }
    
    response = requests.post(
        "http://localhost:8003/mmpose/pose",
        json=video_request,
        timeout=90  # Video generation takes longer
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should have both data and video URL
    assert "data" in data
    if "video_url" in data:
        assert isinstance(data["video_url"], str)
        assert data["video_url"].startswith("http")