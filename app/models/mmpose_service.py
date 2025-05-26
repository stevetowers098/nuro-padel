from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
import tempfile
import os
import cv2
import numpy as np
import supervision as sv
from typing import Dict, Any, List, Union, Optional
import io
import sys
import logging
import math
import random
import base64
import json
import httpx
from pydantic import BaseModel, HttpUrl

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MMPose Biomechanics Service", version="1.0.0")

# Define request model
class VideoRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False
    data: bool = False

# Simulated MMPose biomechanical analysis function
# In a real implementation, this would use the actual MMPose model
def analyze_biomechanics(frame: np.ndarray) -> Dict[str, Any]:
    """
    Analyze biomechanics in a frame using MMPose.
    
    Args:
        frame: The input frame as a numpy array
        
    Returns:
        A dictionary with pose keypoints, joint angles, and biomechanical metrics
    """
    # Simulate biomechanical analysis
    # In a real implementation, this would use the actual MMPose model
    height, width = frame.shape[:2]
    
    # Simulate keypoints detection
    keypoints = {
        "nose": {"x": width // 2, "y": height // 3, "confidence": 0.95},
        "left_eye": {"x": width // 2 - width // 20, "y": height // 3 - height // 40, "confidence": 0.93},
        "right_eye": {"x": width // 2 + width // 20, "y": height // 3 - height // 40, "confidence": 0.93},
        "left_ear": {"x": width // 2 - width // 15, "y": height // 3, "confidence": 0.9},
        "right_ear": {"x": width // 2 + width // 15, "y": height // 3, "confidence": 0.9},
        "left_shoulder": {"x": width // 3, "y": height // 2, "confidence": 0.92},
        "right_shoulder": {"x": 2 * width // 3, "y": height // 2, "confidence": 0.92},
        "left_elbow": {"x": width // 4, "y": 2 * height // 3, "confidence": 0.9},
        "right_elbow": {"x": 3 * width // 4, "y": 2 * height // 3, "confidence": 0.9},
        "left_wrist": {"x": width // 5, "y": 3 * height // 4, "confidence": 0.88},
        "right_wrist": {"x": 4 * width // 5, "y": 3 * height // 4, "confidence": 0.88},
        "left_hip": {"x": 2 * width // 5, "y": 3 * height // 4, "confidence": 0.85},
        "right_hip": {"x": 3 * width // 5, "y": 3 * height // 4, "confidence": 0.85},
        "left_knee": {"x": width // 3, "y": 5 * height // 6, "confidence": 0.82},
        "right_knee": {"x": 2 * width // 3, "y": 5 * height // 6, "confidence": 0.82},
        "left_ankle": {"x": width // 4, "y": 11 * height // 12, "confidence": 0.8},
        "right_ankle": {"x": 3 * width // 4, "y": 11 * height // 12, "confidence": 0.8},
    }
    
    # Calculate joint angles
    joint_angles = {
        "left_elbow": calculate_angle(
            (keypoints["left_shoulder"]["x"], keypoints["left_shoulder"]["y"]),
            (keypoints["left_elbow"]["x"], keypoints["left_elbow"]["y"]),
            (keypoints["left_wrist"]["x"], keypoints["left_wrist"]["y"])
        ),
        "right_elbow": calculate_angle(
            (keypoints["right_shoulder"]["x"], keypoints["right_shoulder"]["y"]),
            (keypoints["right_elbow"]["x"], keypoints["right_elbow"]["y"]),
            (keypoints["right_wrist"]["x"], keypoints["right_wrist"]["y"])
        ),
        "left_shoulder": calculate_angle(
            (keypoints["left_hip"]["x"], keypoints["left_hip"]["y"]),
            (keypoints["left_shoulder"]["x"], keypoints["left_shoulder"]["y"]),
            (keypoints["left_elbow"]["x"], keypoints["left_elbow"]["y"])
        ),
        "right_shoulder": calculate_angle(
            (keypoints["right_hip"]["x"], keypoints["right_hip"]["y"]),
            (keypoints["right_shoulder"]["x"], keypoints["right_shoulder"]["y"]),
            (keypoints["right_elbow"]["x"], keypoints["right_elbow"]["y"])
        ),
        "left_hip": calculate_angle(
            (keypoints["left_knee"]["x"], keypoints["left_knee"]["y"]),
            (keypoints["left_hip"]["x"], keypoints["left_hip"]["y"]),
            (keypoints["left_shoulder"]["x"], keypoints["left_shoulder"]["y"])
        ),
        "right_hip": calculate_angle(
            (keypoints["right_knee"]["x"], keypoints["right_knee"]["y"]),
            (keypoints["right_hip"]["x"], keypoints["right_hip"]["y"]),
            (keypoints["right_shoulder"]["x"], keypoints["right_shoulder"]["y"])
        ),
        "left_knee": calculate_angle(
            (keypoints["left_hip"]["x"], keypoints["left_hip"]["y"]),
            (keypoints["left_knee"]["x"], keypoints["left_knee"]["y"]),
            (keypoints["left_ankle"]["x"], keypoints["left_ankle"]["y"])
        ),
        "right_knee": calculate_angle(
            (keypoints["right_hip"]["x"], keypoints["right_hip"]["y"]),
            (keypoints["right_knee"]["x"], keypoints["right_knee"]["y"]),
            (keypoints["right_ankle"]["x"], keypoints["right_ankle"]["y"])
        ),
    }
    
    # Calculate biomechanical metrics
    biomechanical_metrics = {
        "posture_score": calculate_posture_score(keypoints),
        "balance_score": calculate_balance_score(keypoints),
        "movement_efficiency": calculate_movement_efficiency(joint_angles),
        "power_potential": calculate_power_potential(joint_angles, keypoints),
    }
    
    return {
        "keypoints": keypoints,
        "joint_angles": joint_angles,
        "biomechanical_metrics": biomechanical_metrics
    }

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def calculate_posture_score(keypoints):
    """Calculate a posture score based on alignment of key points."""
    # Simplified simulation
    return round(random.uniform(70, 95), 1)

def calculate_balance_score(keypoints):
    """Calculate a balance score based on center of mass and support base."""
    # Simplified simulation
    return round(random.uniform(65, 90), 1)

def calculate_movement_efficiency(joint_angles):
    """Calculate movement efficiency based on joint angles."""
    # Simplified simulation
    return round(random.uniform(60, 95), 1)

def calculate_power_potential(joint_angles, keypoints):
    """Calculate power potential based on joint angles and positions."""
    # Simplified simulation
    return round(random.uniform(50, 100), 1)

def draw_biomechanics_on_frame(frame: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """
    Draw biomechanical analysis on a frame.
    
    Args:
        frame: The input frame as a numpy array
        analysis: The biomechanical analysis data
        
    Returns:
        The frame with biomechanical analysis drawn on it
    """
    annotated_frame = frame.copy()
    keypoints = analysis["keypoints"]
    joint_angles = analysis["joint_angles"]
    metrics = analysis["biomechanical_metrics"]
    
    # Define the connections between keypoints for visualization
    connections = [
        ("nose", "left_eye"), ("nose", "right_eye"),
        ("left_eye", "left_ear"), ("right_eye", "right_ear"),
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle")
    ]
    
    # Draw keypoints
    for keypoint_name, keypoint_data in keypoints.items():
        x, y = int(keypoint_data["x"]), int(keypoint_data["y"])
        confidence = keypoint_data["confidence"]
        
        # Only draw keypoints with confidence above threshold
        if confidence > 0.5:
            # Use color based on confidence (green to red)
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.circle(annotated_frame, (x, y), 5, color, -1)
    
    # Draw connections
    for connection in connections:
        start_point_name, end_point_name = connection
        
        if (start_point_name in keypoints and end_point_name in keypoints and
            keypoints[start_point_name]["confidence"] > 0.5 and
            keypoints[end_point_name]["confidence"] > 0.5):
            
            start_x, start_y = int(keypoints[start_point_name]["x"]), int(keypoints[start_point_name]["y"])
            end_x, end_y = int(keypoints[end_point_name]["x"]), int(keypoints[end_point_name]["y"])
            
            # Average confidence for this connection
            avg_confidence = (keypoints[start_point_name]["confidence"] + keypoints[end_point_name]["confidence"]) / 2
            
            # Use color based on confidence (green to yellow)
            color = (0, 255, int(255 * (1 - avg_confidence)))
            thickness = max(1, int(3 * avg_confidence))
            
            cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), color, thickness)
    
    # Draw joint angles
    for joint_name, angle in joint_angles.items():
        if joint_name in keypoints:
            x, y = int(keypoints[joint_name]["x"]), int(keypoints[joint_name]["y"])
            cv2.putText(annotated_frame, f"{angle:.1f}°", (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw biomechanical metrics
    height, width = annotated_frame.shape[:2]
    metrics_y = 30
    for metric_name, value in metrics.items():
        cv2.putText(annotated_frame, f"{metric_name}: {value}", (10, metrics_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        metrics_y += 25
    
    return annotated_frame

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "model": "mmpose"}

async def download_video(url: str) -> str:
    """
    Download a video from a URL and save it to a temporary file.
    
    Args:
        url: The URL of the video to download
        
    Returns:
        The path to the temporary file
    """
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_path = temp_file.name
        temp_file.close()
        
        # Download the video
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                
                with open(temp_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
        
        return temp_path
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")

@app.post("/mmpose")
async def analyze_video(
    request: VideoRequest = Body(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Analyze biomechanics in a video using MMPose.
    
    Args:
        request: The request body containing the video URL and parameters
        file: Optional file upload (for backward compatibility)
        
    Returns:
        If return_both is True, returns both JSON data and video content.
        If return_video is True, returns the annotated video as a StreamingResponse.
        If both are False, returns the biomechanical analysis as JSON.
    """
    try:
        # Get parameters from request
        video_url = request.video_url
        return_video = request.video
        return_both = request.data
        
        # Download the video from the URL
        temp_path = await download_video(str(video_url))
        
        # Get video info
        video_info = get_video_info(temp_path)
        logger.info(f"Processing video: {video_info}")
        
        # Extract frames
        frames = extract_frames(temp_path)
        logger.info(f"Extracted {len(frames)} frames")
        
        # Process each frame
        all_analyses = []
        annotated_frames = []
        
        for i, frame in enumerate(frames):
            # Analyze biomechanics in the frame
            analysis = analyze_biomechanics(frame)
            all_analyses.append(analysis)
            
            # If video or data is True, annotate the frame
            if return_video or return_both:
                annotated_frame = draw_biomechanics_on_frame(frame, analysis)
                annotated_frames.append(annotated_frame)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Prepare the JSON response
        json_response = {"biomechanics": all_analyses}
        
        # If data is True, create the video and return both
        if return_both:
            # Create a video from the annotated frames
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            
            # Get the original video's properties
            height, width = annotated_frames[0].shape[:2]
            fps = video_info["fps"]
            
            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write the frames to the video
            for frame in annotated_frames:
                out.write(frame)
            
            # Release the VideoWriter
            out.release()
            
            # Read the video file
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            
            # Clean up the temporary output file
            os.unlink(output_path)
            
            # Encode video as base64
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            # Return combined response
            return {
                "data": json_response,
                "video_base64": video_base64
            }
        
        # If video is True, return the video as a StreamingResponse
        elif return_video:
            # Create a video from the annotated frames
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            
            # Get the original video's properties
            height, width = annotated_frames[0].shape[:2]
            fps = video_info["fps"]
            
            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write the frames to the video
            for frame in annotated_frames:
                out.write(frame)
            
            # Release the VideoWriter
            out.release()
            
            # Return the video as a StreamingResponse
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            
            # Clean up the temporary output file
            os.unlink(output_path)
            
            return StreamingResponse(
                io.BytesIO(video_bytes),
                media_type="video/mp4"
            )
        else:
            # Return the analyses as JSON
            return json_response
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)