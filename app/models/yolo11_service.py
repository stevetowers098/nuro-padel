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
import base64
import json
import httpx
import uuid
from datetime import datetime
from google.cloud import storage
from pydantic import HttpUrl

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO11 Pose Service", version="1.0.0")

# Google Cloud Storage configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed"

async def upload_to_gcs(video_path: str) -> str:
    """
    Upload a video to Google Cloud Storage.
    
    Args:
        video_path: Path to the video file to upload
        
    Returns:
        The public URL of the uploaded video
    """
    try:
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{GCS_FOLDER}/video_{timestamp}_{unique_id}.mp4"
        
        # Upload the file to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        
        # Upload the file
        blob.upload_from_filename(video_path)
        
        # Make the blob publicly accessible
        blob.make_public()
        
        # Return the public URL
        return blob.public_url
    
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading to GCS: {str(e)}")

# Simulated YOLO11 pose detection function
# In a real implementation, this would use the actual YOLO11 model with yolo11n-pose.pt
def detect_poses(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect poses in a frame using YOLO11.
    Specialized for human pose estimation with 17 keypoints from the COCO keypoints dataset.
    
    Args:
        frame: The input frame as a numpy array
        
    Returns:
        A list of detected poses, each with keypoints and confidence scores
    """
    # Simulate pose detection
    # In a real implementation, this would use the actual YOLO11 model
    height, width = frame.shape[:2]
    
    # Simulate a person detection
    poses = [{
        "keypoints": {
            "nose": {"x": width // 2, "y": height // 3, "confidence": 0.9},
            "left_shoulder": {"x": width // 3, "y": height // 2, "confidence": 0.85},
            "right_shoulder": {"x": 2 * width // 3, "y": height // 2, "confidence": 0.85},
            "left_elbow": {"x": width // 4, "y": 2 * height // 3, "confidence": 0.8},
            "right_elbow": {"x": 3 * width // 4, "y": 2 * height // 3, "confidence": 0.8},
            "left_wrist": {"x": width // 5, "y": 3 * height // 4, "confidence": 0.75},
            "right_wrist": {"x": 4 * width // 5, "y": 3 * height // 4, "confidence": 0.75},
            "left_hip": {"x": 2 * width // 5, "y": 3 * height // 4, "confidence": 0.7},
            "right_hip": {"x": 3 * width // 5, "y": 3 * height // 4, "confidence": 0.7},
            "left_knee": {"x": width // 3, "y": 5 * height // 6, "confidence": 0.65},
            "right_knee": {"x": 2 * width // 3, "y": 5 * height // 6, "confidence": 0.65},
            "left_ankle": {"x": width // 4, "y": 11 * height // 12, "confidence": 0.6},
            "right_ankle": {"x": 3 * width // 4, "y": 11 * height // 12, "confidence": 0.6},
        },
        "confidence": 0.85
    }]
    
    return poses

def draw_poses_on_frame(frame: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw detected poses on a frame.
    
    Args:
        frame: The input frame as a numpy array
        poses: A list of detected poses
        
    Returns:
        The frame with poses drawn on it
    """
    annotated_frame = frame.copy()
    
    # Define the connections between keypoints for visualization
    connections = [
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle")
    ]
    
    for pose in poses:
        keypoints = pose["keypoints"]
        
        # Draw keypoints
        for keypoint_name, keypoint_data in keypoints.items():
            x, y = int(keypoint_data["x"]), int(keypoint_data["y"])
            confidence = keypoint_data["confidence"]
            
            # Only draw keypoints with confidence above threshold
            if confidence > 0.5:
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        for connection in connections:
            start_point_name, end_point_name = connection
            
            if (start_point_name in keypoints and end_point_name in keypoints and
                keypoints[start_point_name]["confidence"] > 0.5 and
                keypoints[end_point_name]["confidence"] > 0.5):
                
                start_x, start_y = int(keypoints[start_point_name]["x"]), int(keypoints[start_point_name]["y"])
                end_x, end_y = int(keypoints[end_point_name]["x"]), int(keypoints[end_point_name]["y"])
                
                cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    
    return annotated_frame

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "model": "yolo11"}

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

@app.post("/yolo11")
async def detect_pose(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = None,
    video: bool = False,
    data: bool = False
):
    """
    Detect poses in a video using YOLO11.
    Specialized for human pose estimation with 17 keypoints from the COCO keypoints dataset.
    
    Args:
        file: Optional file upload
        video_url: Optional URL of the video to analyze
        video: Whether to return the annotated video (default: False)
        data: Whether to return JSON data (default: False)
        
    Returns:
        - If data=True only: Returns JSON analysis data
        - If video=True only: Returns processed video URL
        - If both=True: Returns both JSON data AND video URL
        - If both=False: Returns error (must specify at least one)
    """
    try:
        # Validate that at least one output format is requested
        if not video and not data:
            raise HTTPException(status_code=400, detail="Must specify at least one of 'video' or 'data'")
            
        # Handle video URL input
        if video_url:
            try:
                # Download the video from the URL
                temp_path = await download_video(video_url)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")
        elif file:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
        else:
            raise HTTPException(status_code=400, detail="Either file or video_url is required")
        
        # Get video info
        video_info = get_video_info(temp_path)
        logger.info(f"Processing video: {video_info}")
        
        # Extract frames
        frames = extract_frames(temp_path)
        logger.info(f"Extracted {len(frames)} frames")
        
        # Process each frame
        all_poses = []
        annotated_frames = []
        
        for i, frame in enumerate(frames):
            # Detect poses in the frame
            poses = detect_poses(frame)
            all_poses.append(poses)
            
            # If video or data is True, annotate the frame
            if video or data:
                annotated_frame = draw_poses_on_frame(frame, poses)
                annotated_frames.append(annotated_frame)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Prepare the JSON response
        json_response = {"poses": all_poses}
        
        # If video or data is True, create the video and upload to GCS
        if video or data:
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
            
            # Upload the video to GCS
            video_url = await upload_to_gcs(output_path)
            
            # Clean up the temporary output file
            os.unlink(output_path)
            
            # Return response with video URL
            if data and video:
                # Return both data and video URL
                return {
                    "data": json_response,
                    "video_url": video_url
                }
            elif video:
                # Return just the video URL
                return {
                    "video_url": video_url
                }
        else:
            # Return the poses as JSON
            return json_response
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)