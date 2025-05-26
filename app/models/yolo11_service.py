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
import torch
import base64
import json
import httpx
import uuid
from datetime import datetime
from google.cloud import storage
from pydantic import HttpUrl
from ultralytics import YOLO

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

# Load the YOLO11 pose model with optimizations
model = YOLO('yolo11m-pose.pt')  # Load the medium model for better accuracy
# GPU optimization if available
if torch.cuda.is_available():
    model.to('cuda')
    model.fuse()  # Fuse layers for better performance

def detect_poses(frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
    """
    Detect human poses in a list of frames using YOLO11 pose model.
    Specialized for human pose estimation with 17 keypoints from the COCO keypoints dataset.
    
    Args:
        frames: A list of frames as numpy arrays
        
    Returns:
        A list of lists of poses, where each pose has keypoints and a bounding box
    """
    all_poses = []
    
    # Use batch processing for better performance
    batch_size = 8  # Adjust based on available memory
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        # Process batch with half precision for speed
        results = model(batch, verbose=False, half=True)
        
        for j, result in enumerate(results):
            frame_idx = i + j
            if frame_idx >= len(frames):
                break
                
            # Extract poses from this frame's results
            frame_poses = []
            
            # Process keypoints for each person
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                for k, keypoints in enumerate(result.keypoints.data):
                    # Get bounding box from keypoints
                    kpts = keypoints.cpu().numpy()
                    valid_kpts = kpts[kpts[:, 2] > 0]
                    if len(valid_kpts) == 0:
                        continue
                        
                    x1, y1 = valid_kpts[:, 0].min(), valid_kpts[:, 1].min()
                    x2, y2 = valid_kpts[:, 0].max(), valid_kpts[:, 1].max()
                    
                    # Format keypoints
                    keypoint_list = {}
                    keypoint_names = [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ]
                    
                    for kpt_idx, kpt in enumerate(kpts):
                        x, y, conf = kpt
                        if conf > 0 and kpt_idx < len(keypoint_names):  # Only include valid keypoints
                            keypoint_list[keypoint_names[kpt_idx]] = {
                                "x": float(x),
                                "y": float(y),
                                "confidence": float(conf)
                            }
                    
                    frame_poses.append({
                        "keypoints": keypoint_list,
                        "confidence": float(valid_kpts[:, 2].mean()),
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        }
                    })
            
            all_poses.append(frame_poses)
    
    return all_poses

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
        
        # Extract frames with smart sampling
        if data and not video:
            # For data-only requests, sample every 3rd frame to speed up processing
            frames = extract_frames(temp_path, sample_every=3)
            logger.info(f"Extracted {len(frames)} frames (sampled every 3rd frame)")
        else:
            # For video requests, extract all frames
            frames = extract_frames(temp_path)
            logger.info(f"Extracted {len(frames)} frames")
        
        # Process frames in batches
        all_poses = detect_poses(frames)
        
        # Annotate frames if needed
        annotated_frames = []
        if video or data:
            for i, frame in enumerate(frames):
                if i < len(all_poses):  # Safety check
                    annotated_frame = draw_poses_on_frame(frame, all_poses[i])
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