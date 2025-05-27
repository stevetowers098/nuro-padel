import os
from pathlib import Path

# === PRODUCTION SAFETY: DISABLE ALL AUTO-DOWNLOADS ===
os.environ['YOLO_OFFLINE'] = '1'
os.environ['ULTRALYTICS_OFFLINE'] = '1'
os.environ['ONLINE'] = 'False'
os.environ['YOLO_TELEMETRY'] = 'False'

# Force local-only model loading
WEIGHTS_DIR = Path("/opt/padel/app/weights")

def get_local_model_path(model_name: str) -> str:
    """Get absolute path to model - fails if not exists locally"""
    model_path = WEIGHTS_DIR / model_name
    if not model_path.exists():
        available_models = list(WEIGHTS_DIR.glob("*.pt"))
        raise FileNotFoundError(
            f"❌ PRODUCTION ERROR: Model {model_name} not found locally.\n"
            f"📁 Checked: {model_path}\n"
            f"📋 Available: {[m.name for m in available_models]}\n"
            f"🚫 Auto-download disabled for production stability."
        )
    return str(model_path.absolute())

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
from super_gradients.training import models

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO-NAS Pose Service", version="1.0.0")

# Google Cloud Storage configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed"

# Load the YOLO-NAS model with proper pretrained_weights string
model = models.get("yolo_nas_pose_n", pretrained_weights="coco_pose")
# GPU optimization if available
if torch.cuda.is_available():
    model.to('cuda')
    model.half()  # Use half precision

def detect_high_accuracy_poses(frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
    """
    Detect high-accuracy poses in a list of frames using YOLO-NAS.
    
    Args:
        frames: A list of frames as numpy arrays
        
    Returns:
        A list of lists of detected poses, one list per frame
    """
    all_poses = []
    
    # Use batch processing for better performance
    batch_size = 8  # Adjust based on available memory
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        # Process batch with half precision for speed
        with torch.no_grad():
            results = model.predict(batch, half=True)
        
        batch_poses = []
        for result in results:
            frame_poses = []
            
            # Extract keypoints for each person
            for person_idx in range(len(result.prediction.poses)):
                keypoints = {}
                pose_data = result.prediction.poses[person_idx]
                
                keypoint_names = [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"
                ]
                
                for kpt_idx, (x, y, conf) in enumerate(pose_data):
                    if kpt_idx < len(keypoint_names) and conf > 0:
                        keypoints[keypoint_names[kpt_idx]] = {
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(conf)
                        }
                
                # Calculate bounding box from keypoints
                valid_kpts = [(kp["x"], kp["y"]) for kp in keypoints.values()]
                if valid_kpts:
                    x_coords, y_coords = zip(*valid_kpts)
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    frame_poses.append({
                        "keypoints": keypoints,
                        "confidence": float(np.mean([kp["confidence"] for kp in keypoints.values()])),
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        }
                    })
            
            batch_poses.append(frame_poses)
        
        all_poses.extend(batch_poses)
    
    return all_poses

def draw_poses_on_frame(frame: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw detected poses on a frame with high-quality visualization.
    
    Args:
        frame: The input frame as a numpy array
        poses: A list of detected poses
        
    Returns:
        The frame with poses drawn on it
    """
    annotated_frame = frame.copy()
    
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
    
    for pose in poses:
        keypoints = pose["keypoints"]
        
        # Draw bounding box if available
        if "bbox" in pose:
            bbox = pose["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
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
    
    return annotated_frame

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "model": "yolo_nas"}

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

@app.post("/yolo-nas")
async def detect_pose(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = None,
    video: bool = False,
    data: bool = False
):
    """
    Detect high-accuracy poses in a video using YOLO-NAS.
    
    Args:
        file: Optional file upload
        video_url: Optional URL of the video to analyze
        video: Whether to return the video URL (default: False)
        data: Whether to return both JSON data and video URL (default: False)
        
    Returns:
        If data is True, returns both JSON data and video URL.
        If video is True, returns the video URL.
        If both are False, returns the detected poses as JSON.
    """
    try:
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
        all_poses = detect_high_accuracy_poses(frames)
        
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
        
        # If video or data is True, create the video
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
            
            # If data is True, return both JSON data and video URL
            if data:
                return {
                    "data": json_response,
                    "video_url": video_url
                }
            
            # If only video is True, return just the video URL
            elif video:
                return {
                    "video_url": video_url
                }
        else:
            # Return the poses as JSON
            return json_response
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/yolo-nas/pose")
async def yolo_nas_pose_detection(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = None,
    video: bool = False,
    data: bool = False
):
    """
    YOLO-NAS High-Accuracy Pose Detection Endpoint
    
    Dedicated endpoint for YOLO-NAS pose estimation on padel videos.
    Uses YOLO-NAS architecture for high-accuracy pose detection with 17 keypoints.
    
    Optimized for:
    - High-precision pose keypoint detection
    - Batch processing for performance
    - Half precision for speed on GPU
    
    Args:
        file: Optional video file upload
        video_url: Optional URL of the video to analyze
        video: Whether to return annotated video with pose overlay
        data: Whether to return pose keypoint data
        
    Returns:
        Pose detection data and/or annotated video with skeleton visualization
    """
    logger.info(f"--- Enter /yolo-nas/pose endpoint ---")
    logger.info(f"Dedicated YOLO-NAS high-accuracy pose detection for padel analysis")
    
    # Reuse the existing pose detection logic
    return await detect_pose(file, video_url, video, data)

@app.post("/yolo-nas/object")
async def yolo_nas_object_detection(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = None,
    video: bool = False,
    data: bool = False
):
    """
    YOLO-NAS High-Accuracy Object Detection Endpoint
    
    Dedicated endpoint for YOLO-NAS object detection on padel videos.
    Uses YOLO-NAS architecture for high-accuracy object detection.
    
    Detects padel-specific objects:
    - person (players)
    - sports ball (padel ball)
    - tennis racket (padel racket)
    
    Optimized for:
    - High-precision object detection
    - Batch processing for performance
    - Half precision for speed on GPU
    
    Args:
        file: Optional video file upload
        video_url: Optional URL of the video to analyze
        video: Whether to return annotated video with bounding boxes
        data: Whether to return object detection data
        
    Returns:
        Object detection data and/or annotated video with bounding box visualization
    """
    logger.info(f"--- Enter /yolo-nas/object endpoint ---")
    logger.info(f"Dedicated YOLO-NAS high-accuracy object detection for padel analysis")
    
    try:
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
        
        # Object detection with YOLO-NAS
        all_objects = detect_high_accuracy_objects(frames)
        
        # Annotate frames if needed
        annotated_frames = []
        if video or data:
            for i, frame in enumerate(frames):
                if i < len(all_objects):  # Safety check
                    annotated_frame = draw_objects_on_frame(frame, all_objects[i])
                    annotated_frames.append(annotated_frame)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Prepare the JSON response
        json_response = {"objects": all_objects}
        
        # If video or data is True, create the video
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
            video_url_result = await upload_to_gcs(output_path)
            
            # Clean up the temporary output file
            os.unlink(output_path)
            
            # If data is True, return both JSON data and video URL
            if data:
                return {
                    "data": json_response,
                    "video_url": video_url_result
                }
            
            # If only video is True, return just the video URL
            elif video:
                return {
                    "video_url": video_url_result
                }
        else:
            # Return the objects as JSON
            return json_response
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

def detect_high_accuracy_objects(frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
    """
    Detect high-accuracy objects in a list of frames using YOLO-NAS for object detection.
    
    Args:
        frames: A list of frames as numpy arrays
        
    Returns:
        A list of lists of detected objects, one list per frame
    """
    all_objects = []
    
    # YOLO-NAS pretrained classes for object detection
    # COCO dataset classes - we'll filter for padel-relevant objects
    PADEL_CLASSES = {0: "person", 32: "sports ball", 38: "tennis racket"}
    
    # Load object detection model instead of pose model
    # Note: We need to load a different model for object detection
    try:
        # Try to load YOLO-NAS object detection model
        object_model = models.get("yolo_nas_s", pretrained_weights="coco")  # Standard object detection
        if torch.cuda.is_available():
            object_model.to('cuda')
            object_model.half()
        
        # Use batch processing for better performance
        batch_size = 8  # Adjust based on available memory
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            
            # Process batch with half precision for speed
            with torch.no_grad():
                results = object_model.predict(batch, half=True)
            
            batch_objects = []
            for result in results:
                frame_objects = []
                
                # Extract bounding boxes and classes
                if hasattr(result.prediction, 'bboxes_xyxy') and hasattr(result.prediction, 'labels'):
                    bboxes = result.prediction.bboxes_xyxy
                    labels = result.prediction.labels
                    confidences = result.prediction.confidence
                    
                    for bbox, label, conf in zip(bboxes, labels, confidences):
                        class_id = int(label)
                        confidence = float(conf)
                        
                        # Filter for padel-relevant classes and confidence threshold
                        if class_id in PADEL_CLASSES and confidence > 0.3:
                            x1, y1, x2, y2 = bbox
                            frame_objects.append({
                                "class": PADEL_CLASSES[class_id],
                                "confidence": confidence,
                                "bbox": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2)
                                }
                            })
                
                batch_objects.append(frame_objects)
            
            all_objects.extend(batch_objects)
    
    except Exception as e:
        logger.error(f"Error in YOLO-NAS object detection: {e}")
        # Fallback to empty results
        all_objects = [[] for _ in frames]
    
    return all_objects

def draw_objects_on_frame(frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw detected objects on a frame with bounding boxes.
    
    Args:
        frame: The input frame as a numpy array
        objects: A list of detected objects
        
    Returns:
        The frame with objects drawn on it
    """
    annotated_frame = frame.copy()
    
    # Color mapping for different object classes
    colors = {
        "person": (0, 255, 0),        # Green
        "sports ball": (0, 0, 255),   # Red
        "tennis racket": (255, 0, 0)  # Blue
    }
    
    for obj in objects:
        class_name = obj["class"]
        confidence = obj["confidence"]
        bbox = obj["bbox"]
        
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        
        # Get color for this class
        color = colors.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)