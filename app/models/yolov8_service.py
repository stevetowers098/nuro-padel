from fastapi import FastAPI, UploadFile, File, HTTPException
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
import random
import base64
import json
import httpx
import uuid
from datetime import datetime
from google.cloud import storage

# Add parent directory to path to import utils
# from init.path_initializer import *
from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv8 Object Detection Service", version="1.0.0")

# YOLOv8 should focus on these COCO classes for padel
PADEL_CLASSES = {
    0: "person",        # Players
    32: "sports ball",  # Padel ball  
    43: "tennis racket" # Padel racket (close enough)
}

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

# Simulated YOLOv8 object detection function
# In a real implementation, this would use the actual YOLOv8 model with yolov8n.pt
def track_objects(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect objects in a frame using YOLOv8.
    Specialized for padel-relevant objects: person (0), sports ball (32), tennis racket (43)
    
    Args:
        frame: The input frame as a numpy array
        
    Returns:
        A list of tracked objects, each with a bounding box, class, and tracking ID
    """
    # Simulate object tracking
    # In a real implementation, this would use the actual YOLOv8 model
    height, width = frame.shape[:2]
    
    # Simulate some object detections (players, ball, racket)
    objects = []
    
    # Simulate player detections (COCO class 0: person)
    player_classes = [PADEL_CLASSES[0], PADEL_CLASSES[0]]
    player_positions = [
        (width // 4, height // 2, width // 6, height // 3),  # Player 1
        (3 * width // 4, height // 2, width // 6, height // 3)  # Player 2
    ]
    
    for i, (player_class, (x, y, w, h)) in enumerate(zip(player_classes, player_positions)):
        objects.append({
            "class": player_class,
            "confidence": 0.9,
            "bbox": {
                "x1": x - w // 2,
                "y1": y - h // 2,
                "x2": x + w // 2,
                "y2": y + h // 2
            },
            "track_id": i + 1
        })
    
    # Simulate ball detection (COCO class 32: sports ball)
    ball_x = width // 2 + random.randint(-width // 8, width // 8)
    ball_y = height // 2 + random.randint(-height // 8, height // 8)
    ball_size = min(width, height) // 20
    
    objects.append({
        "class": PADEL_CLASSES[32],
        "confidence": 0.85,
        "bbox": {
            "x1": ball_x - ball_size // 2,
            "y1": ball_y - ball_size // 2,
            "x2": ball_x + ball_size // 2,
            "y2": ball_y + ball_size // 2
        },
        "track_id": 3
    })
    
    # Simulate racket detection (COCO class 43: tennis racket)
    racket_x = width // 4 + random.randint(-width // 16, width // 16)
    racket_y = height // 2 + random.randint(-height // 16, height // 16)
    racket_w = width // 10
    racket_h = height // 8
    
    objects.append({
        "class": PADEL_CLASSES[43],
        "confidence": 0.8,
        "bbox": {
            "x1": racket_x - racket_w // 2,
            "y1": racket_y - racket_h // 2,
            "x2": racket_x + racket_w // 2,
            "y2": racket_y + racket_h // 2
        },
        "track_id": 4
    })
    
    return objects

def draw_objects_on_frame(frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw tracked objects on a frame.
    
    Args:
        frame: The input frame as a numpy array
        objects: A list of tracked objects
        
    Returns:
        The frame with objects drawn on it
    """
    annotated_frame = frame.copy()
    
    # Define colors for different classes
    colors = {
        PADEL_CLASSES[0]: (0, 255, 0),   # Green for person
        PADEL_CLASSES[32]: (0, 0, 255),  # Red for sports ball
        PADEL_CLASSES[43]: (255, 0, 0)   # Blue for tennis racket
    }
    
    for obj in objects:
        # Get bounding box coordinates
        x1, y1 = int(obj["bbox"]["x1"]), int(obj["bbox"]["y1"])
        x2, y2 = int(obj["bbox"]["x2"]), int(obj["bbox"]["y2"])
        
        # Get class and track ID
        class_name = obj["class"]
        track_id = obj["track_id"]
        confidence = obj["confidence"]
        
        # Get color for this class
        color = colors.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} {track_id} ({confidence:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "model": "yolov8"}

@app.post("/yolov8")
async def detect_objects(
    file: Optional[UploadFile] = File(None), 
    video_url: Optional[str] = None,
    video: bool = False, 
    data: bool = False
):
    """
    Detect objects in a video using YOLOv8.
    Specialized for padel-relevant objects: person (0), sports ball (32), tennis racket (43)
    
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
        # Handle video URL input
        if video_url:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(video_url)
                    response.raise_for_status()
                    
                # Save downloaded content to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")
        
        elif file:
            # Existing file upload logic
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
        all_objects = []
        annotated_frames = []
        
        for i, frame in enumerate(frames):
            # Track objects in the frame
            objects = track_objects(frame)
            all_objects.append(objects)
            
            # If video or data is True, annotate the frame
            if video or data:
                annotated_frame = draw_objects_on_frame(frame, objects)
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
            # Return the objects as JSON
            return json_response
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)