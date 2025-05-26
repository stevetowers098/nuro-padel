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
import torch
import uuid
from datetime import datetime
from google.cloud import storage
from ultralytics import YOLO

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

# Load the YOLOv8 model with optimizations
model = YOLO('yolov8m.pt')  # Load the medium model for better accuracy
# GPU optimization if available
if torch.cuda.is_available():
    model.to('cuda')
    model.fuse()  # Fuse layers for better performance

def track_objects(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect objects in a frame using YOLOv8.
    Specialized for padel-relevant objects: person (0), sports ball (32), tennis racket (43)
    
    Args:
        frame: The input frame as a numpy array
        
    Returns:
        A list of tracked objects, each with a bounding box, class, and tracking ID
    """
    # Run inference with YOLOv8
    results = model(frame, verbose=False)[0]
    
    # Filter for padel-relevant classes
    objects = []
    for i, det in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        
        # Only include padel-relevant classes
        if cls in PADEL_CLASSES:
            objects.append({
                "class": PADEL_CLASSES[cls],
                "confidence": float(conf),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                },
                "track_id": i + 1  # Simple tracking ID
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
        
        # Extract frames with smart sampling
        if data and not video:
            # For data-only requests, sample every 3rd frame to speed up processing
            frames = extract_frames(temp_path, sample_every=3)
            logger.info(f"Extracted {len(frames)} frames (sampled every 3rd frame)")
        else:
            # For video requests, extract all frames
            frames = extract_frames(temp_path)
            logger.info(f"Extracted {len(frames)} frames")
        
        # Process frames in batches for better performance
        all_objects = []
        annotated_frames = []
        
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
                    
                # Extract objects from this frame's results
                frame_objects = []
                for k, det in enumerate(result.boxes.data.tolist()):
                    x1, y1, x2, y2, conf, cls = det
                    cls = int(cls)
                    
                    # Only include padel-relevant classes
                    if cls in PADEL_CLASSES:
                        frame_objects.append({
                            "class": PADEL_CLASSES[cls],
                            "confidence": float(conf),
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2)
                            },
                            "track_id": k + 1  # Simple tracking ID
                        })
                
                all_objects.append(frame_objects)
                
                # If video or data is True, annotate the frame
                if video or data:
                    annotated_frame = draw_objects_on_frame(frames[frame_idx], frame_objects)
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