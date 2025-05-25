from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import tempfile
import os
import cv2
import numpy as np
import supervision as sv
from typing import Dict, Any, List
import io
import sys
import logging
import random

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv8 Tracking Service", version="1.0.0")

# Simulated YOLOv8 object tracking function
# In a real implementation, this would use the actual YOLOv8 model
def track_objects(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Track objects in a frame using YOLOv8.
    
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
    
    # Simulate player detections
    player_classes = ["player", "player"]
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
    
    # Simulate ball detection
    ball_x = width // 2 + random.randint(-width // 8, width // 8)
    ball_y = height // 2 + random.randint(-height // 8, height // 8)
    ball_size = min(width, height) // 20
    
    objects.append({
        "class": "ball",
        "confidence": 0.85,
        "bbox": {
            "x1": ball_x - ball_size // 2,
            "y1": ball_y - ball_size // 2,
            "x2": ball_x + ball_size // 2,
            "y2": ball_y + ball_size // 2
        },
        "track_id": 3
    })
    
    # Simulate racket detection
    racket_x = width // 4 + random.randint(-width // 16, width // 16)
    racket_y = height // 2 + random.randint(-height // 16, height // 16)
    racket_w = width // 10
    racket_h = height // 8
    
    objects.append({
        "class": "racket",
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
        "player": (0, 255, 0),  # Green
        "ball": (0, 0, 255),    # Red
        "racket": (255, 0, 0)   # Blue
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

@app.post("/track")
async def track_video(file: UploadFile = File(...), return_video: bool = False):
    """
    Track objects in a video using YOLOv8.
    
    Args:
        file: The input video file
        return_video: Whether to return the annotated video (default: False)
        
    Returns:
        If return_video is False, returns the tracked objects as JSON.
        If return_video is True, returns the annotated video as a StreamingResponse.
    """
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
        
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
            
            # If return_video is True, annotate the frame
            if return_video:
                annotated_frame = draw_objects_on_frame(frame, objects)
                annotated_frames.append(annotated_frame)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Return the results
        if return_video:
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
            # Return the objects as JSON
            return {"objects": all_objects}
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)