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

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO-NAS Pose Service", version="1.0.0")

# Simulated YOLO-NAS pose detection function
# In a real implementation, this would use the actual YOLO-NAS model
def detect_high_accuracy_poses(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect high-accuracy poses in a frame using YOLO-NAS.
    
    Args:
        frame: The input frame as a numpy array
        
    Returns:
        A list of detected poses, each with keypoints and confidence scores
    """
    # Simulate pose detection
    # In a real implementation, this would use the actual YOLO-NAS model
    height, width = frame.shape[:2]
    
    # Simulate a person detection with more keypoints and higher accuracy
    poses = [{
        "keypoints": {
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
        },
        "confidence": 0.92,
        "bbox": {
            "x1": width // 4,
            "y1": height // 6,
            "x2": 3 * width // 4,
            "y2": 11 * height // 12
        }
    }]
    
    return poses

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

@app.post("/pose")
async def detect_pose(file: UploadFile = File(...), return_video: bool = False):
    """
    Detect high-accuracy poses in a video using YOLO-NAS.
    
    Args:
        file: The input video file
        return_video: Whether to return the annotated video (default: False)
        
    Returns:
        If return_video is False, returns the detected poses as JSON.
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
        all_poses = []
        annotated_frames = []
        
        for i, frame in enumerate(frames):
            # Detect poses in the frame
            poses = detect_high_accuracy_poses(frame)
            all_poses.append(poses)
            
            # If return_video is True, annotate the frame
            if return_video:
                annotated_frame = draw_poses_on_frame(frame, poses)
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
            # Return the poses as JSON
            return {"poses": all_poses}
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)