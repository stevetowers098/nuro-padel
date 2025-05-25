from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import tempfile
import os
import cv2
import numpy as np
import supervision as sv
import httpx
from typing import Dict, Any, List
import io
import sys
import logging
import asyncio

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Combined AI Service", version="1.0.0")

# Service URLs
SERVICES = {
    "yolo11": "http://localhost:8001",
    "yolov8": "http://localhost:8005", 
    "yolo_nas": "http://localhost:8002",
    "mmpose": "http://localhost:8003"
}

async def call_service(client, service_name, endpoint, file_data):
    """
    Call a service with the given file data.
    
    Args:
        client: The httpx client to use
        service_name: The name of the service to call
        endpoint: The endpoint to call
        file_data: The file data to send
        
    Returns:
        The response from the service
    """
    try:
        url = f"{SERVICES[service_name]}/{endpoint}?return_video=false"
        response = await client.post(url, files={"file": file_data})
        return response.json()
    except Exception as e:
        logger.error(f"Error calling {service_name} service: {str(e)}")
        return None

def combine_annotations(frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
    """
    Combine annotations from multiple services on a single frame.
    
    Args:
        frame: The input frame as a numpy array
        results: The results from all services
        
    Returns:
        The frame with combined annotations
    """
    annotated_frame = frame.copy()
    height, width = annotated_frame.shape[:2]
    
    # Draw YOLOv8 object tracking results
    if "objects" in results and results["objects"]:
        objects = results["objects"]
        
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
    
    # Draw YOLO11 pose estimation results
    if "poses" in results and results["poses"]:
        poses = results["poses"]
        
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
    
    # Draw MMPose biomechanical analysis results
    if "biomechanics" in results and results["biomechanics"]:
        analysis = results["biomechanics"]
        
        # Draw biomechanical metrics
        metrics_y = 30
        for metric_name, value in analysis.get("biomechanical_metrics", {}).items():
            cv2.putText(annotated_frame, f"{metric_name}: {value}", (10, metrics_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            metrics_y += 25
        
        # Draw joint angles
        joint_angles = analysis.get("joint_angles", {})
        keypoints = analysis.get("keypoints", {})
        
        for joint_name, angle in joint_angles.items():
            if joint_name in keypoints:
                x, y = int(keypoints[joint_name]["x"]), int(keypoints[joint_name]["y"])
                cv2.putText(annotated_frame, f"{angle:.1f}°", (x + 10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add a combined service label
    cv2.putText(annotated_frame, "Combined Analysis", (width - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated_frame

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "model": "combined"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...), return_video: bool = False):
    """
    Analyze a video using a combination of all services.
    
    Args:
        file: The input video file
        return_video: Whether to return the annotated video (default: False)
        
    Returns:
        If return_video is False, returns the combined analysis as JSON.
        If return_video is True, returns the annotated video as a StreamingResponse.
    """
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            file_content = await file.read()
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        # Get video info
        video_info = get_video_info(temp_path)
        logger.info(f"Processing video: {video_info}")
        
        # Call all services
        async with httpx.AsyncClient(timeout=300.0) as client:
            tasks = [
                call_service(client, "yolo11", "pose", ("video.mp4", file_content, "video/mp4")),
                call_service(client, "yolov8", "track", ("video.mp4", file_content, "video/mp4")),
                call_service(client, "yolo_nas", "pose", ("video.mp4", file_content, "video/mp4")),
                call_service(client, "mmpose", "analyze", ("video.mp4", file_content, "video/mp4"))
            ]
            
            results = await asyncio.gather(*tasks)
        
        # Combine results
        combined_results = {
            "poses": results[0].get("poses", []) if results[0] else [],
            "objects": results[1].get("objects", []) if results[1] else [],
            "high_accuracy_poses": results[2].get("poses", []) if results[2] else [],
            "biomechanics": results[3].get("biomechanics", []) if results[3] else []
        }
        
        # If return_video is True, create an annotated video
        if return_video:
            # Extract frames
            frames = extract_frames(temp_path)
            logger.info(f"Extracted {len(frames)} frames")
            
            # Process each frame
            annotated_frames = []
            
            for i, frame in enumerate(frames):
                # Get results for this frame
                frame_results = {
                    "poses": combined_results["poses"][i] if i < len(combined_results["poses"]) else [],
                    "objects": combined_results["objects"][i] if i < len(combined_results["objects"]) else [],
                    "high_accuracy_poses": combined_results["high_accuracy_poses"][i] if i < len(combined_results["high_accuracy_poses"]) else [],
                    "biomechanics": combined_results["biomechanics"][i] if i < len(combined_results["biomechanics"]) else []
                }
                
                # Combine annotations on the frame
                annotated_frame = combine_annotations(frame, frame_results)
                annotated_frames.append(annotated_frame)
            
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
            # Return the combined results as JSON
            return combined_results
        
        # Clean up the temporary file
        os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)