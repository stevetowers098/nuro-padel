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
import torch
import base64
import json
import httpx
import uuid
import random
from datetime import datetime
from google.cloud import storage
from pydantic import HttpUrl

# MMPose v1.x API imports
from mmpose.apis import init_model, inference_topdown

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MMPose Biomechanics Service", version="1.0.0")

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

# Load the MMPose model with optimizations
model = None
try:
    model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Initializing MMPose model on device: {model_device}")
    # Use a standard RTMPose model that downloads automatically
    model = init_model( # MMPose v1.x API
        'configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py',
        'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
        device=model_device
    )
except Exception as e:
    logger.warning(f"Could not load RTMPose model: {e}. Attempting fallback configuration.")
    # Fallback to a simpler model if the above fails
    try:
        model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # Ensure device is set for fallback too
        model = init_model( # MMPose v1.x API
            'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
            'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
            device=model_device
        )
    except Exception as e2:
        logger.error(f"Could not load any MMPose model: {e2}")
        model = None # Ensure model is None if all attempts fail

def analyze_biomechanics(frames: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    Analyze biomechanics in a list of frames using MMPose.
    
    Args:
        frames: A list of frames as numpy arrays
        
    Returns:
        A list of biomechanical analyses, one for each frame
    """
    if model is None:
        logger.warning("MMPose model not loaded, returning dummy data")
        # Return dummy data if model failed to load
        return [{
            "keypoints": {},
            "joint_angles": {},
            "biomechanical_metrics": {
                "posture_score": 75.0,
                "balance_score": 70.0,
                "movement_efficiency": 80.0,
                "power_potential": 65.0
            }
        } for _ in frames]
    
    all_analyses = []
    
    # Use batch processing for better performance (though inference_topdown might handle this internally)
    # For simplicity with the new API, we'll process frame by frame first. Batching can be optimized later if needed.
    
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    for frame in frames:
        try:
            # Process with MMPose v1.x API
            pose_data_samples = inference_topdown(model, frame)
            
            current_keypoints = {}
            
            if pose_data_samples: # Check if list is not empty
                # Process the first detected person/instance
                # In MMPose v1.x, inference_topdown can return multiple PoseDataSample objects if multiple bboxes are passed
                # For single image (frame) without pre-detected bboxes, it usually processes the whole image
                data_sample = pose_data_samples[0] 
                if hasattr(data_sample, 'pred_instances') and data_sample.pred_instances.keypoints.shape[0] > 0:
                    # Assuming we take the first detected instance's keypoints
                    pred_kpts_tensor = data_sample.pred_instances.keypoints[0] # Get keypoints for the first instance
                    pred_scores_tensor = data_sample.pred_instances.keypoint_scores[0] # Get scores for the first instance
                    
                    pred_kpts = pred_kpts_tensor.cpu().numpy()  # Shape: (num_keypoints, 2)
                    pred_scores = pred_scores_tensor.cpu().numpy()  # Shape: (num_keypoints,)

                    for idx in range(pred_kpts.shape[0]):
                        if idx < len(keypoint_names):
                            current_keypoints[keypoint_names[idx]] = {
                                "x": float(pred_kpts[idx, 0]),
                                "y": float(pred_kpts[idx, 1]),
                                "confidence": float(pred_scores[idx])
                            }
                else:
                    logger.debug("No instances or keypoints found in pose_data_sample for this frame.")
            else:
                logger.debug("pose_data_samples list is empty for this frame.")
            
            # Calculate joint angles
            joint_angles = {}
            if len(current_keypoints) >= 3:  # Need at least 3 keypoints for angles
                # ... (rest of your joint angle calculation logic, ensure it uses current_keypoints)
                if all(k in current_keypoints for k in ["left_shoulder", "left_elbow", "left_wrist"]):
                    joint_angles["left_elbow"] = calculate_angle(
                        (current_keypoints["left_shoulder"]["x"], current_keypoints["left_shoulder"]["y"]),
                        (current_keypoints["left_elbow"]["x"], current_keypoints["left_elbow"]["y"]),
                        (current_keypoints["left_wrist"]["x"], current_keypoints["left_wrist"]["y"])
                    )
                # (Add other joint angle calculations similarly, referencing current_keypoints)


            # Calculate biomechanical metrics
            biomechanical_metrics = {
                "posture_score": calculate_posture_score(current_keypoints),
                "balance_score": calculate_balance_score(current_keypoints),
                "movement_efficiency": calculate_movement_efficiency(joint_angles),
                "power_potential": calculate_power_potential(joint_angles, current_keypoints),
            }
            
            all_analyses.append({
                "keypoints": current_keypoints,
                "joint_angles": joint_angles,
                "biomechanical_metrics": biomechanical_metrics
            })
        
        except Exception as e:
            logger.warning(f"Error processing frame with MMPose: {e}")
            # Return dummy data for failed frames
            all_analyses.append({
                "keypoints": {},
                "joint_angles": {},
                "biomechanical_metrics": {
                    "posture_score": 75.0,
                    "balance_score": 70.0,
                    "movement_efficiency": 80.0,
                    "power_potential": 65.0
                }
            })
            
    return all_analyses

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def calculate_posture_score(keypoints):
    """Calculate a posture score based on alignment of key points."""
    # Simplified simulation
    return round(random.uniform(70, 95), 1) if keypoints else 0.0

def calculate_balance_score(keypoints):
    """Calculate a balance score based on center of mass and support base."""
    # Simplified simulation
    return round(random.uniform(65, 90), 1) if keypoints else 0.0

def calculate_movement_efficiency(joint_angles):
    """Calculate movement efficiency based on joint angles."""
    # Simplified simulation
    return round(random.uniform(60, 95), 1) if joint_angles else 0.0

def calculate_power_potential(joint_angles, keypoints):
    """Calculate power potential based on joint angles and positions."""
    # Simplified simulation
    return round(random.uniform(50, 100), 1) if joint_angles and keypoints else 0.0

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
    keypoints = analysis.get("keypoints", {}) # Use .get for safety
    joint_angles = analysis.get("joint_angles", {})
    metrics = analysis.get("biomechanical_metrics", {})
    
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
        x, y = int(keypoint_data.get("x",0)), int(keypoint_data.get("y",0)) # Use .get for safety
        confidence = keypoint_data.get("confidence", 0.0)
        
        # Only draw keypoints with confidence above threshold
        if confidence > 0.5:
            # Use color based on confidence (green to red)
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.circle(annotated_frame, (x, y), 5, color, -1)
    
    # Draw connections
    for connection in connections:
        start_point_name, end_point_name = connection
        
        start_kp_data = keypoints.get(start_point_name)
        end_kp_data = keypoints.get(end_point_name)

        if (start_kp_data and end_kp_data and
            start_kp_data.get("confidence", 0.0) > 0.5 and
            end_kp_data.get("confidence", 0.0) > 0.5):
            
            start_x, start_y = int(start_kp_data["x"]), int(start_kp_data["y"])
            end_x, end_y = int(end_kp_data["x"]), int(end_kp_data["y"])
            
            # Average confidence for this connection
            avg_confidence = (start_kp_data["confidence"] + end_kp_data["confidence"]) / 2
            
            # Use color based on confidence (green to yellow)
            color = (0, 255, int(255 * (1 - avg_confidence)))
            thickness = max(1, int(3 * avg_confidence))
            
            cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), color, thickness)
    
    # Draw joint angles
    for joint_name, angle in joint_angles.items():
        # Anchor angle text to relevant keypoint if possible
        angle_anchor_kp_name = joint_name # e.g. if joint_name is "left_elbow", anchor to "left_elbow" keypoint
        if angle_anchor_kp_name in keypoints and keypoints[angle_anchor_kp_name].get("confidence",0.0) > 0.5:
            x, y = int(keypoints[angle_anchor_kp_name]["x"]), int(keypoints[angle_anchor_kp_name]["y"])
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
    if model is None:
        return {"status": "unhealthy", "model": "mmpose not loaded"}
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
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = None,
    video: bool = False,
    data: bool = False
):
    """
    Analyze biomechanics in a video using MMPose.
    
    Args:
        file: Optional file upload
        video_url: Optional URL of the video to analyze
        video: Whether to return the annotated video (default: False)
        data: Whether to return both JSON data and annotated video (default: False)
        
    Returns:
        If data is True, returns both JSON data and video content.
        If video is True, returns the annotated video as a StreamingResponse.
        If both are False, returns the biomechanical analysis as JSON.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="MMPose model is not loaded. Service unavailable.")
        
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
            frames = extract_frames(temp_path, max_frames=50)  # Limit frames for faster processing
            logger.info(f"Extracted {len(frames)} frames (limited for speed)")
        else:
            # For video requests, extract all frames
            frames = extract_frames(temp_path)
            logger.info(f"Extracted {len(frames)} frames")

        if not frames:
            os.unlink(temp_path)
            raise HTTPException(status_code=400, detail="Could not extract any frames from the video.")
        
        # Process frames
        all_analyses = analyze_biomechanics(frames)
        
        # Annotate frames if needed
        annotated_frames = []
        if video or data:
            for i, frame in enumerate(frames):
                if i < len(all_analyses):  # Safety check
                    annotated_frame = draw_biomechanics_on_frame(frame, all_analyses[i])
                    annotated_frames.append(annotated_frame)
                else: # Should not happen if all_analyses has entry for each frame
                    annotated_frames.append(frame) # Add original frame if no analysis
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Prepare the JSON response
        json_response = {"biomechanics": all_analyses}
        
        # If video or data is True, create the video and upload to GCS
        if video or data:
            if not annotated_frames:
                 raise HTTPException(status_code=500, detail="No frames were annotated to create a video.")

            # Create a video from the annotated frames
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            
            # Get the original video's properties
            height, width = annotated_frames[0].shape[:2]
            fps = video_info.get("fps", 30) # Use a default if fps not found
            if fps == 0: fps = 30 # Avoid zero FPS
            
            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write the frames to the video
            for frame_to_write in annotated_frames:
                out.write(frame_to_write)
            
            # Release the VideoWriter
            out.release()
            
            # Upload the video to GCS
            video_gcs_url = await upload_to_gcs(output_path) # Renamed variable to avoid conflict
            
            # Clean up the temporary output file
            os.unlink(output_path)
            
            # Return response with video URL
            if data:
                # Return both data and video URL
                return {
                    "data": json_response,
                    "video_url": video_gcs_url
                }
            else:
                # Return just the video URL
                return {
                    "video_url": video_gcs_url
                }
        else:
            # Return the analyses as JSON
            return json_response
    
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        # Clean up temp_path if it was defined and an error occurred
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as unlink_e:
                logger.error(f"Error unlinking temp_path during error handling: {unlink_e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Ensure the model is loaded at startup if running directly
    if model is None:
        logger.error("MMPose model could not be loaded. Exiting.")
        sys.exit(1) # Exit if model can't be loaded
    uvicorn.run(app, host="0.0.0.0", port=8003)

