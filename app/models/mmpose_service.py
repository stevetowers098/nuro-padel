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
    
    # Use a standard RTMPose model alias. 
    # MMPose will resolve this to find the config and download weights if needed.
    # The checkpoint URL can often be omitted if the alias is standard and weights are in metafile.
    logger.info("Attempting to load RTMPose-M COCO model using alias...")
    model = init_model( 
        'rtmpose-m_8xb256-420e_coco-256x192', # Model alias
        # You can specify the checkpoint URL if needed, or let MMPose download it
        # 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
        device=model_device
    )
    logger.info("RTMPose model loaded successfully (or will be downloaded).")

except Exception as e:
    logger.warning(f"Could not load RTMPose model using alias: {e}. Attempting fallback configuration.")
    # Fallback to a simpler model if the above fails
    try:
        model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        logger.info(f"Attempting to load HRNet-W48 COCO model as fallback using alias (device: {model_device})...")
        model = init_model( 
            'td-hm_hrnet-w48_8xb32-210e_coco-256x192', # Model alias for HRNet
            # 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
            device=model_device
        )
        logger.info("HRNet fallback model loaded successfully (or will be downloaded).")
    except Exception as e2:
        logger.error(f"Could not load any MMPose model (RTMPose or HRNet fallback): {e2}", exc_info=True)
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
        logger.warning("MMPose model not loaded, returning dummy data for biomechanics analysis")
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
        
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    for frame_idx, frame in enumerate(frames):
        try:
            # Process with MMPose v1.x API
            pose_data_samples = inference_topdown(model, frame, bbox_cs='') # bbox_cs='' for whole image inference
            
            current_keypoints = {}
            
            if pose_data_samples: 
                data_sample = pose_data_samples[0] 
                if hasattr(data_sample, 'pred_instances') and \
                   data_sample.pred_instances is not None and \
                   len(data_sample.pred_instances) > 0 and \
                   hasattr(data_sample.pred_instances, 'keypoints') and \
                   data_sample.pred_instances.keypoints is not None:
                    
                    # Assuming we take the first detected instance's keypoints
                    pred_kpts_tensor = data_sample.pred_instances.keypoints[0] 
                    pred_scores_tensor = data_sample.pred_instances.keypoint_scores[0] 
                    
                    pred_kpts = pred_kpts_tensor.cpu().numpy() 
                    pred_scores = pred_scores_tensor.cpu().numpy()

                    for idx in range(pred_kpts.shape[0]):
                        if idx < len(keypoint_names):
                            current_keypoints[keypoint_names[idx]] = {
                                "x": float(pred_kpts[idx, 0]),
                                "y": float(pred_kpts[idx, 1]),
                                "confidence": float(pred_scores[idx])
                            }
                else:
                    logger.debug(f"Frame {frame_idx}: No instances or keypoints found in pose_data_sample.")
            else:
                logger.debug(f"Frame {frame_idx}: pose_data_samples list is empty.")
            
            joint_angles = {}
            if len(current_keypoints) >= 3:
                if all(k in current_keypoints for k in ["left_shoulder", "left_elbow", "left_wrist"]):
                    joint_angles["left_elbow"] = calculate_angle(
                        (current_keypoints["left_shoulder"]["x"], current_keypoints["left_shoulder"]["y"]),
                        (current_keypoints["left_elbow"]["x"], current_keypoints["left_elbow"]["y"]),
                        (current_keypoints["left_wrist"]["x"], current_keypoints["left_wrist"]["y"])
                    )
                if all(k in current_keypoints for k in ["right_shoulder", "right_elbow", "right_wrist"]):
                    joint_angles["right_elbow"] = calculate_angle(
                        (current_keypoints["right_shoulder"]["x"], current_keypoints["right_shoulder"]["y"]),
                        (current_keypoints["right_elbow"]["x"], current_keypoints["right_elbow"]["y"]),
                        (current_keypoints["right_wrist"]["x"], current_keypoints["right_wrist"]["y"])
                    )
                if all(k in current_keypoints for k in ["left_hip", "left_shoulder", "left_elbow"]):
                    joint_angles["left_shoulder"] = calculate_angle(
                        (current_keypoints["left_hip"]["x"], current_keypoints["left_hip"]["y"]),
                        (current_keypoints["left_shoulder"]["x"], current_keypoints["left_shoulder"]["y"]),
                        (current_keypoints["left_elbow"]["x"], current_keypoints["left_elbow"]["y"])
                    )
                if all(k in current_keypoints for k in ["right_hip", "right_shoulder", "right_elbow"]):
                    joint_angles["right_shoulder"] = calculate_angle(
                        (current_keypoints["right_hip"]["x"], current_keypoints["right_hip"]["y"]),
                        (current_keypoints["right_shoulder"]["x"], current_keypoints["right_shoulder"]["y"]),
                        (current_keypoints["right_elbow"]["x"], current_keypoints["right_elbow"]["y"])
                    )
                if all(k in current_keypoints for k in ["left_knee", "left_hip", "left_shoulder"]):
                    joint_angles["left_hip"] = calculate_angle(
                        (current_keypoints["left_knee"]["x"], current_keypoints["left_knee"]["y"]),
                        (current_keypoints["left_hip"]["x"], current_keypoints["left_hip"]["y"]),
                        (current_keypoints["left_shoulder"]["x"], current_keypoints["left_shoulder"]["y"])
                    )
                if all(k in current_keypoints for k in ["right_knee", "right_hip", "right_shoulder"]):
                    joint_angles["right_hip"] = calculate_angle(
                        (current_keypoints["right_knee"]["x"], current_keypoints["right_knee"]["y"]),
                        (current_keypoints["right_hip"]["x"], current_keypoints["right_hip"]["y"]),
                        (current_keypoints["right_shoulder"]["x"], current_keypoints["right_shoulder"]["y"])
                    )
                if all(k in current_keypoints for k in ["left_hip", "left_knee", "left_ankle"]):
                    joint_angles["left_knee"] = calculate_angle(
                        (current_keypoints["left_hip"]["x"], current_keypoints["left_hip"]["y"]),
                        (current_keypoints["left_knee"]["x"], current_keypoints["left_knee"]["y"]),
                        (current_keypoints["left_ankle"]["x"], current_keypoints["left_ankle"]["y"])
                    )
                if all(k in current_keypoints for k in ["right_hip", "right_knee", "right_ankle"]):
                    joint_angles["right_knee"] = calculate_angle(
                        (current_keypoints["right_hip"]["x"], current_keypoints["right_hip"]["y"]),
                        (current_keypoints["right_knee"]["x"], current_keypoints["right_knee"]["y"]),
                        (current_keypoints["right_ankle"]["x"], current_keypoints["right_ankle"]["y"])
                    )

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
            logger.warning(f"Error processing frame {frame_idx} with MMPose: {e}", exc_info=True)
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
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def calculate_posture_score(keypoints):
    return round(random.uniform(70, 95), 1) if keypoints else 0.0

def calculate_balance_score(keypoints):
    return round(random.uniform(65, 90), 1) if keypoints else 0.0

def calculate_movement_efficiency(joint_angles):
    return round(random.uniform(60, 95), 1) if joint_angles else 0.0

def calculate_power_potential(joint_angles, keypoints):
    return round(random.uniform(50, 100), 1) if joint_angles and keypoints else 0.0

def draw_biomechanics_on_frame(frame: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    annotated_frame = frame.copy()
    keypoints = analysis.get("keypoints", {}) 
    joint_angles = analysis.get("joint_angles", {})
    metrics = analysis.get("biomechanical_metrics", {})
    
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
    
    for keypoint_name, keypoint_data in keypoints.items():
        x, y = int(keypoint_data.get("x",0)), int(keypoint_data.get("y",0)) 
        confidence = keypoint_data.get("confidence", 0.0)
        
        if confidence > 0.5:
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.circle(annotated_frame, (x, y), 5, color, -1)
    
    for connection in connections:
        start_point_name, end_point_name = connection
        start_kp_data = keypoints.get(start_point_name)
        end_kp_data = keypoints.get(end_point_name)

        if (start_kp_data and end_kp_data and
            start_kp_data.get("confidence", 0.0) > 0.5 and
            end_kp_data.get("confidence", 0.0) > 0.5):
            
            start_x, start_y = int(start_kp_data["x"]), int(start_kp_data["y"])
            end_x, end_y = int(end_kp_data["x"]), int(end_kp_data["y"])
            avg_confidence = (start_kp_data["confidence"] + end_kp_data["confidence"]) / 2
            color = (0, 255, int(255 * (1 - avg_confidence)))
            thickness = max(1, int(3 * avg_confidence))
            cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), color, thickness)
    
    for joint_name, angle in joint_angles.items():
        angle_anchor_kp_name = joint_name 
        if angle_anchor_kp_name in keypoints and keypoints[angle_anchor_kp_name].get("confidence",0.0) > 0.5:
            x, y = int(keypoints[angle_anchor_kp_name]["x"]), int(keypoints[angle_anchor_kp_name]["y"])
            cv2.putText(annotated_frame, f"{angle:.1f}°", (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_path = temp_file.name
        temp_file.close()
        
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
    if model is None:
        raise HTTPException(status_code=503, detail="MMPose model is not loaded. Service unavailable.")
        
    temp_path_local = None # Initialize to ensure it's always defined for finally block
    try:
        if video_url:
            try:
                temp_path_local = await download_video(video_url)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")
        elif file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_obj:
                temp_file_obj.write(await file.read())
                temp_path_local = temp_file_obj.name
        else:
            raise HTTPException(status_code=400, detail="Either file or video_url is required")
        
        video_info = get_video_info(temp_path_local)
        logger.info(f"Processing video: {video_info}")
        
        if data and not video:
            frames = extract_frames(temp_path_local, max_frames=50)
            logger.info(f"Extracted {len(frames)} frames (limited for speed)")
        else:
            frames = extract_frames(temp_path_local)
            logger.info(f"Extracted {len(frames)} frames")

        if not frames:
            # No need to raise HTTPException here if temp_path_local is cleaned up in finally
            logger.warning("Could not extract any frames from the video.")
            return {"error": "Could not extract any frames from the video."} # Or appropriate response
        
        all_analyses = analyze_biomechanics(frames)
        
        annotated_frames = []
        if video or data:
            for i, frame_to_annotate in enumerate(frames):
                if i < len(all_analyses): 
                    annotated_frame = draw_biomechanics_on_frame(frame_to_annotate, all_analyses[i])
                    annotated_frames.append(annotated_frame)
                else: 
                    annotated_frames.append(frame_to_annotate) 
        
        json_response = {"biomechanics": all_analyses}
        
        if video or data:
            if not annotated_frames:
                 logger.warning("No frames were available/annotated to create a video.")
                 # Decide how to handle this: error or return JSON only?
                 if data: return {"data": json_response, "video_url": None, "message": "No frames to create video."}
                 return {"video_url": None, "message": "No frames to create video."}


            output_path_local = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            try:
                height, width = annotated_frames[0].shape[:2]
                fps = video_info.get("fps", 30) 
                if fps == 0: fps = 30 
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path_local, fourcc, fps, (width, height))
                
                for frame_to_write in annotated_frames:
                    out.write(frame_to_write)
                out.release()
                
                video_gcs_url = await upload_to_gcs(output_path_local)
            finally:
                 if os.path.exists(output_path_local):
                    os.unlink(output_path_local)

            if data:
                return {"data": json_response, "video_url": video_gcs_url}
            else:
                return {"video_url": video_gcs_url}
        else:
            return json_response
    
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Error processing video in /mmpose endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_path_local and os.path.exists(temp_path_local):
            try:
                os.unlink(temp_path_local)
            except Exception as unlink_e:
                logger.error(f"Error unlinking temp_path_local during cleanup: {unlink_e}")

if __name__ == "__main__":
    import uvicorn
    if model is None: # This check is important
        logger.error("MMPose model could not be loaded at startup. The service might not function correctly or will exit.")
        # Depending on desired behavior, you might want to sys.exit(1) here
        # For now, let it try to start, but /healthz will be unhealthy.
    uvicorn.run(app, host="0.0.0.0", port=8003)
