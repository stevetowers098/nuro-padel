# -*- coding: utf-8 -*-
import os
from pathlib import Path

# === PRODUCTION SAFETY: DISABLE ALL AUTO-DOWNLOADS ===
os.environ['YOLO_OFFLINE'] = '1'
os.environ['ULTRALYTICS_OFFLINE'] = '1'
os.environ['ONLINE'] = 'False'
os.environ['YOLO_TELEMETRY'] = 'False'

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import sys
import logging
import httpx
import torch
import uuid
import uvicorn
from datetime import datetime
from google.cloud import storage
from ultralytics import YOLO
import subprocess

# Setup for utils.video_utils
try:
    from utils.video_utils import get_video_info, extract_frames
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- YOLO COMBINED SERVICE (YOLO11 + YOLOv8) STARTED ---")

app = FastAPI(title="YOLO Combined Service - YOLO11 + YOLOv8", version="1.0.0")
logger.info("FastAPI app created for YOLO Combined service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
WEIGHTS_DIR = "/app/weights"
YOLO11_POSE_MODEL = "yolo11n-pose.pt"
YOLOV8_OBJECT_MODEL = "yolov8m.pt"
YOLOV8_POSE_MODEL = "yolov8n-pose.pt"

# Pydantic Models
class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False
    data: bool = True
    confidence: float = 0.3

# Helper Functions
async def upload_to_gcs(video_path: str, folder: str, object_name: Optional[str] = None) -> str:
    if not object_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        object_name = f"{folder}/video_{timestamp}_{unique_id}.mp4"
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(video_path)
        blob.make_public()
        logger.info(f"Successfully uploaded {video_path} to GCS as {object_name}")
        return blob.public_url
    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}", exc_info=True)
        return ""

def load_model(model_name: str, description: str) -> Optional[YOLO]:
    """Load YOLO model with error handling"""
    model_path = os.path.join(WEIGHTS_DIR, model_name)
    try:
        logger.info(f"Loading {description} from: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} does not exist")
            return None
        
        model = YOLO(model_path)
        logger.info(f"{description} loaded successfully")
        
        if torch.cuda.is_available():
            logger.info(f"Moving {description} to CUDA")
            model.to('cuda')
            try:
                model.fuse()
                logger.info(f"{description} on CUDA and fused")
            except Exception:
                logger.warning(f"Could not fuse {description}, continuing without fusion")
        return model
    except Exception as e:
        logger.error(f"Failed to load {description}: {e}", exc_info=True)
        return None

# Load Models
yolo11_pose_model = load_model(YOLO11_POSE_MODEL, "YOLO11 Pose Model")
yolov8_object_model = load_model(YOLOV8_OBJECT_MODEL, "YOLOv8 Object Model")  
yolov8_pose_model = load_model(YOLOV8_POSE_MODEL, "YOLOv8 Pose Model")

# Drawing Functions
def draw_poses_on_frame(frame: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
    annotated_frame = frame.copy()
    connections = [
        ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"),
        ("right_eye", "right_ear"), ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle")
    ]
    
    for pose in poses:
        keypoints = pose.get("keypoints", {})
        bbox = pose.get("bbox", {})
        
        # Draw bounding box
        if bbox and pose.get("confidence", 0) > 0.3:
            cv2.rectangle(annotated_frame, (int(bbox["x1"]), int(bbox["y1"])),
                         (int(bbox["x2"]), int(bbox["y2"])), (255, 0, 0), 2)
        
        # Draw keypoints
        for name, data in keypoints.items():
            x, y, conf = int(data.get("x", 0)), int(data.get("y", 0)), data.get("confidence", 0.0)
            if conf > 0.5:
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw connections
        for p1_name, p2_name in connections:
            p1, p2 = keypoints.get(p1_name), keypoints.get(p2_name)
            if p1 and p2 and p1.get("confidence", 0) > 0.5 and p2.get("confidence", 0) > 0.5:
                cv2.line(annotated_frame, (int(p1["x"]), int(p1["y"])),
                        (int(p2["x"]), int(p2["y"])), (0, 255, 255), 2)
    return annotated_frame

def draw_objects_on_frame(frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    annotated_frame = frame.copy()
    colors = {"person": (0, 255, 0), "sports ball": (0, 0, 255), "tennis racket": (255, 0, 0)}
    
    for obj in objects:
        bbox = obj["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        class_name, conf = obj["class"], obj["confidence"]
        color = colors.get(class_name, (255, 255, 255))
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"{class_name} ({conf:.2f})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_frame

async def create_video_from_frames(frames: List[np.ndarray], video_info: dict, 
                                  folder: str) -> Optional[str]:
    """Create video using FFMPEG and upload to GCS"""
    if not frames:
        return None
    
    output_video_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            output_video_path = temp_file.name
        
        height, width = frames[0].shape[:2]
        fps = float(video_info.get("fps", 30.0))
        if fps <= 0: fps = 30.0
        
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-vcodec', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', output_video_path
        ]
        
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        for frame in frames:
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.write(frame.tobytes())
                except (IOError, BrokenPipeError):
                    break
        
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        
        try:
            stdout, stderr = process.communicate(timeout=120)
            if process.returncode == 0:
                logger.info("Video created successfully")
                return await upload_to_gcs(output_video_path, folder)
            else:
                logger.error(f"FFMPEG failed with code {process.returncode}")
                return None
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("FFMPEG timed out")
            return None
    finally:
        if output_video_path and os.path.exists(output_video_path):
            os.unlink(output_video_path)

# API Endpoints
@app.get("/healthz")
async def health_check():
    models_status = {
        "yolo11_pose": yolo11_pose_model is not None,
        "yolov8_object": yolov8_object_model is not None,
        "yolov8_pose": yolov8_pose_model is not None
    }
    
    if not any(models_status.values()):
        return JSONResponse(content={"status": "unhealthy", "models": models_status}, status_code=503)
    
    return {"status": "healthy", "models": models_status}

@app.post("/yolo11/pose")
async def yolo11_pose_detection(payload: VideoAnalysisURLRequest, request: Request):
    """YOLO11 Pose Detection - 17 keypoint human pose estimation"""
    logger.info("YOLO11 Pose detection request received")
    
    if yolo11_pose_model is None:
        raise HTTPException(status_code=503, detail="YOLO11 Pose model not available")
    
    return await process_pose_detection(payload, yolo11_pose_model, "yolo11_pose", "processed_yolo11_pose")

@app.post("/yolo11/object")
async def yolo11_object_detection(payload: VideoAnalysisURLRequest, request: Request):
    """YOLO11 Object Detection - person, sports ball, tennis racket"""
    logger.info("YOLO11 Object detection request received")
    
    if yolo11_pose_model is None:  # Using pose model for object detection too
        raise HTTPException(status_code=503, detail="YOLO11 model not available")
    
    return await process_object_detection(payload, yolo11_pose_model, "yolo11_object", "processed_yolo11_object")

@app.post("/yolov8/pose")
async def yolov8_pose_detection(payload: VideoAnalysisURLRequest, request: Request):
    """YOLOv8 Pose Detection - 17 keypoint human pose estimation"""
    logger.info("YOLOv8 Pose detection request received")
    
    if yolov8_pose_model is None:
        raise HTTPException(status_code=503, detail="YOLOv8 Pose model not available")
    
    return await process_pose_detection(payload, yolov8_pose_model, "yolov8_pose", "processed_yolov8_pose")

@app.post("/yolov8/object")
async def yolov8_object_detection(payload: VideoAnalysisURLRequest, request: Request):
    """YOLOv8 Object Detection - person, sports ball, tennis racket"""
    logger.info("YOLOv8 Object detection request received")
    
    if yolov8_object_model is None:
        raise HTTPException(status_code=503, detail="YOLOv8 Object model not available")
    
    return await process_object_detection(payload, yolov8_object_model, "yolov8_object", "processed_yolov8_object")

async def process_pose_detection(payload: VideoAnalysisURLRequest, model: YOLO, 
                               service_name: str, gcs_folder: str):
    """Common pose detection processing logic"""
    temp_downloaded_path = None
    try:
        # Download video
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(str(payload.video_url))
            response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(response.content)
            temp_downloaded_path = temp_file.name
        
        # Extract frames
        video_info = get_video_info(temp_downloaded_path)
        # Extract ALL frames for smooth video output
        frames = extract_frames(temp_downloaded_path, num_frames_to_extract=-1)
        
        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from video")
        
        all_poses_per_frame = []
        annotated_frames = []
        
        # Process in batches
        batch_size = 8
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            try:
                results = model(batch, verbose=False, half=torch.cuda.is_available())
                
                for frame_idx, result in enumerate(results):
                    current_poses = []
                    
                    if hasattr(result, 'keypoints') and result.keypoints is not None and result.keypoints.data is not None:
                        for keypoints_tensor in result.keypoints.data:
                            kpts = keypoints_tensor.cpu().numpy()
                            
                            # Extract keypoints
                            keypoint_names = [
                                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                "left_wrist", "right_wrist", "left_hip", "right_hip",
                                "left_knee", "right_knee", "left_ankle", "right_ankle"
                            ]
                            
                            keypoint_dict = {}
                            for kpt_idx, kpt_data in enumerate(kpts):
                                if kpt_idx < len(keypoint_names):
                                    x, y, conf = kpt_data
                                    keypoint_dict[keypoint_names[kpt_idx]] = {
                                        "x": float(x), "y": float(y), "confidence": float(conf)
                                    }
                            
                            if keypoint_dict:
                                overall_conf = float(kpts[:, 2].mean()) if kpts.shape[0] > 0 else 0.0
                                current_poses.append({
                                    "keypoints": keypoint_dict,
                                    "confidence": overall_conf,
                                    "bbox": {}  # TODO: Extract bbox if needed
                                })
                    
                    all_poses_per_frame.append(current_poses)
                    
                    if payload.video:
                        annotated_frames.append(draw_poses_on_frame(batch[frame_idx], current_poses))
                        
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                for _ in batch:
                    all_poses_per_frame.append([])
                    if payload.video:
                        annotated_frames.extend(batch)
        
        # Prepare response
        response_data = {}
        if payload.data:
            response_data["data"] = {"poses_per_frame": all_poses_per_frame}
        
        if payload.video:
            video_url = await create_video_from_frames(annotated_frames, video_info, gcs_folder)
            response_data["video_url"] = video_url
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{service_name} error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{service_name} internal error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)

async def process_object_detection(payload: VideoAnalysisURLRequest, model: YOLO,
                                 service_name: str, gcs_folder: str):
    """Common object detection processing logic"""
    temp_downloaded_path = None
    try:
        # Download video
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(str(payload.video_url))
            response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(response.content)
            temp_downloaded_path = temp_file.name
        
        # Extract frames
        video_info = get_video_info(temp_downloaded_path)
        # Extract ALL frames for smooth video output
        frames = extract_frames(temp_downloaded_path, num_frames_to_extract=-1)
        
        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from video")
        
        # Object classes for padel
        PADEL_CLASSES = {0: "person", 32: "sports ball", 38: "tennis racket"}
        
        all_objects_per_frame = []
        annotated_frames = []
        
        # Process in batches
        batch_size = 8
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            try:
                results = model(batch, verbose=False, half=torch.cuda.is_available())
                
                for frame_idx, result in enumerate(results):
                    current_objects = []
                    
                    if hasattr(result, 'boxes') and result.boxes is not None and result.boxes.data is not None:
                        boxes_data = result.boxes.data.cpu().tolist()
                        for box_data in boxes_data:
                            x1, y1, x2, y2, conf, cls_raw = box_data
                            cls = int(cls_raw)
                            
                            if cls in PADEL_CLASSES and conf > payload.confidence:
                                current_objects.append({
                                    "class": PADEL_CLASSES[cls],
                                    "confidence": float(conf),
                                    "bbox": {"x1": float(x1), "y1": float(y1), 
                                           "x2": float(x2), "y2": float(y2)}
                                })
                    
                    all_objects_per_frame.append(current_objects)
                    
                    if payload.video:
                        annotated_frames.append(draw_objects_on_frame(batch[frame_idx], current_objects))
                        
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                for _ in batch:
                    all_objects_per_frame.append([])
                    if payload.video:
                        annotated_frames.extend(batch)
        
        # Prepare response
        response_data = {}
        if payload.data:
            response_data["data"] = {"objects_per_frame": all_objects_per_frame}
        
        if payload.video:
            video_url = await create_video_from_frames(annotated_frames, video_info, gcs_folder)
            response_data["video_url"] = video_url
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{service_name} error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{service_name} internal error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)

if __name__ == "__main__":
    logger.info("Starting YOLO Combined Service on port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_config=None)