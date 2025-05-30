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

# Setup for utils and shared config
try:
    from utils.video_utils import get_video_info, extract_frames
    from utils.ball_tracker import smooth_ball_trajectory, draw_enhanced_ball_trajectory
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from utils.video_utils import get_video_info, extract_frames
    from utils.ball_tracker import smooth_ball_trajectory, draw_enhanced_ball_trajectory

# Import shared configuration loader
try:
    sys.path.append('/app')
    sys.path.append('../shared')
    from shared.config_loader import ConfigLoader, merge_env_overrides
except ImportError:
    # Fallback if shared module not available
    class ConfigLoader:
        def __init__(self, service_name: str, config_dir: str = "/app/config"):
            pass
        def load_config(self): return {}
        def get_feature_flags(self): return {}
        def is_feature_enabled(self, feature_name: str): return False
        def get_service_info(self): return {"service": "yolo_combined", "version": "2.0.0"}
    def merge_env_overrides(config): return config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- YOLO COMBINED SERVICE (YOLO11 + YOLOv8) STARTED ---")

# Initialize configuration loader
config_loader = ConfigLoader("yolo_combined", "/app/config")
service_config = merge_env_overrides(config_loader.load_config())
logger.info(f"Configuration loaded: {config_loader.get_service_info()}")

app = FastAPI(title="YOLO Combined Service - YOLO11 + YOLOv8", version="2.0.0")
logger.info("FastAPI app created for YOLO Combined service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
WEIGHTS_DIR = "/app/weights"

# Dynamic model configuration based on config file
model_configs = service_config.get("models", {})
YOLO11_POSE_MODEL = model_configs.get("yolo11_pose", {}).get("file", "yolo11n-pose.pt")
YOLO11_OBJECT_MODEL = model_configs.get("yolo11_object", {}).get("file", "yolo11n.pt")
YOLOV8_OBJECT_MODEL = model_configs.get("yolov8_object", {}).get("file", "yolov8n.pt")
YOLOV8_POSE_MODEL = model_configs.get("yolov8_pose", {}).get("file", "yolov8n-pose.pt")

logger.info(f"Model files configured: YOLO11 Pose={YOLO11_POSE_MODEL}, YOLO11 Object={YOLO11_OBJECT_MODEL}")
logger.info(f"Feature flags: {config_loader.get_feature_flags()}")

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
    model_path = os.path.join(WEIGHTS_DIR, "ultralytics", model_name)
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
yolo11_object_model = load_model(YOLO11_OBJECT_MODEL, "YOLO11 Object Model")
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
async def enhanced_health_check():
    """Enhanced health check with model versions and feature flags"""
    
    # Reload config for latest feature flags
    current_config = merge_env_overrides(config_loader.load_config())
    
    # Model status with versions
    models_status = {}
    model_configs = current_config.get("models", {})
    
    for model_key, model_info in model_configs.items():
        is_loaded = False
        model_file = model_info.get("file", "unknown")
        
        # Check if model is actually loaded
        if model_key == "yolo11_pose":
            is_loaded = yolo11_pose_model is not None
        elif model_key == "yolo11_object":
            is_loaded = yolo11_object_model is not None
        elif model_key == "yolov8_object":
            is_loaded = yolov8_object_model is not None
        elif model_key == "yolov8_pose":
            is_loaded = yolov8_pose_model is not None
            
        models_status[model_key] = {
            "loaded": is_loaded,
            "enabled": model_info.get("enabled", False),
            "version": model_info.get("version", "unknown"),
            "file": model_file,
            "fallback": model_info.get("fallback")
        }
    
    # Feature flags status
    feature_flags = current_config.get("features", {})
    active_features = {}
    for feature_name, feature_info in feature_flags.items():
        active_features[feature_name] = {
            "enabled": feature_info.get("enabled", False),
            "description": feature_info.get("description", "")
        }
    
    # Service information
    service_info = config_loader.get_service_info()
    performance_config = current_config.get("performance", {})
    
    # Overall health status
    any_model_loaded = any(status["loaded"] for status in models_status.values())
    overall_status = "healthy" if any_model_loaded else "unhealthy"
    status_code = 200 if any_model_loaded else 503
    
    response_data = {
        "status": overall_status,
        "service": service_info,
        "models": models_status,
        "features": active_features,
        "performance": {
            "batch_size": performance_config.get("batch_size", 8),
            "confidence_threshold": performance_config.get("confidence_threshold", 0.3),
            "max_concurrent_requests": performance_config.get("max_concurrent_requests", 5)
        },
        "deployment": {
            "ready_for_upgrade": any_model_loaded,
            "config_hot_reload": True,
            "environment_overrides": bool(os.getenv("YOLO11_POSE_ENABLED") or os.getenv("FEATURE_ENHANCED_BALL_TRACKING_ENABLED")),
            "last_config_check": datetime.now().isoformat()
        }
    }
    
    if overall_status == "unhealthy":
        return JSONResponse(content=response_data, status_code=status_code)
    
    return response_data

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

    if yolo11_object_model is None:
        raise HTTPException(status_code=503, detail="YOLO11 Object model not available")

    return await process_object_detection(payload, yolo11_object_model, "yolo11_object", "processed_yolo11_object")

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
    """Enhanced object detection with advanced ball tracking"""
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
        fps = float(video_info.get("fps", 30.0))
        if fps <= 0: fps = 30.0
        
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

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                for _ in batch:
                    all_objects_per_frame.append([])

        # Apply enhanced ball tracking with Kalman filtering and trajectory smoothing
        logger.info("Applying enhanced ball tracking with Kalman filtering...")
        enhanced_objects_per_frame = smooth_ball_trajectory(all_objects_per_frame, fps=fps)

        # Create annotated frames with enhanced visualization
        if payload.video:
            for frame_idx, frame_objects in enumerate(enhanced_objects_per_frame):
                if frame_idx < len(frames):
                    # Separate ball objects for enhanced visualization
                    ball_objects = [obj for obj in frame_objects if obj.get("class") == "sports ball"]
                    other_objects = [obj for obj in frame_objects if obj.get("class") != "sports ball"]
                    
                    # Draw regular objects
                    annotated_frame = draw_objects_on_frame(frames[frame_idx], other_objects)
                    
                    # Draw enhanced ball trajectory
                    annotated_frame = draw_enhanced_ball_trajectory(annotated_frame, ball_objects)
                    
                    annotated_frames.append(annotated_frame)

        # Prepare response
        response_data = {}
        if payload.data:
            response_data["data"] = {
                "objects_per_frame": enhanced_objects_per_frame,
                "ball_tracking": {
                    "enhanced": True,
                    "kalman_filtered": True,
                    "trajectory_smoothed": True,
                    "fps": fps
                }
            }

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

# NOTE: Removed if __name__ == "__main__": block for uvicorn CMD compatibility