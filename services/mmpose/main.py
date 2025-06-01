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
import math
import httpx
import torch
import uuid
import uvicorn
from datetime import datetime
from google.cloud import storage
import subprocess

# MMPose v1.x API imports
try:
    from mmpose.apis import init_model, inference_topdown  # type: ignore
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    # Note: Logger not yet configured, message will appear after logging setup

# Simple frame extraction function to replace missing utils
def get_video_info(video_path: str) -> dict:
    """Get basic video info using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "frame_count": frame_count}

def extract_frames(video_path: str, num_frames_to_extract: int = -1) -> List[np.ndarray]:
    """Extract frames from video using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if num_frames_to_extract == -1:
        # Extract all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    else:
        # Extract specific number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, frame_count // num_frames_to_extract)
        
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= num_frames_to_extract:
                break
    
    cap.release()
    return frames

# Import shared configuration loader
try:
    sys.path.append('/app')
    sys.path.append('../shared')
    from shared.config_loader import ConfigLoader, merge_env_overrides  # type: ignore
except ImportError:
    # Fallback if shared module not available
    class ConfigLoader:
        def __init__(self, service_name: str, config_dir: str = "/app/config"):
            pass
        def load_config(self): return {}
        def get_feature_flags(self): return {}
        def is_feature_enabled(self, feature_name: str): return False
        def get_service_info(self): return {"service": "mmpose", "version": "2.0.0"}
    def merge_env_overrides(config): return config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- MMPOSE BIOMECHANICS SERVICE STARTED ---")

# Log import status now that logger is configured
if not MMPOSE_AVAILABLE:
    logger.warning("MMPose not available - service will run in fallback mode")
else:
    logger.info("MMPose imports successful")

# Initialize configuration loader
config_loader = ConfigLoader("mmpose", "/app/config")
service_config = merge_env_overrides(config_loader.load_config())
logger.info(f"Configuration loaded: {config_loader.get_service_info()}")

app = FastAPI(title="MMPose Biomechanics Service", version="2.0.0")
logger.info("FastAPI app created for MMPose service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_mmpose"
WEIGHTS_DIR = "/app/weights"

logger.info(f"Feature flags: {config_loader.get_feature_flags()}")

# Pydantic Models
class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False
    data: bool = True

# Helper Functions
async def upload_to_gcs(video_path: str, object_name: Optional[str] = None) -> str:
    if not object_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        object_name = f"{GCS_FOLDER}/video_{timestamp}_{unique_id}.mp4"
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

# Model Loading - from existing local files only
mmpose_model = None
model_info = {"name": "none", "source": "none"}

if MMPOSE_AVAILABLE:
    model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading MMPose model from local files on device: {model_device}")

    try:
        # Try RTMPose-M from local files
        config_file = '/app/config/rtmpose_complete.py'
        checkpoint_file = '/app/weights/mmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'

        if os.path.exists(config_file) and os.path.exists(checkpoint_file):
            logger.info(f"Loading RTMPose-M from: {checkpoint_file}")
            mmpose_model = init_model(config_file, checkpoint_file, device=model_device)
            model_info = {"name": "RTMPose-M", "source": "local_files", "status": "loaded"}
            logger.info("✅ RTMPose-M loaded successfully from local files")
        else:
            logger.info(f"RTMPose-M files not found - config: {os.path.exists(config_file)}, checkpoint: {os.path.exists(checkpoint_file)}")

        # Try HRNet-W48 fallback from local files
        if mmpose_model is None:
            config_file_hrnet = '/app/config/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
            checkpoint_file_hrnet = '/app/weights/mmpose/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
            
            if os.path.exists(config_file_hrnet) and os.path.exists(checkpoint_file_hrnet):
                logger.info(f"Loading HRNet-W48 from: {checkpoint_file_hrnet}")
                mmpose_model = init_model(config_file_hrnet, checkpoint_file_hrnet, device=model_device)
                model_info = {"name": "HRNet-W48", "source": "local_files_fallback", "status": "loaded"}
                logger.info("✅ HRNet-W48 loaded successfully from local files")
            else:
                logger.info(f"HRNet-W48 files not found - config: {os.path.exists(config_file_hrnet)}, checkpoint: {os.path.exists(checkpoint_file_hrnet)}")

    except Exception as e:
        logger.error(f"❌ Failed to load MMPose model: {e}", exc_info=True)

if mmpose_model is None:
    logger.critical("❌ No MMPose model could be loaded from local files - service will run in fallback mode")
else:
    logger.info(f"✅ MMPose service ready with {model_info.get('name', 'Unknown')} model")
    logger.info("🔧 Speed optimizations: Local file loading, CUDA device acceleration")


def analyze_frame_pose(frame_content: np.ndarray) -> Dict[str, Any]:
    """Simplified pose analysis - returns only raw keypoints from MMPose"""
    if mmpose_model is None:
        logger.warning("MMPose model not loaded")
        return {"keypoints": {}}

    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    current_keypoints = {}

    try:
        # Run MMPose inference using top-down API
        pose_results_list = inference_topdown(mmpose_model, frame_content)
        pose_data_samples = pose_results_list if isinstance(pose_results_list, list) else [pose_results_list]

        if pose_data_samples:
            data_sample = pose_data_samples[0]
            if (hasattr(data_sample, 'pred_instances') and
                data_sample.pred_instances and
                len(data_sample.pred_instances) > 0):

                pred_kpts_tensor = data_sample.pred_instances.keypoints[0]
                pred_scores_tensor = data_sample.pred_instances.keypoint_scores[0]
                pred_kpts = pred_kpts_tensor.cpu().numpy()
                pred_scores = pred_scores_tensor.cpu().numpy()

                # Extract keypoints only
                for idx in range(pred_kpts.shape[0]):
                    if idx < len(keypoint_names):
                        current_keypoints[keypoint_names[idx]] = {
                            "x": float(pred_kpts[idx, 0]),
                            "y": float(pred_kpts[idx, 1]),
                            "confidence": float(pred_scores[idx])
                        }

    except Exception as e:
        logger.error(f"Error during MMPose inference: {e}", exc_info=True)
        return {"keypoints": {}}

    return {"keypoints": current_keypoints}

def draw_pose_on_frame(frame: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """Draw simple pose visualization on frame"""
    annotated_frame = frame.copy()
    keypoints = analysis.get("keypoints", {})

    # Define skeleton connections
    connections = [
        ("nose", "left_eye"), ("nose", "right_eye"),
        ("left_eye", "left_ear"), ("right_eye", "right_ear"),
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle")
    ]

    # Draw keypoints
    for name, data in keypoints.items():
        x, y, conf = int(data.get("x", 0)), int(data.get("y", 0)), data.get("confidence", 0.0)
        if conf > 0.5:
            cv2.circle(annotated_frame, (x, y), 4, (0, 255, 0), -1)

    # Draw skeleton connections
    for p1_name, p2_name in connections:
        p1, p2 = keypoints.get(p1_name), keypoints.get(p2_name)
        if (p1 and p2 and
            p1.get("confidence", 0) > 0.5 and
            p2.get("confidence", 0) > 0.5):
            cv2.line(annotated_frame,
                    (int(p1["x"]), int(p1["y"])),
                    (int(p2["x"]), int(p2["y"])),
                    (0, 255, 255), 2)

    return annotated_frame

async def create_video_from_frames(frames: List[np.ndarray], video_info: dict) -> Optional[str]:
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

        try:
            for frame in frames:
                if process.poll() is None and process.stdin and not process.stdin.closed:
                    try:
                        process.stdin.write(frame.tobytes())
                        process.stdin.flush()
                    except (IOError, BrokenPipeError, OSError) as e:
                        logger.warning(f"FFmpeg stdin pipe error: {e}, stopping frame writing")
                        break
                else:
                    logger.warning("FFmpeg process terminated early or stdin unavailable")
                    break
            
            # Safely close stdin
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except Exception as e:
                    logger.warning(f"Error closing stdin: {e}")

            stdout, stderr = process.communicate(timeout=120)
            if process.returncode == 0:
                logger.info("Video created successfully")
                return await upload_to_gcs(output_video_path)
            else:
                logger.error(f"FFMPEG failed with code {process.returncode}, stderr: {stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("FFMPEG timed out")
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            return None
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            if process:
                process.terminate()
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
    
    for model_key, model_config in model_configs.items():
        is_loaded = False
        if model_key == "rtmpose_m" and mmpose_model is not None:
            is_loaded = model_info.get("name") == "RTMPose-M"
        elif model_key == "hrnet_w48" and mmpose_model is not None:
            is_loaded = model_info.get("name") == "HRNet-W48"
            
        models_status[model_key] = {
            "loaded": is_loaded,
            "enabled": model_config.get("enabled", False),
            "version": model_config.get("version", "unknown"),
            "config": model_config.get("config", "unknown"),
            "checkpoint": model_config.get("checkpoint"),
            "fallback": model_config.get("fallback")
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
    biomechanics_config = current_config.get("biomechanics", {})
    performance_config = current_config.get("performance", {})
    
    # Overall health status
    model_loaded = mmpose_model is not None
    overall_status = "healthy" if model_loaded else "unhealthy"
    status_code = 200 if model_loaded else 503
    
    response_data = {
        "status": overall_status,
        "service": service_info,
        "models": models_status,
        "features": active_features,
        "performance": {
            "confidence_threshold": performance_config.get("confidence_threshold", 0.5),
            "keypoint_threshold": performance_config.get("keypoint_threshold", 0.5),
            "max_concurrent_requests": performance_config.get("max_concurrent_requests", 3)
        },
        "deployment": {
            "ready_for_upgrade": model_loaded,
            "config_hot_reload": True,
            "mmpose_available": MMPOSE_AVAILABLE,
            "current_model": model_info,
            "last_config_check": datetime.now().isoformat()
        }
    }
    
    if overall_status == "unhealthy":
        return JSONResponse(content=response_data, status_code=status_code)
    
    return response_data

@app.post("/mmpose/pose")
async def mmpose_pose_analysis(payload: VideoAnalysisURLRequest, request: Request):
    """
    MMPose Pose Detection Endpoint

    High-precision pose estimation using RTMPose or HRNet.
    Returns raw keypoint data:
    - 17 keypoint pose detection with confidence scores
    - Simple skeleton visualization
    - No post-processing analytics
    """
    logger.info("MMPose pose detection request received")

    if mmpose_model is None:
        raise HTTPException(status_code=503, detail=f"MMPose model not available. Info: {model_info}")

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

        all_analyses = []
        annotated_frames = []

        logger.info(f"Starting biomechanical analysis for {len(frames)} frames using {model_info.get('name', 'unknown')}")

        # Process each frame
        for frame_idx, frame_content in enumerate(frames):
            logger.debug(f"Analyzing frame {frame_idx}")
            analysis_result = analyze_frame_pose(frame_content)
            all_analyses.append(analysis_result)

            if payload.video:
                annotated_frames.append(draw_pose_on_frame(frame_content, analysis_result))

        logger.info(f"Finished analysis. Processed: {len(all_analyses)} frames")

        # Prepare response
        response_data = {}

        if payload.data:
            response_data["data"] = {
                "poses_per_frame": all_analyses
            }

        if payload.video:
            video_url = await create_video_from_frames(annotated_frames, video_info)
            response_data["video_url"] = video_url

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to download video: {e.response.status_code}")
    except Exception as e:
        logger.error(f"MMPose analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)

if __name__ == "__main__":
    logger.info("Starting MMPose service on port 8003")
    if mmpose_model is None:
        logger.critical("MMPose model could not be loaded - service will be unhealthy")
    else:
        logger.info(f"MMPose service starting with {model_info.get('name', 'unknown')} model")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_config=None)