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
import subprocess
import psutil
import gc
import math

# Configure logging (moved to top for early availability)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# MMPose ViTPose++ imports
MMPOSE_AVAILABLE = False
try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.utils import register_all_modules
    register_all_modules()
    MMPOSE_AVAILABLE = True
    logger.info("MMPose imports successful.")
except ImportError as e:
    logger.error(f"MMPose import failed: {e}. ViTPose service will run in fallback mode.", exc_info=True)
except Exception as e:
    logger.critical(f"Unexpected error during MMPose import: {e}", exc_info=True)
 
# Setup for shared config
try:
    sys.path.append('/app')
    sys.path.append('../shared')
    from shared.config_loader import ConfigLoader, merge_env_overrides
except ImportError:
    # Fallback if shared module not available - renamed to avoid conflict
    class FallbackConfigLoaderViTPose:
        def __init__(self, service_name: str, config_dir: str = "/app/config"):
            pass
        def load_config(self): return {}
        def get_feature_flags(self): return {}
        def is_feature_enabled(self, feature_name: str): return False
        def get_service_info(self): return {"service": "vitpose", "version": "1.0.0"}
    ConfigLoader = FallbackConfigLoaderViTPose # Assign fallback to main ConfigLoader name
    def merge_env_overrides(config): return config

logger.info("--- VITPOSE++ POSE ESTIMATION SERVICE STARTED ---")

# Initialize configuration loader
config_loader = ConfigLoader("vitpose", "/app/config")
service_config = merge_env_overrides(config_loader.load_config())
logger.info(f"Configuration loaded: {config_loader.get_service_info()}")

app = FastAPI(title="ViTPose++ Pose Estimation Service", version="1.0.0")
logger.info("FastAPI app created for ViTPose++ service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_vitpose"
WEIGHTS_DIR = "/app/weights"

# Pydantic Models
class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False
    data: bool = True
    confidence: float = 0.3

# Helper Functions
def get_gpu_memory_info():
    """Get GPU memory usage information"""
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            cached = torch.cuda.memory_reserved(device)
            
            return {
                "total_mb": total_memory // (1024 * 1024),
                "allocated_mb": allocated // (1024 * 1024),
                "cached_mb": cached // (1024 * 1024),
                "free_mb": (total_memory - allocated) // (1024 * 1024)
            }
    except Exception as e:
        logger.warning(f"Could not get GPU memory info: {e}")
    
    return {"total_mb": 0, "allocated_mb": 0, "cached_mb": 0, "free_mb": 0}

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("GPU memory cleaned up")
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")

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

# Model Loading - from existing local files only (no downloading)
vitpose_model = None
vitpose_inferencer = None
model_info = {"name": "none", "source": "none"}

if MMPOSE_AVAILABLE:
    logger.info("Attempting to load ViTPose++ model from existing local files")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    try:
        # Check configuration for enabled models
        models_config = service_config.get("models", {})
        vitpose_enabled = models_config.get("vitpose_base", {}).get("enabled", True)
        hrnet_enabled = models_config.get("hrnet_w48", {}).get("enabled", True)
        
        logger.info(f"Model configuration - ViTPose enabled: {vitpose_enabled}, HRNet enabled: {hrnet_enabled}")
        
        # Try loading ViTPose-Base from local checkpoint (if enabled and exists)
        if vitpose_enabled:
            local_checkpoint = '/app/weights/vitpose/vitpose_base_coco_256x192.pth'
            if os.path.exists(local_checkpoint):
                logger.info(f"Loading ViTPose-Base from local checkpoint: {local_checkpoint}")
                try:
                    config_name = 'td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
                    vitpose_model = init_model(config_name, local_checkpoint, device=device)
                    vitpose_inferencer = lambda img: inference_topdown(model=vitpose_model, image=img)
                    
                    # Enable FP16 for VRAM efficiency
                    if torch.cuda.is_available():
                        vitpose_model.half()
                        logger.info("Enabled FP16 precision for VRAM efficiency")
                    
                    model_info = {"name": "ViTPose-Base", "source": "local_checkpoint", "precision": "fp16" if torch.cuda.is_available() else "fp32"}
                    logger.info("✅ ViTPose-Base model loaded successfully from local checkpoint")
                except Exception as e:
                    logger.error(f"Failed to load ViTPose-Base: {e}", exc_info=True)
            else:
                logger.info(f"ViTPose-Base checkpoint not found at: {local_checkpoint}")
        
        # Try HRNet fallback from local files (if enabled and ViTPose failed)
        if hrnet_enabled and vitpose_model is None:
            hrnet_checkpoint = '/app/weights/mmpose/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
            if os.path.exists(hrnet_checkpoint):
                logger.info(f"Loading HRNet-W48 fallback from: {hrnet_checkpoint}")
                try:
                    config_name = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192'
                    vitpose_model = init_model(config_name, hrnet_checkpoint, device=device)
                    vitpose_inferencer = lambda img: inference_topdown(model=vitpose_model, image=img)
                    
                    if torch.cuda.is_available():
                        vitpose_model.half()
                        logger.info("Enabled FP16 precision for VRAM efficiency")
                    
                    model_info = {"name": "HRNet-W48", "source": "local_fallback", "precision": "fp16" if torch.cuda.is_available() else "fp32"}
                    logger.info("✅ HRNet-W48 model loaded successfully from local checkpoint")
                except Exception as e:
                    logger.error(f"Failed to load HRNet-W48: {e}", exc_info=True)
            else:
                logger.info(f"HRNet-W48 checkpoint not found at: {hrnet_checkpoint}")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}", exc_info=True)
        vitpose_model = None

if vitpose_model is None:
    logger.critical("❌ No ViTPose++ model could be loaded from local files - service will run in fallback mode")
else:
    logger.info(f"✅ ViTPose++ service ready with {model_info.get('name', 'unknown')} model")
    logger.info("🔧 Speed optimizations: FP16 precision, GPU memory cleanup enabled")

def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3"""
    try:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        vector1 = np.array([x1 - x2, y1 - y2])
        vector2 = np.array([x3 - x2, y3 - y2])

        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except (ZeroDivisionError, ValueError):
        return 0.0

def analyze_frame_pose(frame_content: np.ndarray, confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """Simplified ViTPose++ analysis - returns only raw keypoints"""
    if vitpose_model is None:
        logger.warning("ViTPose++ model not loaded")
        return {"keypoints": {}}

    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    current_keypoints = {}

    try:
        # Run ViTPose++ inference using inferencer
        if vitpose_inferencer is None:
            logger.warning("ViTPose inferencer not initialized")
            return {"keypoints": {}}
        
        pose_results = vitpose_inferencer(frame_content)
        
        if pose_results and 'predictions' in pose_results:
            pose_data_samples = pose_results['predictions']
        elif pose_results:
            pose_data_samples = [pose_results]
        else:
            pose_data_samples = []

        if pose_data_samples:
            data_sample = pose_data_samples[0]
            if (hasattr(data_sample, 'pred_instances') and
                data_sample.pred_instances and
                len(data_sample.pred_instances) > 0):

                pred_kpts_tensor = data_sample.pred_instances.keypoints[0]
                pred_scores_tensor = data_sample.pred_instances.keypoint_scores[0]
                pred_kpts = pred_kpts_tensor.cpu().numpy()
                pred_scores = pred_scores_tensor.cpu().numpy()

                # Extract keypoints
                for idx in range(pred_kpts.shape[0]):
                    if idx < len(keypoint_names):
                        confidence = float(pred_scores[idx])
                        if confidence >= confidence_threshold:
                            current_keypoints[keypoint_names[idx]] = {
                                "x": float(pred_kpts[idx, 0]),
                                "y": float(pred_kpts[idx, 1]),
                                "confidence": confidence
                            }

    except Exception as e_analyze:
        logger.error(f"Error during ViTPose++ inference: {e_analyze}", exc_info=True)
        return {"keypoints": {}}
    finally:
        # Clean up GPU memory after inference
        cleanup_gpu_memory()

    return {"keypoints": current_keypoints}

def draw_pose_on_frame(frame: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """Draw simplified pose estimation - keypoints and skeleton only"""
    annotated_frame = frame.copy()
    keypoints = analysis.get("keypoints", {})

    # Define skeleton connections
    connections = [
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
        ("nose", "left_eye"), ("nose", "right_eye"),
        ("left_eye", "left_ear"), ("right_eye", "right_ear")
    ]

    # Draw keypoints
    for name, data in keypoints.items():
        x, y, conf = int(data.get("x", 0)), int(data.get("y", 0)), data.get("confidence", 0.0)
        if conf > 0.5:
            cv2.circle(annotated_frame, (x, y), 5, (0, int(255*conf), int(255*(1-conf))), -1)

    # Draw skeleton connections
    for p1_name, p2_name in connections:
        p1, p2 = keypoints.get(p1_name), keypoints.get(p2_name)
        if (p1 and p2 and
            p1.get("confidence", 0) > 0.5 and
            p2.get("confidence", 0) > 0.5):
            cv2.line(annotated_frame,
                    (int(p1["x"]), int(p1["y"])),
                    (int(p2["x"]), int(p2["y"])),
                    (0, 255, 0), 2)

    return annotated_frame

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video information using cv2"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if fps > 0 else 0
    }

def extract_frames(video_path: str, num_frames_to_extract: int = -1) -> List[np.ndarray]:
    """Extract frames from video"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return frames
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        if num_frames_to_extract > 0 and frame_count >= num_frames_to_extract:
            break
    
    cap.release()
    return frames

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
async def health_check():
    """Enhanced health check with model and GPU status"""
    
    # Model status
    model_loaded = vitpose_model is not None
    gpu_info = get_gpu_memory_info()
    
    # Check if models are intentionally disabled in config
    models_config = service_config.get("models", {})
    vitpose_enabled = models_config.get("vitpose_base", {}).get("enabled", True)
    hrnet_enabled = models_config.get("hrnet_w48", {}).get("enabled", True)
    models_intentionally_disabled = not vitpose_enabled and not hrnet_enabled
    
    # Service is healthy if:
    # 1. Models are loaded, OR
    # 2. Models are intentionally disabled in config (fallback mode)
    is_healthy = model_loaded or models_intentionally_disabled
    
    # System info
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    response_data = {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": config_loader.get_service_info(),
        "models": {
            "model_loaded": model_loaded,
            "model_info": model_info,
            "mmpose_available": MMPOSE_AVAILABLE,
            "vitpose_enabled": vitpose_enabled,
            "hrnet_enabled": hrnet_enabled,
            "fallback_mode": models_intentionally_disabled
        },
        "gpu_memory": gpu_info,
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available // (1024 * 1024)
        },
        "features": {
            "fp16_enabled": torch.cuda.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        }
    }
    
    status_code = 200 if is_healthy else 503
    if not is_healthy:
        return JSONResponse(content=response_data, status_code=status_code)
    
    return response_data

@app.post("/analyze")
async def vitpose_pose_analysis(payload: VideoAnalysisURLRequest, request: Request):
    """
    ViTPose++ Pose Analysis Endpoint
    
    High-precision pose estimation using ViTPose++ with:
    - Vision Transformer based pose estimation
    - FP16 precision for VRAM efficiency
    - Advanced keypoint detection and joint angle calculation
    - GPU memory optimization
    - Configurable confidence thresholds
    
    Ideal for detailed pose analysis with minimal VRAM usage.
    """
    logger.info("ViTPose++ pose analysis request received")

    if vitpose_model is None:
        raise HTTPException(status_code=503, detail=f"ViTPose++ model not available. Info: {model_info}")

    temp_downloaded_path = None
    try:
        # Check GPU memory before processing
        initial_gpu_info = get_gpu_memory_info()
        logger.info(f"GPU memory before processing: {initial_gpu_info}")
        
        # Download video
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(str(payload.video_url))
            response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(response.content)
            temp_downloaded_path = temp_file.name

        # Extract frames
        video_info = get_video_info(temp_downloaded_path)
        frames = extract_frames(temp_downloaded_path, num_frames_to_extract=-1)

        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from video")

        all_analyses = []
        annotated_frames = []

        logger.info(f"Starting ViTPose++ analysis for {len(frames)} frames using {model_info.get('name', 'unknown')}")

        # Process each frame
        for frame_idx, frame_content in enumerate(frames):
            logger.debug(f"Analyzing frame {frame_idx}")
            analysis_result = analyze_frame_pose(
                frame_content, 
                confidence_threshold=payload.confidence
            )
            all_analyses.append(analysis_result)

            if payload.video:
                annotated_frames.append(draw_pose_on_frame(frame_content, analysis_result))

        logger.info(f"Finished analysis. Processed: {len(all_analyses)} frames")

        # Prepare response
        response_data = {}

        if payload.data:
            response_data["data"] = {
                "poses_per_frame": all_analyses,
                "model_info": model_info,
                "processing_summary": {
                    "total_frames": len(frames),
                    "successful_analyses": len([a for a in all_analyses if a.get("keypoints", {})]),
                    "total_keypoints": sum(len(a.get("keypoints", {})) for a in all_analyses),
                    "confidence_threshold": payload.confidence
                }
            }

        if payload.video:
            video_url = await create_video_from_frames(annotated_frames, video_info)
            response_data["video_url"] = video_url

        # Final GPU memory check
        final_gpu_info = get_gpu_memory_info()
        logger.info(f"GPU memory after processing: {final_gpu_info}")
        response_data["gpu_memory_usage"] = {
            "initial": initial_gpu_info,
            "final": final_gpu_info
        }

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to download video: {e.response.status_code}")
    except Exception as e:
        logger.error(f"ViTPose++ analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)
        # Final cleanup
        cleanup_gpu_memory()

if __name__ == "__main__":
    logger.info("Starting ViTPose++ service on port 8006")
    if vitpose_model is None:
        logger.critical("ViTPose++ model could not be loaded - service will be unhealthy")
    else:
        logger.info(f"ViTPose++ service starting with {model_info.get('name', 'unknown')} model")
    uvicorn.run(app, host="0.0.0.0", port=8006, log_config=None)