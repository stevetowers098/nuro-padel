import os
from pathlib import Path

# === PRODUCTION SAFETY: DISABLE ALL AUTO-DOWNLOADS ===
# Check if we should run in offline mode
OFFLINE_MODE = os.getenv('YOLO_OFFLINE', 'true').lower() == 'true'
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/app/weights/super-gradients')

if OFFLINE_MODE:
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
import torch
import uuid
import uvicorn
from datetime import datetime
from google.cloud import storage
import subprocess
import httpx
import requests
import zipfile
import shutil

# Super Gradients for YOLO-NAS
try:
    from super_gradients.training import models as sg_models
    models = sg_models
    SUPER_GRADIENTS_AVAILABLE = True
except ImportError:
    SUPER_GRADIENTS_AVAILABLE = False
    logging.warning("Super Gradients not available - service will run in fallback mode")
    
    # Fallback models module
    class FallbackModels:
        @staticmethod
        def get(model_name: str, **kwargs):
            return None
    
    models = FallbackModels()

# Basic video processing utilities (fallback implementations)
def get_video_info(video_path: str) -> dict:
    """Basic video info extraction"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}

def extract_frames(video_path: str, num_frames_to_extract: int = -1) -> List[np.ndarray]:
    """Basic frame extraction"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
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

# Basic model optimizer (simplified for essential optimizations only)
class ModelOptimizer:
    def __init__(self, weights_dir: str):
        self.weights_dir = weights_dir
    
    def load_optimized_model(self, model_name: str, pytorch_model):
        """Return PyTorch model with basic optimizations"""
        if pytorch_model is not None:
            return ("pytorch", pytorch_model)
        return ("none", None)
    
    def predict_optimized(self, model_info, batch_data):
        """Basic prediction fallback"""
        return None

# Simplified config loader
class ConfigLoader:
    def __init__(self, service_name: str, config_dir: str = "/app/config"):
        self.service_name = service_name
        
    def load_config(self):
        return {"models": {}, "features": {}, "optimization": {}, "performance": {}}
        
    def get_feature_flags(self):
        return {}
        
    def is_feature_enabled(self, feature_name: str):
        return False
        
    def get_service_info(self):
        return {"service": "yolo_nas", "version": "2.0.0"}

def merge_env_overrides(config):
    return config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- YOLO-NAS HIGH-ACCURACY SERVICE STARTED ---")

# Initialize configuration loader
config_loader = ConfigLoader("yolo_nas", "/app/config")
service_config = merge_env_overrides(config_loader.load_config())
logger.info(f"Configuration loaded: {config_loader.get_service_info()}")

app = FastAPI(title="YOLO-NAS High-Accuracy Service", version="2.0.0")
logger.info("FastAPI app created for YOLO-NAS service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_yolo_nas"
WEIGHTS_DIR = "/app/weights"

# Enhanced model cache directory from environment
if MODEL_CACHE_DIR and os.path.exists(MODEL_CACHE_DIR):
    SUPER_GRADIENTS_CACHE = MODEL_CACHE_DIR
else:
    SUPER_GRADIENTS_CACHE = os.path.join(WEIGHTS_DIR, "super-gradients")

logger.info(f"🔧 OFFLINE_MODE: {OFFLINE_MODE}")
logger.info(f"🔧 MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")
logger.info(f"🔧 SUPER_GRADIENTS_CACHE: {SUPER_GRADIENTS_CACHE}")

logger.info(f"Feature flags: {config_loader.get_feature_flags()}")

# Basic optimization configuration - company recommended
optimization_config = {
    "fp16_enabled": True,       # Half precision for speed
    "batch_size": 8,           # Optimal batch processing
    "use_cuda": True,          # GPU acceleration
    "confidence_threshold": 0.3 # Detection threshold
}

# Pydantic Models
class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False
    data: bool = True
    confidence: float = 0.3


# Removed EnhancedAnalysisRequest - advanced biomechanics features removed

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

# Model Loading with Optimization Support
yolo_nas_pose_model = None
yolo_nas_object_model = None
model_info = {"pose_model": "none", "object_model": "none", "status": "none"}

# Diagnostic logging for volume mounts and model accessibility
logger.info(f"🔍 DIAGNOSTIC: WEIGHTS_DIR = {WEIGHTS_DIR}")
logger.info(f"🔍 DIAGNOSTIC: Checking weights directory structure...")
try:
    if os.path.exists(WEIGHTS_DIR):
        for root, dirs, files in os.walk(WEIGHTS_DIR):
            level = root.replace(WEIGHTS_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"🔍 {indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                logger.info(f"🔍 {subindent}{file}")
    else:
        logger.warning(f"⚠️ WEIGHTS_DIR {WEIGHTS_DIR} does not exist")
except Exception as e:
    logger.error(f"❌ Error checking weights directory: {e}")

# Initialize model optimizer
model_optimizer = ModelOptimizer(WEIGHTS_DIR)
pose_model_info = None
object_model_info = None

if SUPER_GRADIENTS_AVAILABLE:
    try:
        logger.info("Loading YOLO-NAS models...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load YOLO-NAS Pose Model (with offline support)
        def load_local_pose_model():
            """Try to load pose model from local cache first"""
            local_pose_checkpoint = os.path.join(SUPER_GRADIENTS_CACHE, "yolo_nas_pose_n_coco_pose.pth")
            
            if os.path.exists(local_pose_checkpoint):
                logger.info(f"🔄 Loading YOLO-NAS pose model from local checkpoint: {local_pose_checkpoint}")
                return models.get("yolo_nas_pose_n", checkpoint_path=local_pose_checkpoint, num_classes=17)
            return None

        def load_fallback_pose_model():
            """Fallback pose model loading"""
            try:
                logger.info("🔄 Trying fallback pose model loading...")
                return models.get("yolo_nas_pose_n", pretrained_weights="coco_pose")
            except Exception as e:
                logger.error(f"❌ Fallback pose model failed: {e}")
                return None

        try:
            if OFFLINE_MODE:
                # Try to load from local cache first
                yolo_nas_pose_model = load_local_pose_model()
                if yolo_nas_pose_model is None:
                    logger.warning("⚠️ No local pose model found, trying fallback")
                    yolo_nas_pose_model = load_fallback_pose_model()
            else:
                # Try online download first, fallback to local
                try:
                    yolo_nas_pose_model = models.get("yolo_nas_pose_n", pretrained_weights="coco_pose")
                except Exception as e:
                    logger.warning(f"Online pose model failed: {e}, trying local cache")
                    yolo_nas_pose_model = load_local_pose_model()
            
            if yolo_nas_pose_model and torch.cuda.is_available():
                yolo_nas_pose_model.to('cuda')
                yolo_nas_pose_model.half()  # Use half precision for speed
            
            # Try to load optimized version (TensorRT/ONNX) with PyTorch fallback
            pose_model_info = model_optimizer.load_optimized_model("yolo_nas_pose_n", yolo_nas_pose_model)
            backend, optimized_model = pose_model_info
            
            model_info["pose_model"] = f"yolo_nas_pose_n ({backend})"
            logger.info(f"✅ YOLO-NAS Pose model loaded successfully using {backend} backend")
        except Exception as e_pose:
            logger.error(f"❌ Failed to load YOLO-NAS pose model: {e_pose}")

        # Load YOLO-NAS Object Model (with offline support)
        def load_local_object_model():
            """Try to load object model from local cache first"""
            local_object_checkpoint = os.path.join(SUPER_GRADIENTS_CACHE, "yolo_nas_s_coco.pth")
            
            if os.path.exists(local_object_checkpoint):
                logger.info(f"🔄 Loading YOLO-NAS object model from local checkpoint: {local_object_checkpoint}")
                return models.get("yolo_nas_s", checkpoint_path=local_object_checkpoint, num_classes=80)
            return None

        def load_fallback_object_model():
            """Fallback object model loading"""
            try:
                logger.info("🔄 Trying fallback object model loading...")
                return models.get("yolo_nas_s", pretrained_weights="coco")
            except Exception as e:
                logger.error(f"❌ Fallback object model failed: {e}")
                return None

        try:
            # 🎯 HIGH PRIORITY: Check for custom-trained models first
            custom_object_path = os.path.join(WEIGHTS_DIR, "custom_padel_object.pth")
            
            if os.path.exists(custom_object_path):
                logger.info(f"🎯 Loading custom-trained object model: {custom_object_path}")
                yolo_nas_object_model = models.get("yolo_nas_s", checkpoint_path=custom_object_path, num_classes=4)
            elif OFFLINE_MODE:
                # Try to load from local cache first
                yolo_nas_object_model = load_local_object_model()
                if yolo_nas_object_model is None:
                    logger.warning("⚠️ No local object model found, trying fallback")
                    yolo_nas_object_model = load_fallback_object_model()
            else:
                # Try online download first, fallback to local
                try:
                    yolo_nas_object_model = models.get("yolo_nas_s", pretrained_weights="coco")
                except Exception as e:
                    logger.warning(f"Online object model failed: {e}, trying local cache")
                    yolo_nas_object_model = load_local_object_model()
            
            if yolo_nas_object_model and torch.cuda.is_available():
                yolo_nas_object_model.to('cuda')
                # Apply FP16 quantization for speed optimization
                if optimization_config.get("fp16_enabled", True):
                    yolo_nas_object_model.half()
                    logger.info("✅ Applied FP16 quantization to object model")
            
            # Try to load optimized version (TensorRT/ONNX) with PyTorch fallback
            if yolo_nas_object_model:
                object_model_info = model_optimizer.load_optimized_model("yolo_nas_s", yolo_nas_object_model)
                backend, optimized_model = object_model_info
                
                model_info["object_model"] = f"yolo_nas_s ({backend})"
                logger.info(f"✅ YOLO-NAS Object model loaded successfully using {backend} backend")
        except Exception as e_object:
            logger.error(f"❌ Failed to load YOLO-NAS object model: {e_object}")

        model_info["status"] = "loaded"

    except Exception as e_init:
        logger.error(f"YOLO-NAS model initialization failed: {e_init}")
        model_info["status"] = "failed"

else:
    logger.critical("Super Gradients not available - YOLO-NAS service will run in fallback mode")
    model_info["status"] = "super_gradients_unavailable"

# Model Functions - simplified, no advanced biomechanics

def detect_high_accuracy_poses(frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
    """Detect poses using optimized YOLO-NAS pose model"""
    if pose_model_info is None or pose_model_info[1] is None:
        logger.warning("YOLO-NAS pose model not loaded")
        return [[] for _ in frames]
    
    backend, model = pose_model_info

    all_poses = []
    batch_size = 8

    try:
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]

            # Use optimized inference if available, fallback to PyTorch
            results = None
            if backend in ["onnx", "tensorrt"]:
                try:
                    # Convert batch to numpy for ONNX/TensorRT
                    batch_np = np.array([frame for frame in batch])
                    results = model_optimizer.predict_optimized(pose_model_info, batch_np)
                except Exception as e:
                    logger.warning(f"Optimized inference failed, falling back to PyTorch: {e}")
                    with torch.no_grad():
                        if hasattr(model, 'predict'):
                            results = model.predict(batch, half=True)  # type: ignore
            else:
                # PyTorch inference
                with torch.no_grad():
                    if hasattr(model, 'predict'):
                        results = model.predict(batch, half=True)  # type: ignore

            batch_poses = []
            if results:
                for result in results:
                    frame_poses = []

                    # Extract keypoints for each person
                    if hasattr(result.prediction, 'poses') and len(result.prediction.poses) > 0:
                        for person_idx in range(len(result.prediction.poses)):
                            keypoints = {}
                            pose_data = result.prediction.poses[person_idx]

                        keypoint_names = [
                            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist", "left_hip", "right_hip",
                            "left_knee", "right_knee", "left_ankle", "right_ankle"
                        ]

                        for kpt_idx, (x, y, conf) in enumerate(pose_data):
                            if kpt_idx < len(keypoint_names) and conf > 0:
                                keypoints[keypoint_names[kpt_idx]] = {
                                    "x": float(x),
                                    "y": float(y),
                                    "confidence": float(conf)
                                }

                        # Calculate bounding box from keypoints
                        if keypoints:
                            valid_kpts = [(kp["x"], kp["y"]) for kp in keypoints.values()]
                            if valid_kpts:
                                x_coords, y_coords = zip(*valid_kpts)
                                x1, y1 = min(x_coords), min(y_coords)
                                x2, y2 = max(x_coords), max(y_coords)

                                frame_poses.append({
                                    "keypoints": keypoints,
                                    "confidence": float(np.mean([kp["confidence"] for kp in keypoints.values()])),
                                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                                })

                batch_poses.append(frame_poses)

            all_poses.extend(batch_poses)

    except Exception as e:
        logger.error(f"Error in YOLO-NAS pose detection: {e}")
        all_poses = [[] for _ in frames]

    return all_poses

def detect_high_accuracy_objects(frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
    """Detect objects using optimized YOLO-NAS object detection model"""
    if object_model_info is None or object_model_info[1] is None:
        logger.warning("YOLO-NAS object model not loaded")
        return [[] for _ in frames]
    
    backend, model = object_model_info

    all_objects = []
    batch_size = 8
    PADEL_CLASSES = {0: "person", 32: "sports ball", 38: "tennis racket"}

    try:
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]

            # Use optimized inference if available, fallback to PyTorch
            results = None
            if backend in ["onnx", "tensorrt"]:
                try:
                    # Convert batch to numpy for ONNX/TensorRT
                    batch_np = np.array([frame for frame in batch])
                    results = model_optimizer.predict_optimized(object_model_info, batch_np)
                except Exception as e:
                    logger.warning(f"Optimized inference failed, falling back to PyTorch: {e}")
                    with torch.no_grad():
                        if hasattr(model, 'predict'):
                            results = model.predict(batch, half=True)  # type: ignore
            else:
                # PyTorch inference
                with torch.no_grad():
                    if hasattr(model, 'predict'):
                        results = model.predict(batch, half=True)  # type: ignore

            batch_objects = []
            if results:
                for result in results:
                    frame_objects = []

                    # Extract bounding boxes and classes
                    if (hasattr(result.prediction, 'bboxes_xyxy') and
                        hasattr(result.prediction, 'labels') and
                        hasattr(result.prediction, 'confidence')):

                        bboxes = result.prediction.bboxes_xyxy
                        labels = result.prediction.labels
                        confidences = result.prediction.confidence

                        for bbox, label, conf in zip(bboxes, labels, confidences):
                            class_id = int(label)
                            confidence = float(conf)

                            # Filter for padel-relevant classes and confidence threshold
                            if class_id in PADEL_CLASSES and confidence > 0.3:
                                x1, y1, x2, y2 = bbox
                                frame_objects.append({
                                    "class": PADEL_CLASSES[class_id],
                                    "confidence": confidence,
                                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                                })

                batch_objects.append(frame_objects)

            all_objects.extend(batch_objects)

    except Exception as e:
        logger.error(f"Error in YOLO-NAS object detection: {e}")
        all_objects = [[] for _ in frames]

    return all_objects

# Drawing Functions
def draw_poses_on_frame(frame: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
    """Draw poses with skeleton connections"""
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
        keypoints = pose["keypoints"]

        # Draw bounding box
        if "bbox" in pose:
            bbox = pose["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints
        for keypoint_name, keypoint_data in keypoints.items():
            x, y = int(keypoint_data["x"]), int(keypoint_data["y"])
            confidence = keypoint_data["confidence"]

            if confidence > 0.5:
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                cv2.circle(annotated_frame, (x, y), 5, color, -1)

        # Draw connections
        for start_point_name, end_point_name in connections:
            if (start_point_name in keypoints and end_point_name in keypoints and
                keypoints[start_point_name]["confidence"] > 0.5 and
                keypoints[end_point_name]["confidence"] > 0.5):

                start_x, start_y = int(keypoints[start_point_name]["x"]), int(keypoints[start_point_name]["y"])
                end_x, end_y = int(keypoints[end_point_name]["x"]), int(keypoints[end_point_name]["y"])

                avg_confidence = (keypoints[start_point_name]["confidence"] + keypoints[end_point_name]["confidence"]) / 2
                color = (0, 255, int(255 * (1 - avg_confidence)))
                thickness = max(1, int(3 * avg_confidence))

                cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), color, thickness)

    return annotated_frame

def draw_objects_on_frame(frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    """Draw objects with bounding boxes"""
    annotated_frame = frame.copy()

    colors = {"person": (0, 255, 0), "sports ball": (0, 0, 255), "tennis racket": (255, 0, 0)}

    for obj in objects:
        class_name = obj["class"]
        confidence = obj["confidence"]
        bbox = obj["bbox"]

        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])

        color = colors.get(class_name, (255, 255, 255))

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

        try:
            stdout, stderr = process.communicate(timeout=120)
            if process.returncode == 0:
                logger.info("Video created successfully")
                return await upload_to_gcs(output_video_path)
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
    
    for model_key, model_config in model_configs.items():
        is_loaded = False
        backend = "none"
        
        if model_key == "yolo_nas_pose_n" and pose_model_info is not None:
            is_loaded = pose_model_info[1] is not None
            backend = pose_model_info[0] if pose_model_info else "none"
        elif model_key == "yolo_nas_s" and object_model_info is not None:
            is_loaded = object_model_info[1] is not None
            backend = object_model_info[0] if object_model_info else "none"
        elif model_key == "yolo_nas_m":
            is_loaded = False  # Not currently loaded
            backend = "none"
            
        models_status[model_key] = {
            "loaded": is_loaded,
            "enabled": model_config.get("enabled", False),
            "version": model_config.get("version", "unknown"),
            "model_name": model_config.get("model_name", "unknown"),
            "checkpoint": model_config.get("checkpoint"),
            "backend": backend,
            "task": model_config.get("task", "unknown"),
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
    optimization_config = current_config.get("optimization", {})
    performance_config = current_config.get("performance", {})
    
    # Overall health status - Allow graceful fallback when models are missing
    any_model_loaded = any(status["loaded"] for status in models_status.values())
    super_gradients_available = SUPER_GRADIENTS_AVAILABLE
    
    # Service is healthy if super-gradients is available (even if models will download on first request)
    overall_status = "healthy" if (any_model_loaded or super_gradients_available) else "unhealthy"
    status_code = 200 if (any_model_loaded or super_gradients_available) else 503
    
    response_data = {
        "status": overall_status,
        "service": service_info,
        "models": models_status,
        "features": active_features,
        "optimization": {
            "half_precision": optimization_config.get("use_half_precision", True),
            "batch_size": optimization_config.get("batch_size", 8),
            "fallback_enabled": optimization_config.get("enable_fallback", True),
            "preferred_backend": optimization_config.get("preferred_backend", "pytorch")
        },
        "performance": {
            "confidence_threshold": performance_config.get("confidence_threshold", 0.3),
            "max_concurrent_requests": performance_config.get("max_concurrent_requests", 4),
            "warmup_iterations": performance_config.get("warmup_iterations", 3)
        },
        "deployment": {
            "ready_for_upgrade": any_model_loaded,
            "config_hot_reload": True,
            "super_gradients_available": SUPER_GRADIENTS_AVAILABLE,
            "model_optimizer_available": True,
            "legacy_model_info": model_info,
            "last_config_check": datetime.now().isoformat()
        }
    }
    
    if overall_status == "unhealthy":
        return JSONResponse(content=response_data, status_code=status_code)
    
    return response_data

@app.post("/yolo-nas/pose")
async def yolo_nas_pose_detection(payload: VideoAnalysisURLRequest, request: Request):
    """
    YOLO-NAS High-Accuracy Pose Detection

    Uses YOLO-NAS architecture for high-precision pose estimation.
    Optimized for batch processing with half-precision on GPU.

    Returns 17-keypoint pose data with confidence scores and skeleton visualization.
    """
    logger.info("YOLO-NAS pose detection request received")

    if pose_model_info is None or pose_model_info[1] is None:
        raise HTTPException(status_code=503, detail="YOLO-NAS pose model not available")

    return await process_pose_detection(payload)

@app.post("/yolo-nas/object")
async def yolo_nas_object_detection(payload: VideoAnalysisURLRequest, request: Request):
    """
    YOLO-NAS High-Accuracy Object Detection

    Uses YOLO-NAS architecture for high-precision object detection.
    Detects padel-specific objects: person, sports ball, tennis racket.

    Returns bounding box data with confidence scores and annotated visualization.
    """
    logger.info("YOLO-NAS object detection request received")

    if object_model_info is None or object_model_info[1] is None:
        raise HTTPException(status_code=503, detail="YOLO-NAS object model not available")

    return await process_object_detection(payload)

async def process_pose_detection(payload: VideoAnalysisURLRequest):
    """Common pose detection processing"""
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

        # Process poses
        all_poses = detect_high_accuracy_poses(frames)
        annotated_frames = []

        if payload.video:
            for i, frame in enumerate(frames):
                if i < len(all_poses):
                    annotated_frames.append(draw_poses_on_frame(frame, all_poses[i]))

        # Prepare response
        response_data = {}
        if payload.data:
            response_data["data"] = {"poses_per_frame": all_poses}

        if payload.video:
            video_url = await create_video_from_frames(annotated_frames, video_info)
            response_data["video_url"] = video_url

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO-NAS pose error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"YOLO-NAS pose error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)

async def process_object_detection(payload: VideoAnalysisURLRequest):
    """Common object detection processing"""
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

        # Process objects
        all_objects = detect_high_accuracy_objects(frames)
        annotated_frames = []

        if payload.video:
            for i, frame in enumerate(frames):
                if i < len(all_objects):
                    annotated_frames.append(draw_objects_on_frame(frame, all_objects[i]))

        # Prepare response
        response_data = {}
        if payload.data:
            response_data["data"] = {"objects_per_frame": all_objects}

        if payload.video:
            video_url = await create_video_from_frames(annotated_frames, video_info)
            response_data["video_url"] = video_url

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO-NAS object error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"YOLO-NAS object error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)


# Advanced biomechanics endpoints removed - keeping only basic pose and object detection

if __name__ == "__main__":
    logger.info("Starting YOLO-NAS service on port 8004")
    if pose_model_info is None and object_model_info is None:
        logger.warning("No models loaded - service starting in fallback mode")
    else:
        logger.info("YOLO-NAS service starting with loaded models")
    uvicorn.run(app, host="0.0.0.0", port=8004, log_config=None)