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

# MMPose ViTPose++ imports
try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.utils import register_all_modules
    register_all_modules()
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    logging.warning("MMPose not available - service will run in fallback mode")

# Setup for shared config
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
        def get_service_info(self): return {"service": "vitpose", "version": "1.0.0"}
    def merge_env_overrides(config): return config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

# Model Loading
vitpose_model = None
model_info = {"name": "none", "source": "none"}

if MMPOSE_AVAILABLE:
    try:
        # Check configuration to see if models are enabled
        models_config = service_config.get("models", {})
        vitpose_enabled = models_config.get("vitpose_base", {}).get("enabled", True)
        hrnet_enabled = models_config.get("hrnet_w48", {}).get("enabled", True)
        
        logger.info(f"Model configuration - ViTPose enabled: {vitpose_enabled}, HRNet enabled: {hrnet_enabled}")
        
        if not vitpose_enabled and not hrnet_enabled:
            logger.info("🔕 All models disabled in configuration - running in fallback mode")
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing ViTPose++ model on device: {device}")
            
            # Try efficient ViTPose-Base checkpoint for optimal performance
            local_checkpoint = '/app/weights/vitpose/vitpose_base_coco_256x192.pth'
            
            # Method 1: Try efficient ViTPose-Base config (only if enabled)
            if vitpose_enabled and os.path.exists(local_checkpoint):
                logger.info(f"🔄 Attempting Method 1: Efficient ViTPose-Base checkpoint")
                try:
                    config_name = 'td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
                    logger.info(f"Loading with config: {config_name}, checkpoint: {local_checkpoint}")
                    vitpose_model = init_model(config_name, local_checkpoint, device=device)
                    
                    # Enable FP16 if CUDA available for VRAM efficiency
                    if torch.cuda.is_available():
                        vitpose_model.half()
                        logger.info("Enabled FP16 precision for VRAM efficiency")
                    
                    model_info = {"name": "ViTPose-Base", "source": "local_checkpoint", "precision": "fp16" if torch.cuda.is_available() else "fp32", "variant": "Efficient"}
                    logger.info("✅ Efficient ViTPose-Base model loaded successfully from local checkpoint")
                except Exception as e_local:
                    logger.error(f"❌ Failed Method 1 (efficient ViTPose-Base): {e_local}")
            elif not vitpose_enabled:
                logger.info("🔕 ViTPose-Base model disabled in configuration, skipping")
            
            # Method 2: Use MMPose model zoo if local fails (only if enabled)
            if vitpose_enabled and vitpose_model is None:
                logger.info("🔄 Attempting Method 2: MMPose model zoo download")
                try:
                    config_name = 'td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
                    logger.info(f"Loading from model zoo: {config_name}")
                    vitpose_model = init_model(config_name, None, device=device)
                    
                    # Enable FP16 if CUDA available
                    if torch.cuda.is_available():
                        vitpose_model.half()
                        logger.info("Enabled FP16 precision for VRAM efficiency")
                    
                    model_info = {"name": "ViTPose-Base", "source": "mmpose_zoo", "precision": "fp16" if torch.cuda.is_available() else "fp32"}
                    logger.info("✅ ViTPose++ model loaded successfully from model zoo")
                except Exception as e_zoo:
                    logger.error(f"❌ Failed Method 2 (model zoo): {e_zoo}")
            
            # Method 3: Fallback to HRNet if ViTPose fails (only if enabled)
            if hrnet_enabled and vitpose_model is None:
                logger.info("🔄 Attempting Method 3: HRNet fallback")
                try:
                    config_name = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192'
                    logger.info(f"Loading HRNet fallback: {config_name}")
                    vitpose_model = init_model(config_name, None, device=device)
                    
                    if torch.cuda.is_available():
                        vitpose_model.half()
                        logger.info("Enabled FP16 precision for VRAM efficiency")
                    
                    model_info = {"name": "HRNet-W48", "source": "mmpose_zoo_fallback", "precision": "fp16" if torch.cuda.is_available() else "fp32"}
                    logger.info("✅ HRNet model loaded successfully (fallback)")
                except Exception as e_hrnet:
                    logger.error(f"❌ Failed Method 3 (HRNet fallback): {e_hrnet}")
            elif not hrnet_enabled:
                logger.info("🔕 HRNet model disabled in configuration, skipping")

    except Exception as e_init:
        logger.error(f"ViTPose++ model initialization failed: {e_init}", exc_info=True)

if vitpose_model is None:
    logger.critical("❌ No ViTPose++ model could be loaded - service will run in fallback mode")
else:
    logger.info(f"✅ ViTPose++ service ready with {model_info['name']} model")

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

def calculate_joint_angles(keypoints):
    """Calculate joint angles for biomechanical analysis"""
    angles = {}
    
    # Define joint angle calculations
    angle_definitions = {
        "left_elbow": (["left_shoulder", "left_elbow", "left_wrist"]),
        "right_elbow": (["right_shoulder", "right_elbow", "right_wrist"]),
        "left_knee": (["left_hip", "left_knee", "left_ankle"]),
        "right_knee": (["right_hip", "right_knee", "right_ankle"]),
        "left_shoulder": (["left_elbow", "left_shoulder", "left_hip"]),
        "right_shoulder": (["right_elbow", "right_shoulder", "right_hip"]),
        "hip_angle": (["left_hip", "right_hip", "left_knee"]),
        "spine_angle": (["left_shoulder", "right_shoulder", "left_hip"])
    }
    
    for angle_name, joint_names in angle_definitions.items():
        if all(joint in keypoints for joint in joint_names):
            try:
                points = [(keypoints[joint]["x"], keypoints[joint]["y"]) for joint in joint_names]
                angles[angle_name] = calculate_angle(points[0], points[1], points[2])
            except Exception:
                angles[angle_name] = 0.0
    
    return angles

def assess_balance(angles):
    """Assess player balance and stability based on joint angles"""
    if not angles:
        return {"score": 0.0, "status": "insufficient_data"}
    
    balance_score = 70.0  # Base score
    status = "stable"
    
    # Check knee alignment
    if "left_knee" in angles and "right_knee" in angles:
        knee_diff = abs(angles["left_knee"] - angles["right_knee"])
        if knee_diff < 10:
            balance_score += 15
        elif knee_diff > 30:
            balance_score -= 10
            status = "unstable"
    
    # Check hip alignment
    if "hip_angle" in angles:
        hip_angle = angles["hip_angle"]
        if 160 <= hip_angle <= 180:
            balance_score += 10
        elif hip_angle < 140 or hip_angle > 200:
            balance_score -= 15
            status = "poor_alignment"
    
    # Check shoulder symmetry
    if "left_shoulder" in angles and "right_shoulder" in angles:
        shoulder_diff = abs(angles["left_shoulder"] - angles["right_shoulder"])
        if shoulder_diff < 15:
            balance_score += 5
        elif shoulder_diff > 40:
            balance_score -= 5
    
    balance_score = max(0.0, min(100.0, balance_score))
    
    return {
        "score": balance_score,
        "status": status,
        "metrics": {
            "knee_alignment": angles.get("left_knee", 0) - angles.get("right_knee", 0),
            "hip_stability": angles.get("hip_angle", 0),
            "shoulder_symmetry": abs(angles.get("left_shoulder", 0) - angles.get("right_shoulder", 0))
        }
    }

def analyze_biomechanics(keypoints):
    """Enhanced biomechanical analysis with joint angles and balance assessment"""
    angles = calculate_joint_angles(keypoints)
    balance = assess_balance(angles)
    
    # Movement efficiency analysis
    efficiency_score = 75.0
    if angles:
        # Optimal ranges for athletic movement
        optimal_ranges = {
            "left_knee": (140, 170),
            "right_knee": (140, 170),
            "left_elbow": (90, 150),
            "right_elbow": (90, 150)
        }
        
        efficiency_scores = []
        for joint, (min_opt, max_opt) in optimal_ranges.items():
            if joint in angles:
                angle = angles[joint]
                if min_opt <= angle <= max_opt:
                    efficiency_scores.append(100)
                else:
                    deviation = min(abs(angle - min_opt), abs(angle - max_opt))
                    score = max(50, 100 - (deviation * 1.5))
                    efficiency_scores.append(score)
        
        if efficiency_scores:
            efficiency_score = np.mean(efficiency_scores)
    
    # Power generation potential
    power_score = 60.0
    if "left_knee" in angles and "right_knee" in angles:
        avg_knee = (angles["left_knee"] + angles["right_knee"]) / 2
        if 145 <= avg_knee <= 165:  # Optimal power position
            power_score += 25
    
    if "hip_angle" in angles:
        if 165 <= angles["hip_angle"] <= 180:  # Good hip extension
            power_score += 15
    
    return {
        "angles": angles,
        "balance": balance,
        "movement_efficiency": min(100.0, efficiency_score),
        "power_potential": min(100.0, power_score),
        "stability_metrics": {
            "overall_stability": balance["score"],
            "postural_control": min(100.0, (balance["score"] + efficiency_score) / 2),
            "athletic_readiness": min(100.0, (efficiency_score + power_score) / 2)
        }
    }

def calculate_pose_quality_score(keypoints):
    """Calculate pose quality score based on keypoint visibility and positioning"""
    if not keypoints:
        return 0.0

    visible_keypoints = sum(1 for kp in keypoints.values() if kp.get("confidence", 0) > 0.5)
    base_score = (visible_keypoints / 17) * 100  # 17 total COCO keypoints
    
    # Add quality bonuses based on pose completeness
    quality_bonus = 0
    
    # Check if key joints are visible
    key_joints = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    visible_key_joints = sum(1 for joint in key_joints if keypoints.get(joint, {}).get("confidence", 0) > 0.5)
    quality_bonus += (visible_key_joints / len(key_joints)) * 20
    
    return min(100, base_score + quality_bonus)

def analyze_frame_pose(frame_content: np.ndarray, confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """Analyze pose for a single frame using ViTPose++"""
    if vitpose_model is None:
        logger.warning("ViTPose++ model not loaded, returning dummy data")
        return {
            "keypoints": {},
            "joint_angles": {},
            "pose_metrics": {
                "error_processing_frame": True,
                "model_status": "not_loaded"
            }
        }

    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    current_keypoints = {}
    joint_angles = {}
    pose_metrics = {}

    try:
        # Run ViTPose++ inference with FP16
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    pose_data_samples = inference_topdown(vitpose_model, frame_content)
            else:
                pose_data_samples = inference_topdown(vitpose_model, frame_content)

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

        # Calculate joint angles
        angle_definitions = [
            ("left_elbow", ["left_shoulder", "left_elbow", "left_wrist"]),
            ("right_elbow", ["right_shoulder", "right_elbow", "right_wrist"]),
            ("left_knee", ["left_hip", "left_knee", "left_ankle"]),
            ("right_knee", ["right_hip", "right_knee", "right_ankle"]),
            ("left_shoulder", ["left_elbow", "left_shoulder", "left_hip"]),
            ("right_shoulder", ["right_elbow", "right_shoulder", "right_hip"])
        ]

        for angle_name, joint_names in angle_definitions:
            if all(joint in current_keypoints for joint in joint_names):
                points = [(current_keypoints[joint]["x"], current_keypoints[joint]["y"])
                         for joint in joint_names]
                joint_angles[angle_name] = calculate_angle(points[0], points[1], points[2])

        # Calculate enhanced biomechanical metrics
        biomechanical_analysis = analyze_biomechanics(current_keypoints)
        
        pose_metrics = {
            "pose_quality_score": calculate_pose_quality_score(current_keypoints),
            "visible_keypoints": len(current_keypoints),
            "total_keypoints": 17,
            "confidence_threshold": confidence_threshold,
            "model_used": model_info["name"],
            "model_precision": model_info.get("precision", "fp32"),
            "biomechanical_insights": {
                "movement_efficiency": biomechanical_analysis["movement_efficiency"],
                "power_potential": biomechanical_analysis["power_potential"],
                "balance_score": biomechanical_analysis["balance"]["score"],
                "balance_status": biomechanical_analysis["balance"]["status"],
                "stability_metrics": biomechanical_analysis["stability_metrics"]
            }
        }
        
        # Include joint angles in the analysis
        joint_angles.update(biomechanical_analysis["angles"])

    except Exception as e_analyze:
        logger.error(f"Error during ViTPose++ inference: {e_analyze}", exc_info=True)
        return {
            "keypoints": {},
            "joint_angles": {},
            "pose_metrics": {
                "error_processing_frame": True,
                "error_message": str(e_analyze),
                "model_status": "inference_failed"
            }
        }
    finally:
        # Clean up GPU memory after inference
        cleanup_gpu_memory()

    return {
        "keypoints": current_keypoints,
        "joint_angles": joint_angles,
        "pose_metrics": pose_metrics
    }

def draw_pose_on_frame(frame: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """Draw pose estimation on frame"""
    annotated_frame = frame.copy()
    keypoints = analysis.get("keypoints", {})
    joint_angles = analysis.get("joint_angles", {})
    metrics = analysis.get("pose_metrics", {})

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

    # Draw metrics
    y_offset = 30
    for name, val in metrics.items():
        if isinstance(val, (int, float)):
            text = f"{name}: {val:.1f}"
        else:
            text = f"{name}: {val}"
        cv2.putText(annotated_frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 20

    # Draw joint angles
    for angle_name, angle_val in joint_angles.items():
        text = f"{angle_name}: {angle_val:.1f}°"
        cv2.putText(annotated_frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 18

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
                if process.poll() is None:
                    try:
                        process.stdin.write(frame.tobytes())
                    except (IOError, BrokenPipeError):
                        logger.warning("FFmpeg stdin pipe broken, stopping frame writing")
                        break
                else:
                    logger.warning("FFmpeg process terminated early")
                    break
            
            if process.stdin and not process.stdin.closed:
                process.stdin.close()

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

        logger.info(f"Starting ViTPose++ analysis for {len(frames)} frames using {model_info['name']}")

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
                    "successful_analyses": len([a for a in all_analyses
                                              if not a.get("pose_metrics", {}).get("error_processing_frame", False)]),
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
        logger.info(f"ViTPose++ service starting with {model_info['name']} model")
    uvicorn.run(app, host="0.0.0.0", port=8006, log_config=None)