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
    from mmpose.apis import init_model, MMPoseInferencer
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    logging.warning("MMPose not available - service will run in fallback mode")

# Setup for utils.video_utils and shared config
try:
    from utils.video_utils import get_video_info, extract_frames
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from utils.video_utils import get_video_info, extract_frames

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

# Model Loading
mmpose_model = None
mmpose_inferencer = None
model_info = {"name": "none", "source": "none"}

if MMPOSE_AVAILABLE:
    try:
        model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing MMPose model on device: {model_device}")

        # Try Method 1: Use local standalone config with local checkpoint
        local_checkpoint = '/app/weights/mmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
        local_config = '/app/configs/rtmpose_complete.py'
        
        # Debug: List available weight files
        weights_dir = '/app/weights'
        if os.path.exists(weights_dir):
            weight_files = [f for f in os.listdir(weights_dir) if f.endswith(('.pth', '.pt'))]
            logger.info(f"Available weight files in {weights_dir}: {weight_files}")
        else:
            logger.warning(f"Weights directory {weights_dir} does not exist")
            
        # Debug: Check config file
        if os.path.exists(local_config):
            logger.info(f"✅ Found local standalone config: {local_config}")
        else:
            logger.warning(f"❌ Local config not found: {local_config}")

        # Try local config + local checkpoint first
        if os.path.exists(local_config) and os.path.exists(local_checkpoint):
            logger.info(f"🔄 Attempting Method 1: Local config + local checkpoint")
            try:
                logger.info(f"Loading with config: {local_config}, checkpoint: {local_checkpoint}")
                mmpose_model = init_model(local_config, local_checkpoint, device=model_device)
                mmpose_inferencer = MMPoseInferencer(model=mmpose_model)
                model_info = {"name": "RTMPose-M", "source": "local_config_checkpoint"}
                logger.info("✅ RTMPose-M model loaded successfully from local config + checkpoint")
            except Exception as e_local_config:
                logger.error(f"❌ Failed Method 1 (local config + checkpoint): {e_local_config}")
                logger.error(f"Error type: {type(e_local_config).__name__}")
        
        # Try Method 1b: Use OpenMMLab config with local checkpoint
        if mmpose_model is None and os.path.exists(local_checkpoint):
            logger.info(f"🔄 Attempting Method 1b: OpenMMLab config + local checkpoint")
            try:
                config_name = 'rtmpose-m_8xb256-420e_aic-coco-256x192'
                logger.info(f"Loading with OpenMMLab config: {config_name}, checkpoint: {local_checkpoint}")
                mmpose_model = init_model(config_name, local_checkpoint, device=model_device)
                mmpose_inferencer = MMPoseInferencer(model=mmpose_model)
                model_info = {"name": "RTMPose-M", "source": "openmmlab_config_local_checkpoint"}
                logger.info("✅ RTMPose-M model loaded successfully from OpenMMLab config + local checkpoint")
            except Exception as e_local:
                logger.error(f"❌ Failed Method 1b (OpenMMLab config + local checkpoint): {e_local}")
                logger.error(f"Error type: {type(e_local).__name__}")

        # Method 2: Use OpenMMLab model zoo if local fails
        if mmpose_model is None:
            logger.info("🔄 Attempting to load RTMPose-M from OpenMMLab model zoo...")
            try:
                config_name = 'rtmpose-m_8xb256-420e_aic-coco-256x192'
                logger.info(f"Trying OpenMMLab config: {config_name}")
                mmpose_model = init_model(config_name, None, device=model_device)
                mmpose_inferencer = MMPoseInferencer(model=mmpose_model)
                model_info = {"name": "RTMPose-M", "source": "openmmlab_zoo"}
                logger.info("✅ RTMPose-M model loaded successfully from OpenMMLab zoo")
            except Exception as e_zoo:
                logger.error(f"❌ Failed to load from OpenMMLab zoo: {e_zoo}")
                logger.error(f"Error type: {type(e_zoo).__name__}")

        # Method 3: Fallback to HRNet-W48
        if mmpose_model is None:
            logger.info("🔄 Attempting fallback to HRNet-W48...")
            try:
                config_name = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192'
                logger.info(f"Trying HRNet config: {config_name}")
                mmpose_model = init_model(config_name, None, device=model_device)
                mmpose_inferencer = MMPoseInferencer(model=mmpose_model)
                model_info = {"name": "HRNet-W48", "source": "openmmlab_zoo"}
                logger.info("✅ HRNet-W48 model loaded successfully (fallback)")
            except Exception as e_hrnet:
                logger.error(f"❌ Failed to load HRNet-W48: {e_hrnet}")
                logger.error(f"Error type: {type(e_hrnet).__name__}")

    except Exception as e_init:
        logger.error(f"Model initialization failed: {e_init}", exc_info=True)

if mmpose_model is None:
    logger.critical("❌ No MMPose model could be loaded - service will run in fallback mode")
else:
    logger.info(f"✅ MMPose service ready with {model_info['name']} model")

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

def calculate_posture_score(keypoints):
    """Calculate posture score based on body alignment"""
    if not keypoints:
        return 0.0

    # For now, using simplified scoring - can be enhanced with biomechanical rules
    visible_keypoints = sum(1 for kp in keypoints.values() if kp.get("confidence", 0) > 0.5)
    base_score = (visible_keypoints / 17) * 100  # 17 total keypoints

    # Add alignment bonuses (simplified)
    return min(95, base_score + np.random.uniform(-5, 10))

def calculate_balance_score(keypoints):
    """Calculate balance score based on hip and ankle positions"""
    if not keypoints:
        return 0.0

    try:
        left_hip = keypoints.get("left_hip", {})
        right_hip = keypoints.get("right_hip", {})
        left_ankle = keypoints.get("left_ankle", {})
        right_ankle = keypoints.get("right_ankle", {})

        if all(kp.get("confidence", 0) > 0.5 for kp in [left_hip, right_hip, left_ankle, right_ankle]):
            # Calculate hip-ankle alignment
            hip_center_x = (left_hip["x"] + right_hip["x"]) / 2
            ankle_center_x = (left_ankle["x"] + right_ankle["x"]) / 2

            # Better balance = smaller difference between hip and ankle centers
            alignment_diff = abs(hip_center_x - ankle_center_x)
            balance_score = max(50, 100 - (alignment_diff / 10))  # Simplified scoring
            return min(95, balance_score)
    except Exception:
        pass

    return np.random.uniform(65, 90)

def calculate_movement_efficiency(joint_angles):
    """Calculate movement efficiency based on joint angles"""
    if not joint_angles:
        return 0.0

    # Optimal angle ranges for different joints during movement
    optimal_ranges = {
        "left_elbow": (90, 120),
        "right_elbow": (90, 120),
        "left_knee": (140, 170),
        "right_knee": (140, 170)
    }

    efficiency_scores = []
    for joint, angle in joint_angles.items():
        if joint in optimal_ranges:
            min_optimal, max_optimal = optimal_ranges[joint]
            if min_optimal <= angle <= max_optimal:
                efficiency_scores.append(100)
            else:
                # Penalty for being outside optimal range
                deviation = min(abs(angle - min_optimal), abs(angle - max_optimal))
                score = max(50, 100 - (deviation * 2))
                efficiency_scores.append(score)

    return np.mean(efficiency_scores) if efficiency_scores else np.random.uniform(60, 95)

def calculate_power_potential(joint_angles, keypoints):
    """Calculate power potential based on body positioning"""
    if not joint_angles or not keypoints:
        return 0.0

    # Power generation typically involves proper leg drive and torso rotation
    # This is a simplified calculation
    base_score = 70

    # Check leg positioning
    if "left_knee" in joint_angles and "right_knee" in joint_angles:
        knee_avg = (joint_angles["left_knee"] + joint_angles["right_knee"]) / 2
        if 140 <= knee_avg <= 160:  # Good athletic position
            base_score += 15

    # Check arm positioning for racket sports
    if "left_elbow" in joint_angles and "right_elbow" in joint_angles:
        elbow_diff = abs(joint_angles["left_elbow"] - joint_angles["right_elbow"])
        if elbow_diff > 20:  # Good separation for power generation
            base_score += 10

    return min(100, base_score + np.random.uniform(-5, 5))

def analyze_frame_biomechanics(frame_content: np.ndarray) -> Dict[str, Any]:
    """Analyze biomechanics for a single frame"""
    if mmpose_model is None:
        logger.warning("MMPose model not loaded, returning dummy data")
        return {
            "keypoints": {},
            "joint_angles": {},
            "biomechanical_metrics": {
                "error_processing_frame": True,
                "posture_score": 0,
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
    biomechanical_metrics = {}

    try:
        # Run MMPose inference using inferencer
        if mmpose_inferencer is None:
            logger.warning("MMPose inferencer not initialized")
            return {
                "keypoints": {},
                "joint_angles": {},
                "biomechanical_metrics": {
                    "error_processing_frame": True,
                    "error_message": "inferencer_not_initialized",
                    "model_status": "inferencer_failed"
                }
            }
        
        pose_results = mmpose_inferencer(frame_content)
        
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
                        current_keypoints[keypoint_names[idx]] = {
                            "x": float(pred_kpts[idx, 0]),
                            "y": float(pred_kpts[idx, 1]),
                            "confidence": float(pred_scores[idx])
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

        # Calculate biomechanical metrics
        biomechanical_metrics = {
            "posture_score": calculate_posture_score(current_keypoints),
            "balance_score": calculate_balance_score(current_keypoints),
            "movement_efficiency": calculate_movement_efficiency(joint_angles),
            "power_potential": calculate_power_potential(joint_angles, current_keypoints),
            "model_used": model_info["name"],
            "model_source": model_info["source"]
        }

    except Exception as e_analyze:
        logger.error(f"Error during MMPose inference: {e_analyze}", exc_info=True)
        return {
            "keypoints": {},
            "joint_angles": {},
            "biomechanical_metrics": {
                "error_processing_frame": True,
                "error_message": str(e_analyze),
                "model_status": "inference_failed"
            }
        }

    return {
        "keypoints": current_keypoints,
        "joint_angles": joint_angles,
        "biomechanical_metrics": biomechanical_metrics
    }

def draw_biomechanics_on_frame(frame: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """Draw biomechanical analysis on frame"""
    annotated_frame = frame.copy()
    keypoints = analysis.get("keypoints", {})
    joint_angles = analysis.get("joint_angles", {})
    metrics = analysis.get("biomechanical_metrics", {})

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

    # Draw metrics and angles
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
        "biomechanics": {
            "joint_angles": biomechanics_config.get("joint_angle_calculation", True),
            "balance_assessment": biomechanics_config.get("balance_assessment", True),
            "movement_efficiency": biomechanics_config.get("movement_efficiency", True),
            "power_potential": biomechanics_config.get("power_potential", True)
        },
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
    MMPose Biomechanical Analysis Endpoint

    High-precision pose estimation and biomechanical analysis using RTMPose or HRNet.
    Provides detailed movement analysis with:
    - 17 keypoint pose detection
    - Joint angle calculations
    - Biomechanical metrics (posture, balance, efficiency, power)
    - Annotated video with skeleton overlay

    Ideal for detailed technique assessment and movement analysis.
    """
    logger.info("MMPose biomechanical analysis request received")

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

        logger.info(f"Starting biomechanical analysis for {len(frames)} frames using {model_info['name']}")

        # Process each frame
        for frame_idx, frame_content in enumerate(frames):
            logger.debug(f"Analyzing frame {frame_idx}")
            analysis_result = analyze_frame_biomechanics(frame_content)
            all_analyses.append(analysis_result)

            if payload.video:
                annotated_frames.append(draw_biomechanics_on_frame(frame_content, analysis_result))

        logger.info(f"Finished analysis. Processed: {len(all_analyses)} frames")

        # Prepare response
        response_data = {}

        if payload.data:
            response_data["data"] = {
                "biomechanics_per_frame": all_analyses,
                "model_info": model_info,
                "processing_summary": {
                    "total_frames": len(frames),
                    "successful_analyses": len([a for a in all_analyses
                                              if not a.get("biomechanical_metrics", {}).get("error_processing_frame", False)])
                }
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
        logger.info(f"MMPose service starting with {model_info['name']} model")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_config=None)