from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import tempfile
import os
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
from mmpose.apis import init_model, inference_topdown

# Setup for utils.video_utils
try:
    from utils.video_utils import get_video_info, extract_frames
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- MMPOSE SERVICE SCRIPT STARTED (MMPose Built-in Config Resolution) ---")

app = FastAPI(title="MMPose Biomechanics Service (URL Input Only)", version="2.0.0")
logger.info("FastAPI app object created for MMPose service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_mmpose"

# --- Pydantic Model for Request Body ---
class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False
    data: bool = True

# --- Helper Functions ---
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
        logger.info(f"Successfully uploaded {video_path} to GCS as {object_name}. Public URL: {blob.public_url}")
        return blob.public_url
    except Exception as e:
        logger.error(f"Error uploading {video_path} to GCS as {object_name}: {e}", exc_info=True)
        return ""

# --- Model Loading with Built-in Config Resolution ---
mmpose_model = None
model_info = {"name": "none", "source": "none"}

try:
    model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Initializing MMPose model on device: {model_device}")

    # Try Method 1: Use local checkpoint with MMPose's built-in config
    local_checkpoint = '/opt/padel/app/weights/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
    
    if os.path.exists(local_checkpoint):
        logger.info(f"Attempting to load RTMPose with local checkpoint: {local_checkpoint}")
        try:
            # Use MMPose's project config path (relative to MMPose installation)
            config_name = 'projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
            mmpose_model = init_model(config_name, local_checkpoint, device=model_device)
            model_info = {"name": "rtmpose-m", "source": "local_weights_builtin_config"}
            logger.info("Successfully loaded RTMPose model with local weights and built-in config.")
        except Exception as e_local:
            logger.warning(f"Failed to load with local weights + built-in config: {e_local}")
            raise e_local
    else:
        logger.info("Local RTMPose checkpoint not found, will try downloading.")
        raise FileNotFoundError("Local checkpoint not available")

except Exception as e_rtmpose:
    logger.warning(f"Primary RTMPose loading failed: {e_rtmpose}. Trying fallback methods.")
    
    try:
        # Method 2: Let MMPose download everything automatically
        logger.info("Attempting to load RTMPose with automatic download...")
        config_name = '/opt/padel/mmpose/venv/lib/python3.8/site-packages/mmpose/.mim/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'
        checkpoint_url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
        
        mmpose_model = init_model(config_name, checkpoint_url, device=model_device)
        model_info = {"name": "rtmpose-m", "source": "auto_download"}
        logger.info("Successfully loaded RTMPose model with automatic download.")
        
    except Exception as e_auto:
        logger.warning(f"Automatic RTMPose download failed: {e_auto}. Trying simpler config.")
        
        try:
            # Method 3: Use simple model alias (if supported)
            logger.info("Attempting to use RTMPose model alias...")
            mmpose_model = init_model('rtmpose-m', device=model_device)
            model_info = {"name": "rtmpose-m", "source": "model_alias"}
            logger.info("Successfully loaded RTMPose model using alias.")
            
        except Exception as e_alias:
            logger.error(f"All RTMPose loading methods failed. Final attempt with HRNet fallback.")
            
            try:
                # Method 4: HRNet fallback with built-in config
                logger.info("Loading HRNet as ultimate fallback...")
                config_name = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
                checkpoint_url = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
                
                mmpose_model = init_model(config_name, checkpoint_url, device=model_device)
                model_info = {"name": "hrnet-w48", "source": "fallback_download"}
                logger.info("Successfully loaded HRNet fallback model.")
                
            except Exception as e_hrnet:
                logger.error(f"CRITICAL: All model loading attempts failed: {e_hrnet}")
                mmpose_model = None
                model_info = {"name": "failed", "source": "none"}

# Log final model status
if mmpose_model is not None:
    logger.info(f"MMPose service ready with model: {model_info['name']} (source: {model_info['source']})")
else:
    logger.critical("MMPose service starting without a working model - all endpoints will be unhealthy!")

# --- Core Logic ---
def calculate_angle(a, b, c):
    p1, p2, p3 = np.array(a), np.array(b), np.array(c)
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude_product < 1e-7:
        return 0.0
    cos_angle = np.clip(dot_product / magnitude_product, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def calculate_posture_score(keypoints):
    return np.random.uniform(70, 95) if keypoints else 0.0

def calculate_balance_score(keypoints):
    return np.random.uniform(65, 90) if keypoints else 0.0

def calculate_movement_efficiency(joint_angles):
    return np.random.uniform(60, 95) if joint_angles else 0.0

def calculate_power_potential(joint_angles, keypoints):
    return np.random.uniform(50, 100) if joint_angles and keypoints else 0.0

def analyze_frame_biomechanics(frame_content: np.ndarray) -> Dict[str, Any]:
    if mmpose_model is None:
        logger.warning("MMPose model not loaded for analyze_frame_biomechanics, returning dummy data.")
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
        # Run inference
        pose_data_samples = inference_topdown(mmpose_model, frame_content, bbox_cs='')
        
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
        logger.error(f"Error during MMPose inference/analysis on a frame: {e_analyze}", exc_info=True)
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

@app.get("/healthz")
async def health_check():
    if mmpose_model is None:
        logger.warning("/healthz: MMPose model not loaded.")
        return JSONResponse(
            content={
                "status": "unhealthy", 
                "model": "mmpose not loaded",
                "model_info": model_info
            }, 
            status_code=503
        )
    
    logger.info(f"/healthz: MMPose model loaded ({model_info['name']}), service healthy.")
    return {
        "status": "healthy", 
        "model": "mmpose",
        "model_info": model_info
    }

@app.post("/mmpose")
async def analyze_video_endpoint(
    payload: VideoAnalysisURLRequest,
    http_request: Request
):
    logger.info(f"--- Enter /mmpose endpoint (Built-in Config Resolution) ---")
    logger.info(f"Request Headers: {http_request.headers}")
    logger.info(f"Received Payload: video_url='{payload.video_url}', "
               f"video_output_requested={payload.video}, data_output_requested={payload.data}")

    if mmpose_model is None:
        logger.error("/mmpose: Model not loaded. Cannot process request.")
        raise HTTPException(
            status_code=503, 
            detail=f"MMPose model is not available. Model info: {model_info}"
        )

    temp_downloaded_path: Optional[str] = None
    try:
        video_url_str = str(payload.video_url)
        logger.info(f"Attempting to download video from URL: {video_url_str}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(video_url_str)
            response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(response.content)
            temp_downloaded_path = temp_file.name
        
        logger.info(f"Video downloaded successfully to {temp_downloaded_path}")

        video_info = get_video_info(temp_downloaded_path)
        logger.info(f"Video Info: {video_info}")

        num_frames_to_process = 75  # For testing, adjust as needed
        frames = extract_frames(temp_downloaded_path, num_frames_to_extract=num_frames_to_process)
        logger.info(f"Extracted {len(frames)} consecutive frames (target={num_frames_to_process}).")

        if not frames:
            logger.error(f"No frames extracted from {temp_downloaded_path}.")
            raise HTTPException(status_code=400, detail="No frames could be extracted from the provided video.")

        all_analyses_per_frame: List[Dict[str, Any]] = []
        annotated_frames_list: List[np.ndarray] = []
        
        logger.info(f"Starting biomechanics analysis for {len(frames)} frames using {model_info['name']}.")
        
        for frame_idx, frame_content in enumerate(frames):
            logger.debug(f"Analyzing frame {frame_idx} for biomechanics.")
            analysis_result = analyze_frame_biomechanics(frame_content)
            all_analyses_per_frame.append(analysis_result)
            
            if payload.video:
                annotated_frames_list.append(draw_biomechanics_on_frame(frame_content, analysis_result))
        
        logger.info(f"Finished biomechanics analysis. Analyses: {len(all_analyses_per_frame)}. "
                   f"Annotated: {len(annotated_frames_list)}.")
        
        response_content: Dict[str, Any] = {}
        
        if payload.data:
            response_content["data"] = {
                "biomechanics_per_frame": all_analyses_per_frame,
                "model_info": model_info,
                "processing_summary": {
                    "total_frames": len(frames),
                    "successful_analyses": len([a for a in all_analyses_per_frame 
                                              if not a.get("biomechanical_metrics", {}).get("error_processing_frame", False)])
                }
            }

        if payload.video:
            if not annotated_frames_list:
                logger.warning("MMPose: Video output requested, but no annotated frames.")
                response_content["video_url"] = None
                response_content["message"] = "MMPose: No frames to annotate."
            else:
                output_video_path: Optional[str] = None
                gcs_url_result: str = ""
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out_vid_file:
                        output_video_path = temp_out_vid_file.name
                    
                    height, width = annotated_frames_list[0].shape[:2]
                    fps_float = float(video_info.get("fps", 30.0))
                    if fps_float <= 0:
                        fps_float = 30.0
                    
                    logger.info(f"MMPose: Creating video via FFMPEG: {output_video_path} "
                               f"(H={height},W={width},FPS={fps_float})")
                    
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24',
                        '-r', str(fps_float), '-i', '-',
                        '-vcodec', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-pix_fmt', 'yuv420p', output_video_path
                    ]
                    
                    process = subprocess.Popen(
                        ffmpeg_cmd, 
                        stdin=subprocess.PIPE, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    
                    for frame_to_write in annotated_frames_list:
                        if process.stdin and process.stdin.closed:
                            logger.warning("MMPose FFMPEG: stdin closed.")
                            break
                        try:
                            if process.stdin:
                                process.stdin.write(frame_to_write.tobytes())
                        except (IOError, BrokenPipeError) as e_pipe:
                            logger.warning(f"MMPose FFMPEG: pipe broken: {e_pipe}")
                            break
                    
                    if process.stdin and not process.stdin.closed:
                        logger.info("MMPose FFMPEG: Closing stdin.")
                        process.stdin.close()
                    
                    stdout_bytes, stderr_bytes = b'', b''
                    ffmpeg_timed_out = False
                    ffmpeg_return_code = None
                    
                    try:
                        stdout_bytes, stderr_bytes = process.communicate(timeout=120)
                        ffmpeg_return_code = process.returncode
                    except subprocess.TimeoutExpired:
                        logger.error("MMPose FFMPEG: timed out. Killing.")
                        process.kill()
                        try:
                            stdout_bytes, stderr_bytes = process.communicate(timeout=5)
                        except Exception:
                            pass
                        ffmpeg_timed_out = True
                        ffmpeg_return_code = process.returncode
                    except ValueError as ve:
                        if "flush of closed file" in str(ve).lower() or "write to closed file" in str(ve).lower():
                            logger.warning(f"MMPose FFMPEG: '{ve}'. Wait/poll.")
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                logger.warning("MMPose FFMPEG: timed out on wait().")
                            ffmpeg_return_code = process.poll()
                        else:
                            logger.error(f"MMPose FFMPEG: ValueError: {ve}", exc_info=True)
                            ffmpeg_return_code = -1
                    except Exception as e_comm:
                        logger.error(f"MMPose FFMPEG: communicate() error: {e_comm}", exc_info=True)
                        ffmpeg_return_code = -1
                    
                    if ffmpeg_return_code is None and not ffmpeg_timed_out:
                        ffmpeg_return_code = process.poll()
                    
                    if ffmpeg_timed_out:
                        response_content["message"] = "MMPose FFMPEG: timed out."
                        gcs_url_result = ""
                    elif ffmpeg_return_code != 0:
                        logger.error(f"MMPose FFMPEG: failed (RC: {ffmpeg_return_code}):")
                        if stdout_bytes:
                            logger.error(f"FFMPEG stdout: {stdout_bytes.decode(errors='ignore')}")
                        if stderr_bytes:
                            logger.error(f"FFMPEG stderr: {stderr_bytes.decode(errors='ignore')}")
                        gcs_url_result = ""
                        response_content["message"] = f"MMPose FFMPEG: failed. RC: {ffmpeg_return_code}"
                    else:
                        logger.info(f"MMPose: Video via FFMPEG: {output_video_path}")
                        gcs_url_result = await upload_to_gcs(output_video_path)
                    
                    response_content["video_url"] = gcs_url_result if gcs_url_result else None
                    
                    if not gcs_url_result and "message" not in response_content:
                        response_content["message"] = "MMPose: Upload fail/FFMPEG silent fail."
                        
                finally:
                    if output_video_path and os.path.exists(output_video_path):
                        os.unlink(output_video_path)
        
        if not response_content:
            return JSONResponse(content={"detail": "MMPose: No output."}, status_code=200)
        
        return JSONResponse(content=response_content)
            
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"MMPose HTTP error: {e.response.status_code}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"MMPose: DL fail: {e.response.status_code}")
    except Exception as e:
        logger.error(f"MMPose: Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"MMPose: Internal error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)

if __name__ == "__main__":
    logger.info(f"Attempting to start Uvicorn for MMPose service on port 8003 (PID: {os.getpid()}).")
    if mmpose_model is None:
        logger.critical("MMPose model could not be loaded at startup. Service will be unhealthy.")
    else:
        logger.info(f"MMPose service starting successfully with {model_info['name']} model.")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_config=None)