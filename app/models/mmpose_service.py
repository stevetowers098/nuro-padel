from fastapi import FastAPI, HTTPException, Request # Removed UploadFile, File, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl # For request body validation
import tempfile
import os
import cv2
import numpy as np
# import supervision as sv # Not used in this version for drawing
from typing import Dict, Any, List, Optional
import sys
import logging
import math # Keep for calculate_angle
# import random # Keep for dummy biomechanics scores
import httpx
import torch
import uuid
import uvicorn # Added import
from datetime import datetime
from google.cloud import storage
import subprocess # For FFMPEG

# MMPose v1.x API imports
from mmpose.apis import init_model, inference_topdown

# Setup for utils.video_utils
try:
    from utils.video_utils import get_video_info, extract_frames # This should use the updated version
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG for more info
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- MMPOSE SERVICE SCRIPT STARTED (URL-Only, FFMPEG, Detailed Logging) ---")

app = FastAPI(title="MMPose Biomechanics Service (URL Input Only)", version="1.1.0") # Incremented version
logger.info("FastAPI app object created for MMPose service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_mmpose" # Specific GCS folder for this service

# --- Pydantic Model for Request Body ---
class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False # Request annotated video output?
    data: bool = True   # Request JSON data output? (Defaulting to True)

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

# --- Model Loading ---
mmpose_model = None # Renamed to avoid conflict if other models were in same global scope
try:
    model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Initializing MMPose model on device: {model_device}")
    logger.info("Attempting to load RTMPose-M COCO model using alias...")
    mmpose_model = init_model(
        'rtmpose-m_8xb256-420e_coco-256x192', # Model alias
        device=model_device
    )
    logger.info("RTMPose model loaded/downloaded successfully.")
except Exception as e_rtmpose:
    logger.warning(f"Could not load RTMPose model using alias: {e_rtmpose}. Attempting HRNet fallback.")
    try:
        model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Attempting to load HRNet-W48 COCO model as fallback (device: {model_device})...")
        mmpose_model = init_model(
            'td-hm_hrnet-w48_8xb32-210e_coco-256x192', # Model alias for HRNet
            device=model_device
        )
        logger.info("HRNet fallback model loaded/downloaded successfully.")
    except Exception as e_hrnet:
        logger.error(f"CRITICAL: Failed to load any MMPose model (RTMPose or HRNet): {e_hrnet}", exc_info=True)
        mmpose_model = None

# --- Core Logic ---
# Biomechanical calculation functions (calculate_angle, calculate_..._score) remain the same as in your provided script
# ... (Paste your calculate_angle, calculate_posture_score, etc. functions here) ...
def calculate_angle(a, b, c): # Placeholder, use your actual function
    p1, p2, p3 = np.array(a), np.array(b), np.array(c)
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude_product < 1e-7: return 0.0 # Avoid division by zero for collinear points
    cos_angle = np.clip(dot_product / magnitude_product, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def calculate_posture_score(keypoints): return np.random.uniform(70,95) if keypoints else 0
def calculate_balance_score(keypoints): return np.random.uniform(65,90) if keypoints else 0
def calculate_movement_efficiency(joint_angles): return np.random.uniform(60,95) if joint_angles else 0
def calculate_power_potential(joint_angles, keypoints): return np.random.uniform(50,100) if joint_angles and keypoints else 0


def analyze_frame_biomechanics(frame: np.ndarray) -> Dict[str, Any]: # Renamed for single frame processing
    if mmpose_model is None:
        logger.warning("MMPose model not loaded for analyze_frame_biomechanics, returning dummy data.")
        return {"keypoints": {}, "joint_angles": {}, "biomechanical_metrics": {"posture_score": 0}}

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
        pose_data_samples = inference_topdown(mmpose_model, frame, bbox_cs='')
        if pose_data_samples:
            data_sample = pose_data_samples[0]
            if hasattr(data_sample, 'pred_instances') and data_sample.pred_instances and len(data_sample.pred_instances) > 0:
                pred_kpts_tensor = data_sample.pred_instances.keypoints[0]
                pred_scores_tensor = data_sample.pred_instances.keypoint_scores[0]
                pred_kpts = pred_kpts_tensor.cpu().numpy()
                pred_scores = pred_scores_tensor.cpu().numpy()
                for idx in range(pred_kpts.shape[0]):
                    if idx < len(keypoint_names):
                        current_keypoints[keypoint_names[idx]] = {
                            "x": float(pred_kpts[idx, 0]), "y": float(pred_kpts[idx, 1]),
                            "confidence": float(pred_scores[idx])
                        }
        # Calculate angles (simplified, expand with your full logic)
        if all(k in current_keypoints for k in ["left_shoulder", "left_elbow", "left_wrist"]):
            joint_angles["left_elbow"] = calculate_angle(
                (current_keypoints["left_shoulder"]["x"], current_keypoints["left_shoulder"]["y"]),
                (current_keypoints["left_elbow"]["x"], current_keypoints["left_elbow"]["y"]),
                (current_keypoints["left_wrist"]["x"], current_keypoints["left_wrist"]["y"])
            )
        # ... (add all other angle calculations from your original script) ...

        biomechanical_metrics = {
            "posture_score": calculate_posture_score(current_keypoints),
            "balance_score": calculate_balance_score(current_keypoints),
            "movement_efficiency": calculate_movement_efficiency(joint_angles),
            "power_potential": calculate_power_potential(joint_angles, current_keypoints),
        }
    except Exception as e_analyze:
        logger.error(f"Error during MMPose inference or analysis on a frame: {e_analyze}", exc_info=True)
        # Return dummy/empty on error for this frame
        return {"keypoints": {}, "joint_angles": {}, "biomechanical_metrics": {"error_processing_frame": True}}
        
    return {"keypoints": current_keypoints, "joint_angles": joint_angles, "biomechanical_metrics": biomechanical_metrics}

def draw_biomechanics_on_frame(frame: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    # ... (Your existing draw_biomechanics_on_frame function, ensure it uses .get() for safety) ...
    # This function should be the same as the one you provided for mmpose_service.py earlier.
    # For brevity, I'm not re-pasting the full drawing logic here, but it should be included.
    # Ensure it refers to analysis.get("keypoints", {}), etc.
    # --- PASTE YOUR FULL draw_biomechanics_on_frame HERE ---
    annotated_frame = frame.copy()
    keypoints = analysis.get("keypoints", {}) 
    joint_angles = analysis.get("joint_angles", {})
    metrics = analysis.get("biomechanical_metrics", {})
    connections = [("nose", "left_shoulder"), ("nose", "right_shoulder"), ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"),("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),("right_shoulder", "right_hip"), ("left_hip", "right_hip"),("left_hip", "left_knee"), ("right_hip", "right_knee"),("left_knee", "left_ankle"), ("right_knee", "right_ankle"),("nose", "left_eye"),("nose", "right_eye"),("left_eye","left_ear"),("right_eye","right_ear")] # Added COCO eye connections
    for keypoint_name, keypoint_data in keypoints.items():
        x, y, confidence = int(keypoint_data.get("x",0)), int(keypoint_data.get("y",0)), keypoint_data.get("confidence", 0.0)
        if confidence > 0.5: cv2.circle(annotated_frame, (x, y), 5, (0, int(255 * confidence), int(255 * (1-confidence))), -1)
    for connection in connections:
        start_kp_data, end_kp_data = keypoints.get(connection[0]), keypoints.get(connection[1])
        if start_kp_data and end_kp_data and start_kp_data.get("confidence",0.0)>0.5 and end_kp_data.get("confidence",0.0)>0.5:
            start_x, start_y, end_x, end_y = int(start_kp_data["x"]), int(start_kp_data["y"]), int(end_kp_data["x"]), int(end_kp_data["y"])
            avg_conf = (start_kp_data["confidence"] + end_kp_data["confidence"])/2
            cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), (0,255,int(255*(1-avg_conf))), max(1,int(3*avg_conf)))
    # ... (rest of drawing for angles and metrics as in your original) ...
    metrics_y_offset = 30
    for metric_name, value in metrics.items():
        cv2.putText(annotated_frame, f"{metric_name}: {value}", (10, metrics_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        metrics_y_offset += 25
    for joint_name, angle_val in joint_angles.items(): # Draw angles
        kp_for_angle_text = keypoints.get(joint_name.split('_')[-1]) # e.g. left_elbow -> elbow
        if not kp_for_angle_text and '_' in joint_name: kp_for_angle_text = keypoints.get(joint_name.split('_')[1]) # Try second part
        if kp_for_angle_text and kp_for_angle_text.get("confidence", 0.0) > 0.3:
             cv2.putText(annotated_frame, f"{angle_val:.1f}d", (int(kp_for_angle_text['x'])+5, int(kp_for_angle_text['y'])-5), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,255),1)
    return annotated_frame
    # --- END OF draw_biomechanics_on_frame ---

# --- API Endpoints ---
@app.get("/healthz")
async def health_check():
    if mmpose_model is None:
        logger.warning("/healthz: MMPose model not loaded.")
        return JSONResponse(content={"status": "unhealthy", "model": "mmpose not loaded"}, status_code=503)
    logger.info("/healthz: MMPose model loaded, service healthy.")
    return {"status": "healthy", "model": "mmpose"}

@app.post("/mmpose")
async def analyze_video_endpoint( # Renamed function for clarity
    payload: VideoAnalysisURLRequest,
    http_request: Request
):
    logger.info(f"--- Enter /mmpose endpoint (URL-only, FFMPEG) ---")
    logger.info(f"Request Headers: {http_request.headers}")
    logger.info(f"Received Payload: video_url='{payload.video_url}', video_output_requested={payload.video}, data_output_requested={payload.data}")

    if mmpose_model is None:
        logger.error("/mmpose: Model not loaded. Cannot process request.")
        raise HTTPException(status_code=503, detail="MMPose model is not available.")

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

        # Use the updated extract_frames for consecutive frames
        # For testing, limit frames. For production, adjust or set to -1 for all.
        num_frames_to_process = 75 # Example: process first 75 consecutive frames
        frames = extract_frames(temp_downloaded_path, num_frames_to_extract=num_frames_to_process)
        logger.info(f"Extracted {len(frames)} consecutive frames (num_frames_to_extract={num_frames_to_process}).")

        if not frames:
            logger.error(f"No frames extracted from {temp_downloaded_path}.")
            raise HTTPException(status_code=400, detail="No frames could be extracted from the provided video.")

        all_analyses_per_frame: List[Dict[str, Any]] = []
        annotated_frames_list: List[np.ndarray] = []
        
        logger.info(f"Starting biomechanics analysis for {len(frames)} frames.")
        for frame_idx, frame_content in enumerate(frames):
            logger.debug(f"Analyzing frame {frame_idx} for biomechanics.")
            analysis_result = analyze_frame_biomechanics(frame_content) # Process one frame
            all_analyses_per_frame.append(analysis_result)
            if payload.video:
                annotated_frames_list.append(draw_biomechanics_on_frame(frame_content, analysis_result))
        
        logger.info(f"Finished biomechanics analysis. Analyses collected: {len(all_analyses_per_frame)}. Annotated frames: {len(annotated_frames_list)}.")
        
        response_content: Dict[str, Any] = {}
        if payload.data:
            response_content["data"] = {"biomechanics_per_frame": all_analyses_per_frame}

        if payload.video:
            if not annotated_frames_list:
                logger.warning("Video output requested, but no annotated frames for MMPose.")
                response_content["video_url"] = None
                response_content["message"] = "MMPose: Video output requested, but no frames to annotate."
            else:
                output_video_path: Optional[str] = None
                gcs_url_result: str = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out_vid_file:
                        output_video_path = temp_out_vid_file.name
                    
                    height, width = annotated_frames_list[0].shape[:2]
                    fps_float = float(video_info.get("fps", 30.0))
                    if fps_float <= 0: fps_float = 30.0
                    
                    logger.info(f"MMPose: Creating video via FFMPEG: {output_video_path} (H={height}, W={width}, FPS={fps_float})")
                    ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps_float), '-i', '-', '-vcodec', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', output_video_path]
                    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    for frame_to_write in annotated_frames_list:
                        if process.stdin and process.stdin.closed: logger.warning("MMPose FFMPEG: stdin pipe closed. Stopping."); break
                        try:
                            if process.stdin: process.stdin.write(frame_to_write.tobytes())
                        except (IOError, BrokenPipeError) as e_pipe: logger.warning(f"MMPose FFMPEG: stdin pipe broken: {e_pipe}"); break
                    
                    if process.stdin and not process.stdin.closed: logger.info("MMPose FFMPEG: Closing stdin."); process.stdin.close()
                    
                    stdout_bytes, stderr_bytes = b'', b''; ffmpeg_timed_out = False; ffmpeg_return_code = None
                    try:
                        logger.debug("MMPose FFMPEG: Attempting process.communicate()...")
                        stdout_bytes, stderr_bytes = process.communicate(timeout=120)
                        ffmpeg_return_code = process.returncode
                        logger.debug(f"MMPose FFMPEG: communicate() finished. RC: {ffmpeg_return_code}")
                    except subprocess.TimeoutExpired:
                        logger.error("MMPose FFMPEG: process timed out during communicate(). Killing."); process.kill()
                        try: stdout_bytes, stderr_bytes = process.communicate(timeout=5)
                        except Exception: pass
                        ffmpeg_timed_out = True; ffmpeg_return_code = process.returncode
                    except ValueError as ve:
                        if "flush of closed file" in str(ve).lower() or "write to closed file" in str(ve).lower():
                            logger.warning(f"MMPose FFMPEG: Caught '{ve}'. Trying wait/poll.")
                            try: process.wait(timeout=5)
                            except subprocess.TimeoutExpired: logger.warning("MMPose FFMPEG: timed out on wait() after ValueError.")
                            ffmpeg_return_code = process.poll()
                        else: logger.error(f"MMPose FFMPEG: Unexpected ValueError: {ve}", exc_info=True); ffmpeg_return_code = -1
                    except Exception as e_comm: logger.error(f"MMPose FFMPEG: Error during communicate(): {e_comm}", exc_info=True); ffmpeg_return_code = -1
                    
                    if ffmpeg_return_code is None and not ffmpeg_timed_out:
                        logger.warning("MMPose FFMPEG: RC is None. Polling."); ffmpeg_return_code = process.poll()

                    if ffmpeg_timed_out:
                        response_content["message"] = "MMPose FFMPEG: processing timed out."
                        logger.error(response_content["message"]); gcs_url_result = ""
                    elif ffmpeg_return_code != 0:
                        logger.error(f"MMPose FFMPEG: failed (RC: {ffmpeg_return_code}):")
                        if stdout_bytes: logger.error(f"FFMPEG stdout: {stdout_bytes.decode(errors='ignore')}")
                        if stderr_bytes: logger.error(f"FFMPEG stderr: {stderr_bytes.decode(errors='ignore')}")
                        gcs_url_result = ""; response_content["message"] = f"MMPose FFMPEG: processing failed. RC: {ffmpeg_return_code}"
                    else:
                        logger.info(f"MMPose: Video created via FFMPEG: {output_video_path}")
                        gcs_url_result = await upload_to_gcs(output_video_path)
                    
                    response_content["video_url"] = gcs_url_result if gcs_url_result else None
                    if not gcs_url_result and "message" not in response_content:
                         response_content["message"] = "MMPose: Failed to upload video (or FFMPEG failed silently)."
                finally:
                     if output_video_path and os.path.exists(output_video_path): os.unlink(output_video_path)
        
        if not response_content:
             logger.warning("MMPose: No output to return.")
             return JSONResponse(content={"detail": "MMPose: No output generated."}, status_code=200)
        return JSONResponse(content=response_content)
            
    except HTTPException: raise
    except httpx.HTTPStatusError as e:
        logger.error(f"MMPose HTTP error (video download): {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"MMPose: Failed download/access video_url: {e.response.status_code}")
    except Exception as e:
        logger.error(f"MMPose: Unexpected error in /mmpose endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"MMPose: Internal server error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            logger.debug(f"MMPose: Deleting temp input video: {temp_downloaded_path}")
            os.unlink(temp_downloaded_path)

if __name__ == "__main__":
    logger.info(f"Attempting to start Uvicorn for MMPose service on port 8003 (PID: {os.getpid()}).")
    if mmpose_model is None: # Changed variable name
        logger.critical("MMPose model could not be loaded at startup. Service will be unhealthy.")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_config=None)
