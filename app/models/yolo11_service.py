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
# import math # Not used in this script's pose-only logic
import httpx
import torch
import uuid
import uvicorn # Ensured import
from datetime import datetime
from google.cloud import storage
from ultralytics import YOLO # YOLOv11 is likely a custom or variant handled by Ultralytics YOLO class
import subprocess # For FFMPEG

# Setup for utils.video_utils
try:
    from utils.video_utils import get_video_info, extract_frames # Ensure this uses the updated version
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Assumes utils is sibling to models
    from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- YOLOv11 POSE SERVICE SCRIPT STARTED (URL-Only, FFMPEG, Detailed Logging) ---")

app = FastAPI(title="YOLOv11 Pose Service (URL Input Only)", version="1.1.0") # Incremented version
logger.info("FastAPI app object created for YOLOv11 service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_yolo11" # Specific GCS folder
MODEL_DIR = "/opt/padel/app/weights"
MODEL_NAME = "yolo11n-pose.pt" # Assuming this is the correct model from your project context
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

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
yolo11_pose_model = None # Renamed for clarity
try:
    logger.info(f"Attempting to load YOLOv11 Pose model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"CRITICAL: Model file {MODEL_PATH} does not exist. This service requires the model file to be present. Please ensure it's deployed correctly by your deploy.yml (e.g., via wget or baked into a custom image).")
        # Unlike standard YOLOv8, custom names like 'yolo11n-pose.pt' are unlikely to be auto-downloaded by Ultralytics YOLO()
        # So, we explicitly treat its absence as a critical error for this service.
        yolo11_pose_model = None
    else:
        yolo11_pose_model = YOLO(MODEL_PATH) # Load from the specific path
        logger.info(f"YOLOv11 Pose model '{MODEL_NAME}' loaded successfully from {MODEL_PATH}.")
        if torch.cuda.is_available():
            logger.info("Moving YOLOv11 Pose model to CUDA device and fusing layers.")
            yolo11_pose_model.to('cuda')
            # model.fuse() # fuse() might not be applicable or needed for all custom YOLO models or pose tasks. Test this.
            # If 'fuse' causes issues for this specific model, comment it out.
            try:
                yolo11_pose_model.fuse()
                logger.info("YOLOv11 Pose model on CUDA and fused.")
            except Exception as e_fuse:
                logger.warning(f"Could not fuse YOLOv11 Pose model, continuing without fusion: {e_fuse}")
                logger.info("YOLOv11 Pose model on CUDA (not fused).")
        else:
            logger.info("CUDA not available. YOLOv11 Pose model will run on CPU.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load YOLOv11 Pose model: {e}", exc_info=True)
    yolo11_pose_model = None

# --- Core Logic ---
def detect_frame_poses(frame: np.ndarray) -> List[Dict[str, Any]]: # Processes a single frame
    if yolo11_pose_model is None:
        logger.warning("YOLOv11 Pose model not loaded for detect_frame_poses, returning empty list.")
        return []

    frame_poses = []
    try:
        results = yolo11_pose_model(frame, verbose=False, half=torch.cuda.is_available()) # Single frame result
        result = results[0] # Get the first (and only) result object for the single frame

        if hasattr(result, 'keypoints') and result.keypoints is not None and result.keypoints.data is not None:
            for k_idx, keypoints_tensor in enumerate(result.keypoints.data): # Iterate over detected persons
                kpts = keypoints_tensor.cpu().numpy() # For one person: (N_keypoints, 3)  (x, y, confidence)
                
                # Get bounding box from keypoints or boxes attribute if available
                # Ultralytics results objects usually have a .boxes attribute for detected persons
                person_bbox_data = None
                if hasattr(result, 'boxes') and result.boxes is not None and result.boxes.data is not None and k_idx < len(result.boxes.data):
                    person_bbox_data = result.boxes.data[k_idx].cpu().numpy() # x1, y1, x2, y2, conf, cls
                
                bbox_dict = {}
                overall_confidence = float(kpts[:, 2].mean()) if kpts.shape[0] > 0 else 0.0

                if person_bbox_data is not None:
                    x1, y1, x2, y2, box_conf, _ = person_bbox_data
                    bbox_dict = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                    # Use box confidence if keypoint mean confidence is low, or average them
                    if overall_confidence < 0.1 and box_conf > overall_confidence : overall_confidence = float(box_conf)

                elif kpts.shape[0] > 0: # Fallback to bbox from keypoints if no .boxes attribute used
                    valid_kpts = kpts[kpts[:, 2] > 0.05] # Use a small threshold for bbox calculation
                    if len(valid_kpts) > 0:
                        x1, y1 = valid_kpts[:, 0].min(), valid_kpts[:, 1].min()
                        x2, y2 = valid_kpts[:, 0].max(), valid_kpts[:, 1].max()
                        bbox_dict = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                
                keypoint_list_dict = {}
                keypoint_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
                
                for kpt_idx, kpt_data in enumerate(kpts):
                    x_kpt, y_kpt, conf_kpt = kpt_data
                    if kpt_idx < len(keypoint_names): # Ensure we don't go out of bounds for names
                        keypoint_list_dict[keypoint_names[kpt_idx]] = {
                            "x": float(x_kpt), "y": float(y_kpt), "confidence": float(conf_kpt)
                        }
                
                if keypoint_list_dict: # Only add if keypoints were processed
                    frame_poses.append({
                        "keypoints": keypoint_list_dict,
                        "confidence": overall_confidence, 
                        "bbox": bbox_dict
                    })
        else:
            logger.debug("No keypoints attribute or keypoints data in result.")
            
    except Exception as e_infer:
        logger.error(f"Error during YOLOv11 pose inference/processing on a frame: {e_infer}", exc_info=True)
        # Return empty for this frame on error
    return frame_poses


def draw_poses_on_frame(frame: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
    annotated_frame = frame.copy()
    connections = [("nose", "left_eye"),("nose", "right_eye"),("left_eye","left_ear"),("right_eye","right_ear"),("nose", "left_shoulder"),("nose", "right_shoulder"),("left_shoulder", "right_shoulder"),("left_shoulder", "left_elbow"),("right_shoulder", "right_elbow"),("left_elbow", "left_wrist"),("right_elbow", "right_wrist"),("left_shoulder", "left_hip"),("right_shoulder", "right_hip"),("left_hip", "right_hip"),("left_hip", "left_knee"),("right_hip", "right_knee"),("left_knee", "left_ankle"),("right_knee", "right_ankle")]
    for pose in poses:
        keypoints = pose.get("keypoints", {})
        bbox = pose.get("bbox", {})
        pose_conf = pose.get("confidence", 0.0)

        # Draw bounding box if available
        if bbox and pose_conf > 0.3: # Example: only draw bbox if pose confidence is decent
             cv2.rectangle(annotated_frame, (int(bbox["x1"]), int(bbox["y1"])), (int(bbox["x2"]), int(bbox["y2"])), (255,0,0), 1) # Blue bbox

        for name, data in keypoints.items():
            x,y,conf = int(data.get("x",0)), int(data.get("y",0)), data.get("confidence",0.0)
            if conf > 0.5: cv2.circle(annotated_frame, (x,y), 3, (0,255,0), -1) # Green keypoints
        for p1_name, p2_name in connections:
            p1, p2 = keypoints.get(p1_name), keypoints.get(p2_name)
            if p1 and p2 and p1.get("confidence",0)>0.5 and p2.get("confidence",0)>0.5:
                cv2.line(annotated_frame, (int(p1["x"]), int(p1["y"])), (int(p2["x"]), int(p2["y"])), (0,255,255), 1) # Yellow lines
    return annotated_frame

# --- API Endpoints ---
@app.get("/healthz")
async def health_check():
    if yolo11_pose_model is None:
        logger.warning("/healthz: YOLOv11 Pose model not loaded.")
        return JSONResponse(content={"status": "unhealthy", "model": "yolo11 not loaded"}, status_code=503)
    logger.info("/healthz: YOLOv11 Pose model loaded, service healthy.")
    return {"status": "healthy", "model": "yolo11"}

@app.post("/yolo11") # Changed endpoint name to match service file
async def detect_video_poses( # Renamed function
    payload: VideoAnalysisURLRequest,
    http_request: Request
):
    logger.info(f"--- Enter /yolo11 endpoint (URL-only, FFMPEG) ---")
    logger.info(f"Request Headers: {http_request.headers}")
    logger.info(f"Received Payload: video_url='{payload.video_url}', video_output_requested={payload.video}, data_output_requested={payload.data}")

    if yolo11_pose_model is None:
        logger.error("/yolo11: Model not loaded. Cannot process request.")
        raise HTTPException(status_code=503, detail="YOLOv11 Pose model is not available.")

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

        num_frames_to_process = 75 # For testing; for production, adjust or use -1 for all
        frames = extract_frames(temp_downloaded_path, num_frames_to_extract=num_frames_to_process)
        logger.info(f"Extracted {len(frames)} consecutive frames (target={num_frames_to_process}).")

        if not frames:
            logger.error(f"No frames extracted from {temp_downloaded_path}.")
            raise HTTPException(status_code=400, detail="No frames could be extracted from the provided video.")

        all_poses_per_frame: List[List[Dict[str, Any]]] = []
        annotated_frames_list: List[np.ndarray] = []
        
        # YOLOv11/Ultralytics pose models typically process frame by frame efficiently when called.
        # Batching frames into the yolo11_pose_model() call is often done for object detection.
        # For pose, it often returns a list of results if a list of frames is passed.
        # Let's try batching as it's in your original script's structure.
        batch_size = 8 # As in your original
        logger.info(f"Starting pose detection for {len(frames)} frames. Batch size: {batch_size}")

        for i in range(0, len(frames), batch_size):
            batch_of_frames = frames[i:i+batch_size]
            if not batch_of_frames: continue
            logger.debug(f"Processing pose batch {i//batch_size + 1}, frames {i} to {i+len(batch_of_frames)-1}")
            
            try:
                # Pass the batch of frames to the model
                batch_results = yolo11_pose_model(batch_of_frames, verbose=False, half=torch.cuda.is_available())
                logger.debug(f"Batch {i//batch_size + 1}: Pose inference complete, {len(batch_results)} result objects (one per frame in batch).")

                for frame_idx_in_batch, single_frame_result in enumerate(batch_results):
                    original_frame_index = i + frame_idx_in_batch
                    current_frame_poses: List[Dict[str, Any]] = []

                    if hasattr(single_frame_result, 'keypoints') and single_frame_result.keypoints is not None and single_frame_result.keypoints.data is not None:
                        for k_idx, keypoints_tensor in enumerate(single_frame_result.keypoints.data): # Iterate over detected persons in this frame
                            kpts = keypoints_tensor.cpu().numpy() # For one person: (N_keypoints, 3) (x, y, confidence)
                            
                            person_bbox_data = None
                            if hasattr(single_frame_result, 'boxes') and single_frame_result.boxes and k_idx < len(single_frame_result.boxes.data):
                                person_bbox_data = single_frame_result.boxes.data[k_idx].cpu().numpy()
                            
                            bbox_dict = {}
                            overall_confidence = float(kpts[:, 2].mean()) if kpts.shape[0] > 0 and kpts[:, 2].size > 0 else 0.0

                            if person_bbox_data is not None:
                                x1, y1, x2, y2, box_conf, _ = person_bbox_data
                                bbox_dict = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                                if overall_confidence < 0.1 and box_conf > overall_confidence: overall_confidence = float(box_conf)
                            elif kpts.shape[0] > 0:
                                valid_kpts = kpts[kpts[:, 2] > 0.05]
                                if len(valid_kpts) > 0:
                                    x1,y1,x2,y2 = valid_kpts[:,0].min(),valid_kpts[:,1].min(),valid_kpts[:,0].max(),valid_kpts[:,1].max()
                                    bbox_dict = {"x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2)}
                            
                            keypoint_list_dict = {}
                            keypoint_names = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
                            for kpt_idx, kpt_data in enumerate(kpts):
                                x_kpt, y_kpt, conf_kpt = kpt_data
                                if kpt_idx < len(keypoint_names):
                                    keypoint_list_dict[keypoint_names[kpt_idx]] = {"x": float(x_kpt), "y": float(y_kpt), "confidence": float(conf_kpt)}
                            
                            if keypoint_list_dict:
                                current_frame_poses.append({"keypoints": keypoint_list_dict, "confidence": overall_confidence, "bbox": bbox_dict})
                    else: logger.debug(f"Frame {original_frame_index}: No keypoints data in result.")
                    all_poses_per_frame.append(current_frame_poses)
                    if payload.video:
                        annotated_frames_list.append(draw_poses_on_frame(batch_of_frames[frame_idx_in_batch], current_frame_poses))
            except Exception as e_infer:
                logger.error(f"Pose inference error on batch {i//batch_size + 1}: {e_infer}", exc_info=True)
                for k_batch_idx in range(len(batch_of_frames)): 
                    all_poses_per_frame.append([]) # Append empty for failed frames
                    if payload.video: annotated_frames_list.append(batch_of_frames[k_batch_idx]) # Add original
                continue
        
        logger.info(f"Finished pose detection. Poses collected: {len(all_poses_per_frame)}. Annotated frames: {len(annotated_frames_list)}.")
        
        response_content: Dict[str, Any] = {}
        if payload.data:
            response_content["data"] = {"poses_per_frame": all_poses_per_frame} # Changed key

        if payload.video:
            if not annotated_frames_list:
                logger.warning("YOLOv11: Video output requested, but no annotated frames.")
                response_content["video_url"] = None; response_content["message"] = "YOLOv11: No frames to annotate."
            else:
                output_video_path: Optional[str] = None; gcs_url_result: str = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out_vid_file: output_video_path = temp_out_vid_file.name
                    height, width = annotated_frames_list[0].shape[:2]
                    fps_float = float(video_info.get("fps", 30.0));
                    if fps_float <= 0: fps_float = 30.0
                    logger.info(f"YOLOv11: Creating video via FFMPEG: {output_video_path} (H={height},W={width},FPS={fps_float})")
                    ffmpeg_cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-s',f'{width}x{height}','-pix_fmt','bgr24','-r',str(fps_float),'-i','-','-vcodec','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p',output_video_path]
                    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    for frame_to_write in annotated_frames_list:
                        if process.stdin and process.stdin.closed: logger.warning("YOLOv11 FFMPEG: stdin closed."); break
                        try:
                            if process.stdin: process.stdin.write(frame_to_write.tobytes())
                        except (IOError, BrokenPipeError) as e_pipe: logger.warning(f"YOLOv11 FFMPEG: pipe broken: {e_pipe}"); break
                    if process.stdin and not process.stdin.closed: logger.info("YOLOv11 FFMPEG: Closing stdin."); process.stdin.close()
                    
                    stdout_bytes, stderr_bytes = b'',b''; ffmpeg_timed_out=False; ffmpeg_return_code=None
                    try:
                        stdout_bytes, stderr_bytes = process.communicate(timeout=120)
                        ffmpeg_return_code = process.returncode
                    except subprocess.TimeoutExpired:
                        logger.error("YOLOv11 FFMPEG: timed out. Killing."); process.kill()
                        try: stdout_bytes, stderr_bytes = process.communicate(timeout=5)
                        except Exception: pass
                        ffmpeg_timed_out=True; ffmpeg_return_code=process.returncode
                    except ValueError as ve:
                        if "flush of closed file" in str(ve).lower() or "write to closed file" in str(ve).lower():
                            logger.warning(f"YOLOv11 FFMPEG: '{ve}'. Wait/poll.");
                            try: process.wait(timeout=5)
                            except subprocess.TimeoutExpired: logger.warning("YOLOv11 FFMPEG: timed out on wait().")
                            ffmpeg_return_code = process.poll()
                        else: logger.error(f"YOLOv11 FFMPEG: ValueError: {ve}", exc_info=True); ffmpeg_return_code = -1
                    except Exception as e_comm: logger.error(f"YOLOv11 FFMPEG: communicate() error: {e_comm}", exc_info=True); ffmpeg_return_code = -1
                    
                    if ffmpeg_return_code is None and not ffmpeg_timed_out: ffmpeg_return_code = process.poll()
                    if ffmpeg_timed_out: response_content["message"] = "YOLOv11 FFMPEG: processing timed out."; gcs_url_result = ""
                    elif ffmpeg_return_code != 0:
                        logger.error(f"YOLOv11 FFMPEG: failed (RC: {ffmpeg_return_code}):")
                        if stdout_bytes: logger.error(f"FFMPEG stdout: {stdout_bytes.decode(errors='ignore')}")
                        if stderr_bytes: logger.error(f"FFMPEG stderr: {stderr_bytes.decode(errors='ignore')}")
                        gcs_url_result = ""; response_content["message"] = f"YOLOv11 FFMPEG: failed. RC: {ffmpeg_return_code}"
                    else: logger.info(f"YOLOv11: Video via FFMPEG: {output_video_path}"); gcs_url_result = await upload_to_gcs(output_video_path)
                    response_content["video_url"] = gcs_url_result if gcs_url_result else None
                    if not gcs_url_result and "message" not in response_content: response_content["message"] = "YOLOv11: Upload fail/FFMPEG silent fail."
                finally:
                     if output_video_path and os.path.exists(output_video_path): os.unlink(output_video_path)
        
        if not response_content: return JSONResponse(content={"detail": "YOLOv11: No output."}, status_code=200)
        return JSONResponse(content=response_content)
            
    except HTTPException: raise
    except httpx.HTTPStatusError as e: logger.error(f"YOLOv11 HTTP error: {e.response.status_code}", exc_info=True); raise HTTPException(status_code=400, detail=f"YOLOv11: DL fail: {e.response.status_code}")
    except Exception as e: # This is the block where line 67 (in original script) with the IndentationError was.
        logger.error(f"YOLOv11: Unexpected error in /yolo11 endpoint: {str(e)}", exc_info=True) # Corrected indentation and added exc_info
        raise HTTPException(status_code=500, detail=f"YOLOv11: Internal error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path): os.unlink(temp_downloaded_path)

if __name__ == "__main__":
    logger.info(f"Attempting to start Uvicorn for YOLOv11 service on port 8001 (PID: {os.getpid()}).")
    if yolo11_pose_model is None: # Changed variable name
        logger.critical("YOLOv11 Pose model could not be loaded at startup. Service will be unhealthy.")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_config=None)
