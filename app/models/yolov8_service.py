import os
from pathlib import Path

# === PRODUCTION SAFETY: DISABLE ALL AUTO-DOWNLOADS ===
os.environ['YOLO_OFFLINE'] = '1'
os.environ['ULTRALYTICS_OFFLINE'] = '1'
os.environ['ONLINE'] = 'False'
os.environ['YOLO_TELEMETRY'] = 'False'

# Force local-only model loading
WEIGHTS_DIR = Path("/opt/padel/app/weights")

def get_local_model_path(model_name: str) -> str:
    """Get absolute path to model - fails if not exists locally"""
    model_path = WEIGHTS_DIR / model_name
    if not model_path.exists():
        available_models = list(WEIGHTS_DIR.glob("*.pt"))
        raise FileNotFoundError(
            f"❌ PRODUCTION ERROR: Model {model_name} not found locally.\n"
            f"📁 Checked: {model_path}\n"
            f"📋 Available: {[m.name for m in available_models]}\n"
            f"🚫 Auto-download disabled for production stability."
        )
    return str(model_path.absolute())

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import tempfile
import os
import cv2
import numpy as np
# import supervision as sv # Not used in this version for drawing
from typing import Dict, Any, List, Optional
import sys
import logging
import httpx
import torch
import uuid
import uvicorn # Ensured import
from datetime import datetime
from google.cloud import storage
from ultralytics import YOLO
import subprocess # For FFMPEG

# Setup for utils.video_utils
try:
    from utils.video_utils import get_video_info, extract_frames
except ImportError:
    # Fallback if the above doesn't work, assuming utils is one level up from app/models
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- YOLOv8 SERVICE SCRIPT STARTED (URL-Only, FFMPEG, Detailed Logging, Refined RC Handling) ---")

app = FastAPI(title="YOLOv8 Object Detection Service (URL Input Only)", version="1.2.2") # Incremented version
logger.info("FastAPI app object created for YOLOv8 service.")

PADEL_CLASSES = {0: "person", 32: "sports ball", 38: "tennis racket"}
logger.info(f"PADEL_CLASSES defined as: {PADEL_CLASSES}")

GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_yolov8"
MODEL_DIR = "/opt/padel/app/weights"
MODEL_NAME = "yolov8m.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = True
    data: bool = True
    confidence: float = 0.3

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

model = None
try:
    logger.info(f"Attempting to load YOLOv8 model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"CRITICAL: Model file {MODEL_PATH} does not exist. This service requires the model file to be present.")
        logger.error("Please ensure YOLOv8 model is downloaded to weights directory during deployment.")
        model = None
    else:
        model = YOLO(MODEL_PATH)  # Only use manual path, no fallback
    logger.info(f"YOLOv8 model '{MODEL_NAME}' loaded successfully from {MODEL_PATH}.")
    if torch.cuda.is_available():
        logger.info("Moving YOLOv8 model to CUDA device and fusing layers.")
        model.to('cuda')
        model.fuse()
        logger.info("YOLOv8 model on CUDA and fused.")
    else:
        logger.info("CUDA not available. YOLOv8 model will run on CPU.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load YOLOv8 model: {e}", exc_info=True)
    model = None

def draw_objects_on_frame(frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    annotated_frame = frame.copy()
    colors = { PADEL_CLASSES[0]: (0, 255, 0), PADEL_CLASSES[32]: (0, 0, 255), PADEL_CLASSES[38]: (255, 0, 0) }
    for obj in objects:
        x1, y1, x2, y2 = int(obj["bbox"]["x1"]), int(obj["bbox"]["y1"]), int(obj["bbox"]["x2"]), int(obj["bbox"]["y2"])
        class_name, conf = obj["class"], obj["confidence"]
        color = colors.get(class_name, (255, 255, 255))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_frame

@app.get("/healthz")
async def health_check():
    if model is None:
        logger.warning("/healthz: Model not loaded.")
        return JSONResponse(content={"status": "unhealthy", "model": "yolov8 not loaded"}, status_code=503)
    logger.info("/healthz: Model loaded, service healthy.")
    return {"status": "healthy", "model": "yolov8"}

@app.post("/yolov8")
async def detect_objects_in_video(
    payload: VideoAnalysisURLRequest,
    http_request: Request
):
    logger.info(f"--- Enter /yolov8 endpoint (URL-only, FFMPEG) ---")
    logger.info(f"Request Headers: {http_request.headers}")
    logger.info(f"Received Payload: video_url='{payload.video_url}', video_output_requested={payload.video}, data_output_requested={payload.data}")

    if model is None:
        logger.error("/yolov8: Model not loaded. Cannot process request.")
        raise HTTPException(status_code=503, detail="YOLOv8 model is not available.")

    temp_downloaded_path: Optional[str] = None
    try:
        video_url_str = str(payload.video_url)
        logger.info(f"Attempting to download video from URL: {video_url_str}")
        async with httpx.AsyncClient(timeout=60.0) as client: # Download timeout
            response = await client.get(video_url_str)
            response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(response.content)
            temp_downloaded_path = temp_file.name
        logger.info(f"Video downloaded successfully to {temp_downloaded_path}")

        video_info = get_video_info(temp_downloaded_path)
        logger.info(f"Video Info: {video_info}")

        # For timeout testing, keep this low. For production, adjust or remove max_frames.
        num_frames_to_process = -1 
        frames = extract_frames(temp_downloaded_path, num_frames_to_extract=num_frames_to_process)
        logger.info(f"Extracted {len(frames)} (max_frames={num_frames_to_process} for timeout testing) frames.")

        if not frames:
            logger.error(f"No frames extracted from {temp_downloaded_path}.")
            raise HTTPException(status_code=400, detail="No frames could be extracted from the provided video.")

        all_objects_per_frame: List[List[Dict[str, Any]]] = []
        annotated_frames_list: List[np.ndarray] = []
        
        batch_size = 8
        logger.info(f"Starting batch processing of {len(frames)} frames. Batch size: {batch_size}")

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            if not batch_frames: continue
            logger.debug(f"Processing batch {i//batch_size + 1}, frames {i} to {i+len(batch_frames)-1}")
            try:
                results_list = model(batch_frames, verbose=False, half=torch.cuda.is_available())
                logger.debug(f"Batch {i//batch_size + 1}: Inference complete, {len(results_list)} results.")
            except Exception as e_infer:
                logger.error(f"Inference error on batch {i//batch_size + 1}: {e_infer}", exc_info=True)
                for _ in batch_frames: all_objects_per_frame.append([])
                if payload.video: annotated_frames_list.extend(batch_frames)
                continue

            for frame_idx_in_batch, result_obj in enumerate(results_list):
                original_frame_index = i + frame_idx_in_batch
                current_frame_objects: List[Dict[str, Any]] = []
                if result_obj and result_obj.boxes and result_obj.boxes.data is not None:
                    raw_boxes_data = result_obj.boxes.data.cpu().tolist()
                    logger.debug(f"Frame {original_frame_index}: Raw detections: {len(raw_boxes_data)}")
                    for k_det, det_tensor_list in enumerate(raw_boxes_data):
                        x1, y1, x2, y2, conf, cls_raw = det_tensor_list; cls = int(cls_raw)
                        logger.debug(f"  Frame {original_frame_index}, Detection {k_det}: ClassID={cls}, Conf={conf:.2f}")
                        if cls in PADEL_CLASSES and conf > payload.confidence:
                            current_frame_objects.append({
                                "class": PADEL_CLASSES[cls], "confidence": float(conf),
                                "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                            })
                else: logger.debug(f"Frame {original_frame_index}: No detections or invalid result object.")
                all_objects_per_frame.append(current_frame_objects)
                if payload.video:
                    annotated_frames_list.append(draw_objects_on_frame(batch_frames[frame_idx_in_batch], current_frame_objects))
        
        logger.info(f"Finished processing. Object lists: {len(all_objects_per_frame)}. Annotated frames: {len(annotated_frames_list)}.")
        
        response_content: Dict[str, Any] = {}
        if payload.data:
            response_content["data"] = {"objects_per_frame": all_objects_per_frame}

        if payload.video:
            if not annotated_frames_list:
                logger.warning("Video output requested, but no annotated frames.")
                response_content["video_url"] = None
                response_content["message"] = "Video output requested, but no frames to annotate."
            else:
                output_video_path: Optional[str] = None
                gcs_url_result: str = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out_vid_file:
                        output_video_path = temp_out_vid_file.name
                    
                    height, width = annotated_frames_list[0].shape[:2]
                    fps_float = float(video_info.get("fps", 30.0))
                    if fps_float <= 0: fps_float = 30.0
                    
                    logger.info(f"Creating video via FFMPEG: {output_video_path} (H={height}, W={width}, FPS={fps_float})")
                    
                    ffmpeg_cmd = [
                        'ffmpeg', '-y',
                        '-f', 'rawvideo', '-vcodec', 'rawvideo',
                        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24',
                        '-r', str(fps_float), '-i', '-',
                        '-vcodec', 'libx264', '-preset', 'fast',
                        '-crf', '23', '-pix_fmt', 'yuv420p',
                        output_video_path
                    ]
                    
                    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    for frame_to_write in annotated_frames_list:
                        if process.stdin and process.stdin.closed: 
                            logger.warning("FFMPEG stdin pipe closed prematurely. Stopping frame write.")
                            break
                        try:
                            if process.stdin: process.stdin.write(frame_to_write.tobytes())
                        except (IOError, BrokenPipeError) as e_pipe:
                            logger.warning(f"FFMPEG stdin pipe broken while writing frame: {e_pipe}")
                            break
                    
                    if process.stdin and not process.stdin.closed:
                        logger.info("Finished writing frames to FFMPEG stdin. Closing pipe.")
                        process.stdin.close()
                    
                    stdout_bytes, stderr_bytes = b'', b''
                    ffmpeg_timed_out = False
                    ffmpeg_return_code = None # Initialize

                    try:
                        logger.debug("Attempting process.communicate() with timeout...")
                        stdout_bytes, stderr_bytes = process.communicate(timeout=120) # Timeout for ffmpeg processing
                        ffmpeg_return_code = process.returncode
                        logger.debug(f"process.communicate() finished. RC from communicate: {ffmpeg_return_code}")
                    except subprocess.TimeoutExpired:
                        logger.error("FFMPEG process timed out during communicate(). Killing process.")
                        process.kill()
                        try: # Best effort to get output after kill
                            stdout_bytes, stderr_bytes = process.communicate(timeout=5)
                        except Exception: pass # Ignore errors getting output after kill
                        ffmpeg_timed_out = True
                        ffmpeg_return_code = process.returncode # May be None or non-zero
                    except ValueError as ve:
                        if "flush of closed file" in str(ve).lower() or "write to closed file" in str(ve).lower():
                            logger.warning(f"Caught '{ve}' during communicate(). Assuming ffmpeg might have finished or errored. Will try wait/poll.")
                            # Try to wait briefly, then poll for return code
                            try: process.wait(timeout=5) 
                            except subprocess.TimeoutExpired: logger.warning("ffmpeg also timed out on wait() after ValueError.")
                            ffmpeg_return_code = process.poll()
                        else:
                            logger.error(f"Unexpected ValueError during communicate: {ve}", exc_info=True)
                            ffmpeg_return_code = -1 # Indicate error
                            # raise # Re-raise if it's a different ValueError not handled
                    except Exception as e_comm:
                        logger.error(f"Unexpected error during process.communicate(): {e_comm}", exc_info=True)
                        ffmpeg_return_code = -1 # Indicate error

                    # Final check on return code if it wasn't set by communicate() or timeout logic
                    if ffmpeg_return_code is None:
                        logger.warning("ffmpeg_return_code is still None. Waiting briefly for process to finish and polling.")
                        try: process.wait(timeout=5) # Brief wait
                        except subprocess.TimeoutExpired: logger.warning("ffmpeg timed out on final wait.")
                        ffmpeg_return_code = process.poll() # Get what we can

                    if ffmpeg_timed_out: # This takes precedence
                        response_content["message"] = "FFMPEG processing timed out."
                        logger.error(response_content["message"])
                        gcs_url_result = ""
                    elif ffmpeg_return_code != 0:
                        logger.error(f"FFMPEG processing failed (Return Code: {ffmpeg_return_code}):")
                        if stdout_bytes: logger.error(f"FFMPEG stdout: {stdout_bytes.decode(errors='ignore')}")
                        if stderr_bytes: logger.error(f"FFMPEG stderr: {stderr_bytes.decode(errors='ignore')}")
                        gcs_url_result = ""
                        response_content["message"] = f"FFMPEG processing failed. RC: {ffmpeg_return_code}"
                    else: # ffmpeg_return_code == 0
                        logger.info(f"Video created successfully via FFMPEG: {output_video_path}")
                        gcs_url_result = await upload_to_gcs(output_video_path)
                    
                    response_content["video_url"] = gcs_url_result if gcs_url_result else None
                    if not gcs_url_result and "message" not in response_content :
                         response_content["message"] = "Failed to upload annotated video to GCS (or FFMPEG failed without explicit error)."
                finally:
                     if output_video_path and os.path.exists(output_video_path):
                        logger.debug(f"Deleting temporary FFMPEG output video file: {output_video_path}")
                        os.unlink(output_video_path)
        
        if not response_content:
             logger.warning("No output (data or video URL) to return.")
             return JSONResponse(content={"detail": "No output generated based on request flags."}, status_code=200)

        return JSONResponse(content=response_content)
            
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during video download: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to download/access video_url: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Unexpected error in /yolov8 endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            logger.debug(f"Deleting temporary input video file: {temp_downloaded_path}")
            os.unlink(temp_downloaded_path)

@app.post("/yolov8/object")
async def yolov8_object_detection(
    payload: VideoAnalysisURLRequest,
    http_request: Request
):
    """
    YOLOv8 Object Detection Endpoint
    
    Dedicated endpoint for YOLOv8 object detection on padel videos.
    Detects: person (class 0), sports ball (class 32), tennis racket (class 38)
    
    Returns annotated video with bounding boxes and detection data.
    """
    logger.info(f"--- Enter /yolov8/object endpoint ---")
    logger.info(f"Dedicated YOLOv8 object detection for padel analysis")
    
    # Reuse the existing detection logic
    return await detect_objects_in_video(payload, http_request)

@app.post("/yolov8/pose")
async def yolov8_pose_detection(
    payload: VideoAnalysisURLRequest,
    http_request: Request
):
    """
    YOLOv8 Pose Detection Endpoint
    
    Dedicated endpoint for YOLOv8 pose estimation on padel videos.
    Uses YOLOv8 pose model for detecting person poses with 17 keypoints.
    
    Note: Requires YOLOv8 pose model (yolov8n-pose.pt, yolov8s-pose.pt, etc.)
    
    Returns annotated video with skeleton overlay and pose keypoint data.
    """
    logger.info(f"--- Enter /yolov8/pose endpoint ---")
    logger.info(f"Dedicated YOLOv8 pose detection for padel analysis")
    
    if model is None:
        logger.error("/yolov8/pose: Model not loaded. Cannot process request.")
        raise HTTPException(status_code=503, detail="YOLOv8 model is not available.")

    # Check if loaded model supports pose detection
    # YOLOv8 pose models have different architecture - we need to load a pose-specific model
    # For now, let's adapt the existing logic to handle pose detection if the model supports it
    
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

        frames = extract_frames(temp_downloaded_path, num_frames_to_extract=75)
        logger.info(f"Extracted {len(frames)} frames for pose detection.")

        if not frames:
            logger.error(f"No frames extracted from {temp_downloaded_path}.")
            raise HTTPException(status_code=400, detail="No frames could be extracted from the provided video.")

        all_poses_per_frame: List[List[Dict[str, Any]]] = []
        annotated_frames_list: List[np.ndarray] = []
        
        batch_size = 8
        logger.info(f"Starting pose detection for {len(frames)} frames. Batch size: {batch_size}")

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            if not batch_frames: continue
            logger.debug(f"Processing pose batch {i//batch_size + 1}, frames {i} to {i+len(batch_frames)-1}")
            
            try:
                results_list = model(batch_frames, verbose=False, half=torch.cuda.is_available())
                logger.debug(f"Batch {i//batch_size + 1}: Pose inference complete, {len(results_list)} results.")
            except Exception as e_infer:
                logger.error(f"Pose inference error on batch {i//batch_size + 1}: {e_infer}", exc_info=True)
                for _ in batch_frames: all_poses_per_frame.append([])
                if payload.video: annotated_frames_list.extend(batch_frames)
                continue

            for frame_idx_in_batch, result_obj in enumerate(results_list):
                original_frame_index = i + frame_idx_in_batch
                current_frame_poses: List[Dict[str, Any]] = []
                
                # Check if result has keypoints (pose detection)
                if hasattr(result_obj, 'keypoints') and result_obj.keypoints is not None and result_obj.keypoints.data is not None:
                    logger.debug(f"Frame {original_frame_index}: Found keypoints data")
                    for k_idx, keypoints_tensor in enumerate(result_obj.keypoints.data):
                        kpts = keypoints_tensor.cpu().numpy()  # (N_keypoints, 3) - x, y, confidence
                        
                        # Get bounding box if available
                        person_bbox_data = None
                        if hasattr(result_obj, 'boxes') and result_obj.boxes is not None and k_idx < len(result_obj.boxes.data):
                            person_bbox_data = result_obj.boxes.data[k_idx].cpu().numpy()
                        
                        bbox_dict = {}
                        overall_confidence = float(kpts[:, 2].mean()) if kpts.shape[0] > 0 else 0.0
                        
                        if person_bbox_data is not None:
                            x1, y1, x2, y2, box_conf, _ = person_bbox_data
                            bbox_dict = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                            if overall_confidence < 0.1 and box_conf > overall_confidence:
                                overall_confidence = float(box_conf)
                        elif kpts.shape[0] > 0:
                            # Calculate bbox from keypoints
                            valid_kpts = kpts[kpts[:, 2] > 0.05]
                            if len(valid_kpts) > 0:
                                x1, y1 = valid_kpts[:, 0].min(), valid_kpts[:, 1].min()
                                x2, y2 = valid_kpts[:, 0].max(), valid_kpts[:, 1].max()
                                bbox_dict = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                        
                        # Create keypoints dictionary
                        keypoint_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                                        "left_knee", "right_knee", "left_ankle", "right_ankle"]
                        
                        keypoint_dict = {}
                        for kpt_idx, kpt_data in enumerate(kpts):
                            if kpt_idx < len(keypoint_names):
                                x_kpt, y_kpt, conf_kpt = kpt_data
                                keypoint_dict[keypoint_names[kpt_idx]] = {
                                    "x": float(x_kpt), "y": float(y_kpt), "confidence": float(conf_kpt)
                                }
                        
                        if keypoint_dict:
                            current_frame_poses.append({
                                "keypoints": keypoint_dict,
                                "confidence": overall_confidence,
                                "bbox": bbox_dict
                            })
                else:
                    logger.debug(f"Frame {original_frame_index}: No keypoints data - this might not be a pose model")
                
                all_poses_per_frame.append(current_frame_poses)
                
                if payload.video:
                    # Draw poses on frame
                    annotated_frame = batch_frames[frame_idx_in_batch].copy()
                    
                    # Define skeleton connections
                    connections = [
                        ("nose", "left_eye"), ("nose", "right_eye"),
                        ("left_eye", "left_ear"), ("right_eye", "right_ear"),
                        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
                        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
                        ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"),
                        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
                        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
                        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
                        ("left_knee", "left_ankle"), ("right_knee", "right_ankle")
                    ]
                    
                    for pose in current_frame_poses:
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
                        
                        # Draw skeleton connections
                        for p1_name, p2_name in connections:
                            p1, p2 = keypoints.get(p1_name), keypoints.get(p2_name)
                            if (p1 and p2 and p1.get("confidence", 0) > 0.5 and p2.get("confidence", 0) > 0.5):
                                cv2.line(annotated_frame, (int(p1["x"]), int(p1["y"])),
                                        (int(p2["x"]), int(p2["y"])), (0, 255, 255), 2)
                    
                    annotated_frames_list.append(annotated_frame)
        
        logger.info(f"Finished pose detection. Poses collected: {len(all_poses_per_frame)}. Annotated frames: {len(annotated_frames_list)}.")
        
        response_content: Dict[str, Any] = {}
        if payload.data:
            response_content["data"] = {"poses_per_frame": all_poses_per_frame}

        if payload.video:
            if not annotated_frames_list:
                logger.warning("YOLOv8 Pose: Video output requested, but no annotated frames.")
                response_content["video_url"] = None
                response_content["message"] = "YOLOv8 Pose: No frames to annotate."
            else:
                output_video_path: Optional[str] = None
                gcs_url_result: str = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out_vid_file:
                        output_video_path = temp_out_vid_file.name
                    
                    height, width = annotated_frames_list[0].shape[:2]
                    fps_float = float(video_info.get("fps", 30.0))
                    if fps_float <= 0: fps_float = 30.0
                    
                    logger.info(f"YOLOv8 Pose: Creating video via FFMPEG: {output_video_path}")
                    
                    ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                                '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps_float),
                                '-i', '-', '-vcodec', 'libx264', '-preset', 'fast', '-crf', '23',
                                '-pix_fmt', 'yuv420p', output_video_path]
                    
                    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    for frame_to_write in annotated_frames_list:
                        if process.stdin and process.stdin.closed: break
                        try:
                            if process.stdin: process.stdin.write(frame_to_write.tobytes())
                        except (IOError, BrokenPipeError): break
                    
                    if process.stdin and not process.stdin.closed: process.stdin.close()
                    
                    try:
                        stdout_bytes, stderr_bytes = process.communicate(timeout=120)
                        ffmpeg_return_code = process.returncode
                    except subprocess.TimeoutExpired:
                        process.kill()
                        ffmpeg_return_code = -1
                    
                    if ffmpeg_return_code == 0:
                        logger.info(f"YOLOv8 Pose: Video created successfully")
                        gcs_url_result = await upload_to_gcs(output_video_path)
                    else:
                        logger.error(f"YOLOv8 Pose: FFMPEG failed with code {ffmpeg_return_code}")
                        
                    response_content["video_url"] = gcs_url_result if gcs_url_result else None
                finally:
                    if output_video_path and os.path.exists(output_video_path): os.unlink(output_video_path)
        
        if not response_content:
            return JSONResponse(content={"detail": "YOLOv8 Pose: No output."}, status_code=200)
        return JSONResponse(content=response_content)
            
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"YOLOv8 Pose HTTP error: {e.response.status_code}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"YOLOv8 Pose: DL fail: {e.response.status_code}")
    except Exception as e:
        logger.error(f"YOLOv8 Pose: Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"YOLOv8 Pose: Internal error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path): os.unlink(temp_downloaded_path)

if __name__ == "__main__":
    logger.info(f"Attempting to start Uvicorn for YOLOv8 service on port 8002 (PID: {os.getpid()}).")
    if model is None:
        logger.critical("YOLOv8 model could not be loaded at startup. Service will be unhealthy.")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None)
