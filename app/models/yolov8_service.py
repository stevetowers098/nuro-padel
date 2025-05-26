from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse # StreamingResponse might not be needed if not streaming video back directly
from pydantic import BaseModel, HttpUrl # For request body validation
import tempfile
import os
import cv2
import numpy as np
# import supervision as sv # Only if you use supervision for drawing here; current draw_objects_on_frame uses cv2
from typing import Dict, Any, List, Optional # Union might not be needed anymore
import sys
import logging
# import random # Not used in this version
# import base64 # Not used
# import json # Used implicitly by FastAPI/Pydantic
import httpx
import torch
import uuid
import uvicorn # Make sure this is imported
from datetime import datetime
from google.cloud import storage
from ultralytics import YOLO

# Setup for utils.video_utils
try:
    from utils.video_utils import get_video_info, extract_frames
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Assumes utils is sibling to models dir
    from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- YOLOv8 SERVICE SCRIPT STARTED (URL-Only, Detailed Logging) ---")

app = FastAPI(title="YOLOv8 Object Detection Service (URL Input Only)", version="1.1.0")
logger.info("FastAPI app object created for YOLOv8 service.")

# PADEL_CLASSES for standard COCO
PADEL_CLASSES = {
    0: "person",
    32: "sports ball",
    39: "tennis racket" # Corrected COCO ID
}
logger.info(f"PADEL_CLASSES defined as: {PADEL_CLASSES}")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_yolov8" # Consider a specific subfolder
MODEL_DIR = "/opt/padel/app/weights"
MODEL_NAME = "yolov8m.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# --- Pydantic Model for Request Body ---
class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl  # Validates it's a URL
    video: bool = False # Request annotated video output?
    data: bool = True   # Request JSON data output? (Defaulting to True as primary output)

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
        # Don't raise HTTPException here directly, let the endpoint handle it or return None
        return "" # Indicate failure

# --- Model Loading ---
model = None
try:
    logger.info(f"Attempting to load YOLOv8 model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model file {MODEL_PATH} does not exist. YOLO will attempt to download '{MODEL_NAME}'.")
        os.makedirs(MODEL_DIR, exist_ok=True)
    # YOLO() will download to a cache if only name is given and not found.
    # If MODEL_PATH exists, it uses it. Otherwise, it tries to find/download MODEL_NAME.
    model = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME)
    logger.info(f"YOLOv8 model '{MODEL_NAME}' loaded.")
    if torch.cuda.is_available():
        logger.info("Moving YOLOv8 model to CUDA device and fusing layers.")
        model.to('cuda')
        model.fuse()
        logger.info("YOLOv8 model on CUDA and fused.")
    else:
        logger.info("CUDA not available. YOLOv8 model on CPU.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load YOLOv8 model: {e}", exc_info=True)
    model = None

# --- Core Logic Functions ---
def draw_objects_on_frame(frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    annotated_frame = frame.copy()
    colors = { PADEL_CLASSES[0]: (0, 255, 0), PADEL_CLASSES[32]: (0, 0, 255), PADEL_CLASSES[39]: (255, 0, 0) }
    for obj in objects:
        x1, y1, x2, y2 = int(obj["bbox"]["x1"]), int(obj["bbox"]["y1"]), int(obj["bbox"]["x2"]), int(obj["bbox"]["y2"])
        class_name, conf = obj["class"], obj["confidence"]
        color = colors.get(class_name, (255, 255, 255))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_frame

# --- API Endpoints ---
@app.get("/healthz")
async def health_check():
    if model is None:
        logger.warning("/healthz: Model not loaded.")
        return JSONResponse(content={"status": "unhealthy", "model": "yolov8 not loaded"}, status_code=503)
    logger.info("/healthz: Model loaded, service healthy.")
    return {"status": "healthy", "model": "yolov8"}

@app.post("/yolov8")
async def detect_objects_in_video(
    payload: VideoAnalysisURLRequest, # Request body will be parsed into this Pydantic model
    http_request: Request # For accessing headers, etc.
):
    logger.info(f"--- Enter /yolov8 endpoint (URL-only) ---")
    logger.info(f"Request Headers: {http_request.headers}")
    logger.info(f"Received Payload: video_url='{payload.video_url}', video_output_requested={payload.video}, data_output_requested={payload.data}")

    if model is None:
        logger.error("/yolov8: Model not loaded. Cannot process request.")
        raise HTTPException(status_code=503, detail="YOLOv8 model is not available. Service temporarily unavailable.")

    temp_downloaded_path: Optional[str] = None
    try:
        video_url_str = str(payload.video_url) # Convert HttpUrl to string
        logger.info(f"Attempting to download video from URL: {video_url_str}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(video_url_str)
            response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
        
        # Save downloaded content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(response.content)
            temp_downloaded_path = temp_file.name
        logger.info(f"Video downloaded successfully to {temp_downloaded_path}")

        video_info = get_video_info(temp_downloaded_path)
        logger.info(f"Video Info: {video_info}")

        frames_to_extract_log_info = "all"
        if payload.data and not payload.video: # Only data, less frames
            frames = extract_frames(temp_downloaded_path, max_frames=150) # Example: more frames for data only
            frames_to_extract_log_info = f"{len(frames)} (max_frames=150 for data-only)"
        else: # Video output requested (or data with video), process more/all frames
            frames = extract_frames(temp_downloaded_path, max_frames=900) # Example: more frames if video needed
            frames_to_extract_log_info = f"{len(frames)} (max_frames=900)"
        logger.info(f"Extracted {frames_to_extract_log_info} frames.")

        if not frames:
            logger.error(f"No frames extracted from {temp_downloaded_path}.")
            raise HTTPException(status_code=400, detail="No frames could be extracted from the provided video.")

        all_objects_per_frame: List[List[Dict[str, Any]]] = []
        annotated_frames_list: List[np.ndarray] = []
        
        batch_size = 8 # Consider making this configurable or dynamic based on VRAM
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
                for _ in batch_frames: all_objects_per_frame.append([]) # Append empty for failed frames in batch
                if payload.video: annotated_frames_list.extend(batch_frames) # Add original frames for video output
                continue

            for frame_idx_in_batch, result_obj in enumerate(results_list):
                original_frame_index = i + frame_idx_in_batch
                current_frame_objects: List[Dict[str, Any]] = []
                
                if result_obj and result_obj.boxes and result_obj.boxes.data is not None:
                    raw_boxes_data = result_obj.boxes.data.cpu().tolist() # tolist for easier iteration
                    logger.debug(f"Frame {original_frame_index}: Raw detections: {len(raw_boxes_data)}")
                    for k_det, det_tensor_list in enumerate(raw_boxes_data):
                        x1, y1, x2, y2, conf, cls_raw = det_tensor_list
                        cls = int(cls_raw)
                        logger.debug(f"  Frame {original_frame_index}, Detection {k_det}: ClassID={cls}, Conf={conf:.2f}")
                        if cls in PADEL_CLASSES:
                            current_frame_objects.append({
                                "class": PADEL_CLASSES[cls], "confidence": float(conf),
                                "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                            })
                else:
                    logger.debug(f"Frame {original_frame_index}: No detections or invalid result object.")
                
                all_objects_per_frame.append(current_frame_objects)
                if payload.video: # Only annotate if video output is requested
                    annotated_frames_list.append(draw_objects_on_frame(batch_frames[frame_idx_in_batch], current_frame_objects))
        
        logger.info(f"Finished processing all frames. Object lists: {len(all_objects_per_frame)}. Annotated frames: {len(annotated_frames_list)}.")
        
        # Prepare response
        response_content: Dict[str, Any] = {}
        if payload.data:
            response_content["data"] = {"objects_per_frame": all_objects_per_frame}

        if payload.video:
            if not annotated_frames_list:
                logger.warning("Video output requested, but no annotated frames available.")
                response_content["video_url"] = None
                response_content["message"] = "Video output requested, but no frames were annotated (e.g., no objects found or error)."
            else:
                output_video_path: Optional[str] = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out_vid:
                        output_video_path = temp_out_vid.name
                    
                    height, width = annotated_frames_list[0].shape[:2]
                    fps = video_info.get("fps", 30.0)
                    if not isinstance(fps, (int, float)) or fps <= 0: fps = 30.0
                    
                    logger.info(f"Creating annotated video at {output_video_path} (H={height}, W={width}, FPS={float(fps)})")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), (width, height))
                    for frame_to_write in annotated_frames_list:
                        video_writer.write(frame_to_write)
                    video_writer.release()
                    logger.info(f"Annotated video created: {output_video_path}")
                    
                    gcs_url = await upload_to_gcs(output_video_path)
                    if gcs_url:
                        response_content["video_url"] = gcs_url
                    else:
                        response_content["video_url"] = None
                        response_content["message"] = "Failed to upload annotated video to GCS."
                finally:
                    if output_video_path and os.path.exists(output_video_path):
                        os.unlink(output_video_path)
        
        if not response_content: # Should not happen if data defaults to True
             logger.warning("No output type (data or video) resulted in content. This is unexpected.")
             return JSONResponse(content={"detail": "No output generated based on request flags."}, status_code=200) # Or 204

        return JSONResponse(content=response_content)
            
    except HTTPException: # Re-raise HTTPExceptions raised by our code
        raise
    except httpx.HTTPStatusError as e: # Specifically catch errors from httpx.get().raise_for_status()
        logger.error(f"HTTP error during video download: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to download or access video_url: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Unexpected error in /yolov8 endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing video: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            logger.debug(f"Deleting temporary input video file: {temp_downloaded_path}")
            os.unlink(temp_downloaded_path)

if __name__ == "__main__":
    logger.info(f"Attempting to start Uvicorn for YOLOv8 service on port 8002 (PID: {os.getpid()}).")
    if model is None:
        logger.critical("YOLOv8 model could not be loaded at startup. The service will report as unhealthy and fail requests.")
    
    uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None)
