from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import tempfile
import os
import cv2
import numpy as np
import supervision as sv
from typing import Dict, Any, List, Union, Optional
import io
import sys
import logging
import random
import base64
import json
import httpx
import torch
import uuid
from datetime import datetime
from google.cloud import storage
from ultralytics import YOLO

# Add parent directory to path to import utils
# Ensure this path is correct for your project structure
# If utils is in the same directory as models, this might be:
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Or if utils is one level up from app:
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Assuming 'utils' is a sibling directory to 'models' within 'app'
# For `python -m models.yolov8_service` from `/opt/padel/app`, utils should be importable if in PYTHONPATH
try:
    from utils.video_utils import get_video_info, extract_frames
except ImportError:
    # Fallback if the above doesn't work, assuming utils is one level up from app/models
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.video_utils import get_video_info, extract_frames


# Configure logging
# More verbose logging format and explicitly send to stdout for journald
logging.basicConfig(
    level=logging.DEBUG, # Changed to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] 
)
logger = logging.getLogger(__name__)

logger.info("--- YOLOv8 SERVICE SCRIPT STARTED (DETAILED LOGGING ENABLED) ---")

app = FastAPI(title="YOLOv8 Object Detection Service", version="1.0.0")
logger.info("FastAPI app object created for YOLOv8 service.")

# CORRECTED PADEL_CLASSES for standard COCO
PADEL_CLASSES = {
    0: "person",        # Players
    32: "sports ball",  # Padel ball  
    39: "tennis racket" # Correct COCO ID for tennis racket
}
logger.info(f"PADEL_CLASSES defined as: {PADEL_CLASSES}")

# Google Cloud Storage configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed"
MODEL_DIR = "/opt/padel/app/weights" # Define model directory
MODEL_NAME = "yolov8m.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

async def upload_to_gcs(video_path: str) -> str:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{GCS_FOLDER}/video_{timestamp}_{unique_id}.mp4"
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        blob.upload_from_filename(video_path)
        blob.make_public()
        logger.info(f"Successfully uploaded {video_path} to GCS: {blob.public_url}")
        return blob.public_url
    except Exception as e:
        logger.error(f"Error uploading {video_path} to GCS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading to GCS: {str(e)}")

# Load the YOLOv8 model
model = None
try:
    logger.info(f"Attempting to load YOLOv8 model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model file {MODEL_PATH} does not exist. YOLO will attempt to download it.")
        # Ensure the directory exists for download
        os.makedirs(MODEL_DIR, exist_ok=True)
        # YOLO will download to current dir if path is just filename, or to specified path
        model = YOLO(MODEL_NAME) # Let YOLO download to default or cache if path is just filename
        # If you want it specifically in MODEL_PATH, ensure YOLO() handles full path for download or download manually.
        # Forcing download to specific path usually involves a manual download step.
        # For now, we assume YOLO handles 'yolov8m.pt' by downloading to a cache or local dir if not found.
        # To be certain, you might need a dedicated download step if MODEL_PATH is critical for first load.
        # Forcing model load to a specific path after download might be:
        # if not os.path.exists(MODEL_PATH): model.save(MODEL_PATH)
        # For now, let's use the direct model name and assume Ultralytics handles caching/download path
        # This will make it download to a default cache or current working directory if not found by name.
    model = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME)

    logger.info(f"YOLOv8 model '{MODEL_NAME}' loaded successfully.")
    if torch.cuda.is_available():
        logger.info("Moving YOLOv8 model to CUDA device and fusing layers.")
        model.to('cuda')
        model.fuse() 
        logger.info("YOLOv8 model moved to CUDA and fused.")
    else:
        logger.info("CUDA not available. YOLOv8 model will run on CPU.")
except Exception as e:
    logger.error(f"Failed to load YOLOv8 model: {e}", exc_info=True)
    model = None # Ensure model is None if loading fails

# track_objects function is not used by the /yolov8 endpoint, but keeping it if used elsewhere or for reference
# def track_objects(frame: np.ndarray) -> List[Dict[str, Any]]: ...

def draw_objects_on_frame(frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    annotated_frame = frame.copy()
    colors = {
        PADEL_CLASSES[0]: (0, 255, 0),  
        PADEL_CLASSES[32]: (0, 0, 255), 
        PADEL_CLASSES[39]: (255, 0, 0) # Corrected for tennis racket
    }
    for obj in objects:
        x1, y1 = int(obj["bbox"]["x1"]), int(obj["bbox"]["y1"])
        x2, y2 = int(obj["bbox"]["x2"]), int(obj["bbox"]["y2"])
        class_name = obj["class"]
        track_id = obj.get("track_id", "N/A") # Make track_id optional
        confidence = obj["confidence"]
        color = colors.get(class_name, (255, 255, 255))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} ({confidence:.2f})" # Simplified label
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_frame

@app.get("/healthz")
async def health_check():
    if model is None:
        logger.warning("/healthz called, but model is not loaded.")
        return {"status": "unhealthy", "model": "yolov8 not loaded"}
    logger.info("/healthz called, model is loaded.")
    return {"status": "healthy", "model": "yolov8"}

@app.post("/yolov8")
async def detect_objects(
    file: Optional[UploadFile] = File(None), 
    video_url: Optional[str] = None,
    video: bool = False, 
    data: bool = False
):
    logger.info(f"Received request for /yolov8. video_url: {video_url}, file provided: {file is not None}, video output: {video}, data output: {data}")
    if model is None:
        logger.error("YOLOv8 model is not loaded. Cannot process request for /yolov8.")
        raise HTTPException(status_code=503, detail="YOLOv8 model not available")

    temp_path_local = None # Initialize for finally block
    try:
        if video_url:
            logger.info(f"Downloading video from URL: {video_url}")
            try:
                async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout for download
                    response = await client.get(video_url)
                    response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(response.content)
                    temp_path_local = temp_file.name
                logger.info(f"Video downloaded successfully to {temp_path_local}")
            except Exception as e:
                logger.error(f"Failed to download video from URL {video_url}: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")
        elif file:
            logger.info(f"Processing uploaded file: {file.filename}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path_local = temp_file.name
            logger.info(f"Uploaded file saved to {temp_path_local}")
        else:
            logger.error("Missing file or video_url in /yolov8 request.")
            raise HTTPException(status_code=400, detail="Either file or video_url is required")
        
        video_info = get_video_info(temp_path_local)
        logger.info(f"Processing video: {temp_path_local}, Info: {video_info}")
        
        frames_to_extract_count_for_log = "all"
        if data and not video:
            frames = extract_frames(temp_path_local, sample_every=3)
            frames_to_extract_count_for_log = f"{len(frames)} (sampled every 3rd)"
        else:
            frames = extract_frames(temp_path_local)
            frames_to_extract_count_for_log = f"{len(frames)} (all)"
        logger.info(f"Extracted {frames_to_extract_count_for_log} frames.")

        if not frames:
            logger.error(f"No frames were extracted from the video: {temp_path_local}.")
            if data and video: return {"data": {"objects": []}, "video_url": None, "error": "No frames extracted"}
            elif data: return {"objects": [], "error": "No frames extracted"}
            elif video: return {"video_url": None, "error": "No frames extracted"}
            else: return HTTPException(status_code=400, detail="No frames extracted from video")

        all_objects_per_frame = [] # Renamed for clarity
        annotated_frames_list = [] # Renamed for clarity
        
        batch_size = 8 
        logger.info(f"Starting batch processing of {len(frames)} frames. Batch size: {batch_size}")
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+len(batch)] # Corrected batch slicing
            if not batch:
                logger.warning(f"Batch at index {i} is empty. Skipping.")
                continue
            
            logger.debug(f"Processing batch starting at frame index {i}, actual batch size {len(batch)}")
            
            try:
                results_list = model(batch, verbose=False, half=torch.cuda.is_available()) # Use half precision only if CUDA is available
                logger.debug(f"YOLOv8 inference complete for batch. Number of result objects from model: {len(results_list)}")
            except Exception as e_infer:
                logger.error(f"Exception during YOLOv8 model inference for batch starting at frame index {i}: {e_infer}", exc_info=True)
                # Fill with empty results for this batch's frames if inference fails
                for _ in range(len(batch)): all_objects_per_frame.append([])
                if video or data: annotated_frames_list.extend(batch) # Add original frames
                continue 

            if not results_list:
                logger.warning(f"YOLOv8 model returned no results (empty list or None) for batch starting at frame index {i}.")
                for k_batch_idx in range(len(batch)):
                    all_objects_per_frame.append([])
                    if video or data: annotated_frames_list.append(frames[i+k_batch_idx])
                continue

            for j, result_obj in enumerate(results_list):
                current_frame_index = i + j
                if current_frame_index >= len(frames):
                    logger.warning(f"Result index {j} for batch {i} is out of bounds for total frames {len(frames)}. Breaking from inner loop.")
                    break
                
                logger.debug(f"--- Processing frame {current_frame_index} (Original Batch Index {j}) ---")
                
                if result_obj is None or result_obj.boxes is None or result_obj.boxes.data is None:
                    logger.warning(f"Result object or boxes data is None for frame {current_frame_index}. Skipping detection processing.")
                    all_objects_per_frame.append([])
                    if video or data: annotated_frames_list.append(frames[current_frame_index])
                    continue

                raw_boxes_data = result_obj.boxes.data
                logger.debug(f"Raw result.boxes.data for frame {current_frame_index} (shape: {raw_boxes_data.shape}): {raw_boxes_data}")

                current_frame_objects = []
                if raw_boxes_data.shape[0] == 0:
                    logger.debug(f"No objects tensor (result.boxes.data is empty) in frame {current_frame_index}.")
                
                for k_det, det_tensor in enumerate(raw_boxes_data.tolist()): 
                    x1, y1, x2, y2, conf, cls_raw = det_tensor 
                    cls = int(cls_raw)
                    
                    # Log every single detection BEFORE filtering
                    logger.debug(f"Frame {current_frame_index}: Raw Detection {k_det}: ClassID={cls}, Conf={conf:.2f}, BBox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

                    if cls in PADEL_CLASSES:
                        logger.debug(f"Frame {current_frame_index}: Object with Class ID {cls} ({PADEL_CLASSES[cls]}) IS IN PADEL_CLASSES. Adding it.")
                        current_frame_objects.append({
                            "class": PADEL_CLASSES[cls],
                            "confidence": float(conf),
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            # "track_id": k_det + 1 # Simple track_id if needed later, but not essential for detection output
                        })
                    else:
                        logger.debug(f"Frame {current_frame_index}: Object with Class ID {cls} IS NOT in PADEL_CLASSES. Skipping it.")
                
                all_objects_per_frame.append(current_frame_objects)
                if not current_frame_objects:
                    logger.debug(f"No PADEL_RELEVANT objects found in frame {current_frame_index} after filtering.")
                
                if video or data:
                    annotated_frame = draw_objects_on_frame(frames[current_frame_index], current_frame_objects)
                    annotated_frames_list.append(annotated_frame)
        
        logger.info(f"Finished processing all frames. Total object lists collected: {len(all_objects_per_frame)}. Total annotated frames: {len(annotated_frames_list)}")
        
        json_response = {"objects_per_frame": all_objects_per_frame} # Changed key for clarity
        
        if video or data:
            if not annotated_frames_list:
                 logger.warning("No frames were available/annotated to create a video.")
                 if data: return {"data": json_response, "video_url": None, "message": "No frames to create video."}
                 return {"video_url": None, "message": "No frames to create video."}

            output_path_local = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            try:
                height, width = annotated_frames_list[0].shape[:2]
                fps = video_info.get("fps", 30.0) 
                if not fps or fps == 0: fps = 30.0
                
                logger.info(f"Creating video: {output_path_local} with H={height}, W={width}, FPS={fps}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path_local, fourcc, float(fps), (width, height))
                
                for frame_to_write in annotated_frames_list:
                    out.write(frame_to_write)
                out.release()
                logger.info(f"Video created successfully: {output_path_local}")
                
                video_gcs_url = await upload_to_gcs(output_path_local)
            finally: # Ensure cleanup of local video file
                 if os.path.exists(output_path_local):
                    logger.debug(f"Deleting temporary output video file: {output_path_local}")
                    os.unlink(output_path_local)
            
            if data:
                return {"data": json_response, "video_url": video_gcs_url}
            else: # only video=True
                return {"video_url": video_gcs_url}
        else: # only data=False, video=False (though your endpoint description implies at least one)
              # This case might need clearer definition or could default to JSON
            logger.info("Only JSON data requested (or video=False, data=False).")
            return json_response 
    
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Critical error in /yolov8 endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if temp_path_local and os.path.exists(temp_path_local):
            logger.debug(f"Deleting temporary input video file: {temp_path_local}")
            os.unlink(temp_path_local)

if __name__ == "__main__":
    logger.info("Starting Uvicorn for YOLOv8 service on port 8002.")
    # Ensure model is loaded before starting, or handle it gracefully in /healthz and /yolov8
    if model is None:
        logger.critical("YOLOv8 model could not be loaded at startup. Service will be unhealthy.")
        # Depending on policy, you might sys.exit(1) here.
        # For now, let it start so /healthz can report unhealthy.
    
    # Use log_config=None to prevent Uvicorn from overriding basicConfig
    uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None)
