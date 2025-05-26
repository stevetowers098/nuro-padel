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

logger.info("--- YOLOv8 SERVICE SCRIPT STARTED (URL-Only, FFMPEG, Detailed Logging, ValueError Fix) ---")

app = FastAPI(title="YOLOv8 Object Detection Service (URL Input Only)", version="1.2.1") # Incremented version
logger.info("FastAPI app object created for YOLOv8 service.")

PADEL_CLASSES = {0: "person", 32: "sports ball", 39: "tennis racket"}
logger.info(f"PADEL_CLASSES defined as: {PADEL_CLASSES}")

GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_yolov8"
MODEL_DIR = "/opt/padel/app/weights"
MODEL_NAME = "yolov8m.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False
    data: bool = True

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
        logger.warning(f"Model file {MODEL_PATH} does not exist. YOLO will attempt to download '{MODEL_NAME}'.")
        os.makedirs(MODEL_DIR, exist_ok=True)
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(video_url_str)
            response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(response.content)
            temp_downloaded_path = temp_file.name
        logger.info(f"Video downloaded successfully to {temp_downloaded_path}")

        video_info = get_video_info(temp_downloaded_path)
        logger.info(f"Video Info: {video_info}")

        # Determine number of frames to process
        # For timeout testing, keep this low. For production, adjust or remove max_frames.
        num_frames_to_process = 30 # Default for testing
        if payload.video: # If video output is true, maybe process more, but still keep it reasonable for now
            num_frames_to_process = 30 # Still 30 for testing to ensure timeout is not the issue
        # elif payload.data: # If only data, could be slightly more
            # num_frames_to_process = 150
        
        frames = extract_frames(temp_downloaded_path, max_frames=num_frames_to_process)
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
                        if cls in PADEL_CLASSES and conf > 0.3:
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
                    
                    # Use Popen with explicit stdin, stdout, stderr pipes
                    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Write frames to FFmpeg's stdin
                    for frame_to_write in annotated_frames_list:
                        if process.stdin.closed: # Check if stdin is already closed
                            logger.warning("FFMPEG stdin pipe closed prematurely. Stopping frame write.")
                            break
                        try:
                            process.stdin.write(frame_to_write.tobytes())
                        except (IOError, BrokenPipeError) as e_pipe:
                            logger.warning(f"FFMPEG stdin pipe broken while writing frame: {e_pipe}")
                            break
                    
                    # Close stdin only if it's not already closed
                    if not process.stdin.closed:
                        logger.info("Finished writing frames to FFMPEG stdin. Closing pipe.")
                        process.stdin.close()
                    
                    # Now communicate, this will wait for the process to terminate
                    stdout_bytes, stderr_bytes = b'', b'' # Initialize
                    try:
                        stdout_bytes, stderr_bytes = process.communicate(timeout=120)
                    except subprocess.TimeoutExpired:
                        logger.error("FFMPEG process timed out during communicate(). Killing process.")
                        process.kill()
                        # Try to get any final output after kill
                        stdout_bytes, stderr_bytes = process.communicate()
                        raise HTTPException(status_code=500, detail="FFMPEG processing timed out")
                    except ValueError as ve: # Catch specific "flush of closed file"
                        if "flush of closed file" in str(ve).lower():
                            logger.warning(f"Caught '{ve}' during communicate, likely due to already closed stdin. Checking return code.")
                            # stdout/stderr might be None or incomplete here, rely on process.poll() or wait() if needed
                            # For now, assume we proceed to check return code.
                            pass # Proceed to check returncode
                        else:
                            logger.error(f"Unexpected ValueError during communicate: {ve}", exc_info=True)
                            raise # Re-raise if it's a different ValueError

                    if process.returncode != 0:
                        logger.error(f"FFMPEG processing failed (return code {process.returncode}):")
                        if stdout_bytes: logger.error(f"FFMPEG stdout: {stdout_bytes.decode(errors='ignore')}")
                        if stderr_bytes: logger.error(f"FFMPEG stderr: {stderr_bytes.decode(errors='ignore')}")
                        gcs_url_result = "" 
                        response_content["message"] = f"FFMPEG processing failed. RC: {process.returncode}"
                    else:
                        logger.info(f"Video created successfully via FFMPEG: {output_video_path}")
                        gcs_url_result = await upload_to_gcs(output_video_path)
                    
                    response_content["video_url"] = gcs_url_result if gcs_url_result else None
                    if not gcs_url_result and "message" not in response_content : 
                         response_content["message"] = "Failed to upload annotated video to GCS."
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

if __name__ == "__main__":
    logger.info(f"Attempting to start Uvicorn for YOLOv8 service on port 8002 (PID: {os.getpid()}).")
    if model is None:
        logger.critical("YOLOv8 model could not be loaded at startup. Service will be unhealthy.")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None)

