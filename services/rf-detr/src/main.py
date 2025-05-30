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

# RF-DETR imports
try:
    from rfdetr import RFDETRBase
    RF_DETR_AVAILABLE = True
except ImportError:
    RF_DETR_AVAILABLE = False
    logging.warning("RF-DETR not available - service will run in fallback mode")

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
        def get_service_info(self): return {"service": "rf-detr", "version": "1.0.0"}
    def merge_env_overrides(config): return config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- RF-DETR DETECTION SERVICE STARTED ---")

# Initialize configuration loader
config_loader = ConfigLoader("rf-detr", "/app/config")
service_config = merge_env_overrides(config_loader.load_config())
logger.info(f"Configuration loaded: {config_loader.get_service_info()}")

app = FastAPI(title="RF-DETR Detection Service", version="1.0.0")
logger.info("FastAPI app created for RF-DETR service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_rf_detr"
WEIGHTS_DIR = "/app/weights"

# Pydantic Models
class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False
    data: bool = True
    confidence: float = 0.3
    resolution: int = 672  # Must be divisible by 56 for RF-DETR

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure resolution is divisible by 56
        if self.resolution % 56 != 0:
            self.resolution = ((self.resolution // 56) + 1) * 56
            logger.warning(f"Resolution adjusted to {self.resolution} (must be divisible by 56)")

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
rf_detr_model = None
model_info = {"name": "none", "source": "none"}

if RF_DETR_AVAILABLE:
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing RF-DETR model on device: {device}")
        
        # Load RF-DETR with FP16 for VRAM efficiency
        rf_detr_model = RFDETRBase()
        rf_detr_model.to(device)
        
        # Enable FP16 if CUDA available
        if torch.cuda.is_available():
            rf_detr_model.half()
            logger.info("Enabled FP16 precision for VRAM efficiency")
        
        model_info = {"name": "RF-DETR-Base", "source": "official", "precision": "fp16" if torch.cuda.is_available() else "fp32"}
        logger.info("✅ RF-DETR model loaded successfully")
        
    except Exception as e:
        logger.error(f"RF-DETR model initialization failed: {e}", exc_info=True)

if rf_detr_model is None:
    logger.critical("❌ No RF-DETR model could be loaded - service will run in fallback mode")
else:
    logger.info(f"✅ RF-DETR service ready with {model_info['name']} model")

def analyze_frame_detections(frame_content: np.ndarray, confidence_threshold: float = 0.3, resolution: int = 672) -> Dict[str, Any]:
    """Analyze detections for a single frame using RF-DETR"""
    if rf_detr_model is None:
        logger.warning("RF-DETR model not loaded, returning dummy data")
        return {
            "detections": [],
            "detection_metrics": {
                "error_processing_frame": True,
                "model_status": "not_loaded"
            }
        }

    detections = []
    detection_metrics = {}

    try:
        # Resize frame to required resolution (must be divisible by 56)
        height, width = frame_content.shape[:2]
        if width != resolution or height != resolution:
            frame_content = cv2.resize(frame_content, (resolution, resolution))
        
        # Convert BGR to RGB for RF-DETR
        frame_rgb = cv2.cvtColor(frame_content, cv2.COLOR_BGR2RGB)
        
        # Run RF-DETR inference with FP16
        with torch.no_grad():
            if torch.cuda.is_available():
                # Use autocast for FP16
                with torch.cuda.amp.autocast():
                    results = rf_detr_model.predict(frame_rgb, confidence=confidence_threshold)
            else:
                results = rf_detr_model.predict(frame_rgb, confidence=confidence_threshold)
        
        # Process detections
        if results and len(results) > 0:
            for detection in results:
                if hasattr(detection, 'boxes') and detection.boxes is not None:
                    boxes = detection.boxes
                    for i in range(len(boxes)):
                        # Scale coordinates back to original if needed
                        scale_x = width / resolution
                        scale_y = height / resolution
                        
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        detections.append({
                            "class_id": cls,
                            "class_name": rf_detr_model.names.get(cls, f"class_{cls}") if hasattr(rf_detr_model, 'names') else f"class_{cls}",
                            "confidence": conf,
                            "bbox": {
                                "x1": float(box[0] * scale_x),
                                "y1": float(box[1] * scale_y),
                                "x2": float(box[2] * scale_x),
                                "y2": float(box[3] * scale_y),
                                "width": float((box[2] - box[0]) * scale_x),
                                "height": float((box[3] - box[1]) * scale_y)
                            }
                        })
        
        detection_metrics = {
            "total_detections": len(detections),
            "confidence_threshold": confidence_threshold,
            "resolution_used": resolution,
            "model_used": model_info["name"],
            "model_precision": model_info.get("precision", "fp32")
        }

    except Exception as e_analyze:
        logger.error(f"Error during RF-DETR inference: {e_analyze}", exc_info=True)
        return {
            "detections": [],
            "detection_metrics": {
                "error_processing_frame": True,
                "error_message": str(e_analyze),
                "model_status": "inference_failed"
            }
        }
    finally:
        # Clean up GPU memory after inference
        cleanup_gpu_memory()

    return {
        "detections": detections,
        "detection_metrics": detection_metrics
    }

def draw_detections_on_frame(frame: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """Draw detections on frame"""
    annotated_frame = frame.copy()
    detections = analysis.get("detections", [])
    metrics = analysis.get("detection_metrics", {})
    
    # Draw bounding boxes
    for detection in detections:
        bbox = detection["bbox"]
        conf = detection["confidence"]
        class_name = detection["class_name"]
        
        # Draw rectangle
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        
        color = (0, 255, 0)  # Green
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw metrics
    y_offset = 30
    for name, val in metrics.items():
        if isinstance(val, (int, float)):
            text = f"{name}: {val:.2f}" if isinstance(val, float) else f"{name}: {val}"
        else:
            text = f"{name}: {val}"
        cv2.putText(annotated_frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 20
    
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
    model_loaded = rf_detr_model is not None
    gpu_info = get_gpu_memory_info()
    
    # System info
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    response_data = {
        "status": "healthy" if model_loaded else "unhealthy",
        "service": config_loader.get_service_info(),
        "models": {
            "model_loaded": model_loaded,
            "model_info": model_info,
            "rf_detr_available": RF_DETR_AVAILABLE
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
    
    status_code = 200 if model_loaded else 503
    if not model_loaded:
        return JSONResponse(content=response_data, status_code=status_code)
    
    return response_data

@app.post("/analyze")
async def rf_detr_detection_analysis(payload: VideoAnalysisURLRequest, request: Request):
    """
    RF-DETR Detection Analysis Endpoint
    
    High-performance object detection using RF-DETR with:
    - Stable RF-DETR v0.1.0 implementation
    - FP16 precision for VRAM efficiency
    - Resolution constraint enforcement (divisible by 56)
    - GPU memory optimization
    - Configurable confidence thresholds
    
    Ideal for real-time object detection with minimal VRAM usage.
    """
    logger.info("RF-DETR detection analysis request received")

    if rf_detr_model is None:
        raise HTTPException(status_code=503, detail=f"RF-DETR model not available. Info: {model_info}")

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

        logger.info(f"Starting RF-DETR analysis for {len(frames)} frames using {model_info['name']}")

        # Process each frame
        for frame_idx, frame_content in enumerate(frames):
            logger.debug(f"Analyzing frame {frame_idx}")
            analysis_result = analyze_frame_detections(
                frame_content, 
                confidence_threshold=payload.confidence,
                resolution=payload.resolution
            )
            all_analyses.append(analysis_result)

            if payload.video:
                annotated_frames.append(draw_detections_on_frame(frame_content, analysis_result))

        logger.info(f"Finished analysis. Processed: {len(all_analyses)} frames")

        # Prepare response
        response_data = {}

        if payload.data:
            response_data["data"] = {
                "detections_per_frame": all_analyses,
                "model_info": model_info,
                "processing_summary": {
                    "total_frames": len(frames),
                    "successful_analyses": len([a for a in all_analyses
                                              if not a.get("detection_metrics", {}).get("error_processing_frame", False)]),
                    "total_detections": sum(len(a.get("detections", [])) for a in all_analyses),
                    "confidence_threshold": payload.confidence,
                    "resolution_used": payload.resolution
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
        logger.error(f"RF-DETR analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)
        # Final cleanup
        cleanup_gpu_memory()

if __name__ == "__main__":
    logger.info("Starting RF-DETR service on port 8005")
    if rf_detr_model is None:
        logger.critical("RF-DETR model could not be loaded - service will be unhealthy")
    else:
        logger.info(f"RF-DETR service starting with {model_info['name']} model")
    uvicorn.run(app, host="0.0.0.0", port=8005, log_config=None)