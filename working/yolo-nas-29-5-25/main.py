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
import torch
import uuid
import uvicorn
from datetime import datetime
from google.cloud import storage
import subprocess
import httpx

# Super Gradients for YOLO-NAS
try:
    from super_gradients.training import models
    SUPER_GRADIENTS_AVAILABLE = True
except ImportError:
    SUPER_GRADIENTS_AVAILABLE = False
    logging.warning("Super Gradients not available - service will run in fallback mode")

# Setup for utils.video_utils
try:
    from utils.video_utils import get_video_info, extract_frames
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from utils.video_utils import get_video_info, extract_frames

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("--- YOLO-NAS HIGH-ACCURACY SERVICE STARTED ---")

app = FastAPI(title="YOLO-NAS High-Accuracy Service", version="1.0.0")
logger.info("FastAPI app created for YOLO-NAS service.")

# Configuration
GCS_BUCKET_NAME = "padel-ai"
GCS_FOLDER = "processed_yolo_nas"
WEIGHTS_DIR = "/app/weights"

# Pydantic Models
class VideoAnalysisURLRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False
    data: bool = True
    confidence: float = 0.3

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
yolo_nas_pose_model = None
yolo_nas_object_model = None
model_info = {"pose_model": "none", "object_model": "none", "status": "none"}

if SUPER_GRADIENTS_AVAILABLE:
    try:
        logger.info("Loading YOLO-NAS models...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load YOLO-NAS Pose Model
        try:
            yolo_nas_pose_model = models.get("yolo_nas_pose_n", pretrained_weights="coco_pose")
            if torch.cuda.is_available():
                yolo_nas_pose_model.to('cuda')
                yolo_nas_pose_model.half()  # Use half precision for speed
            model_info["pose_model"] = "yolo_nas_pose_n"
            logger.info("✅ YOLO-NAS Pose model loaded successfully")
        except Exception as e_pose:
            logger.error(f"Failed to load YOLO-NAS pose model: {e_pose}")

        # Load YOLO-NAS Object Model
        try:
            yolo_nas_object_model = models.get("yolo_nas_s", pretrained_weights="coco")
            if torch.cuda.is_available():
                yolo_nas_object_model.to('cuda')
                yolo_nas_object_model.half()  # Use half precision for speed
            model_info["object_model"] = "yolo_nas_s"
            logger.info("✅ YOLO-NAS Object model loaded successfully")
        except Exception as e_object:
            logger.error(f"Failed to load YOLO-NAS object model: {e_object}")

        model_info["status"] = "loaded"

    except Exception as e_init:
        logger.error(f"YOLO-NAS model initialization failed: {e_init}")
        model_info["status"] = "failed"

else:
    logger.critical("Super Gradients not available - YOLO-NAS service will run in fallback mode")
    model_info["status"] = "super_gradients_unavailable"

# Model Functions
def detect_high_accuracy_poses(frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
    """Detect poses using YOLO-NAS pose model"""
    if yolo_nas_pose_model is None:
        logger.warning("YOLO-NAS pose model not loaded")
        return [[] for _ in frames]

    all_poses = []
    batch_size = 8

    try:
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]

            with torch.no_grad():
                results = yolo_nas_pose_model.predict(batch, half=True)

            batch_poses = []
            for result in results:
                frame_poses = []

                # Extract keypoints for each person
                if hasattr(result.prediction, 'poses') and len(result.prediction.poses) > 0:
                    for person_idx in range(len(result.prediction.poses)):
                        keypoints = {}
                        pose_data = result.prediction.poses[person_idx]

                        keypoint_names = [
                            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist", "left_hip", "right_hip",
                            "left_knee", "right_knee", "left_ankle", "right_ankle"
                        ]

                        for kpt_idx, (x, y, conf) in enumerate(pose_data):
                            if kpt_idx < len(keypoint_names) and conf > 0:
                                keypoints[keypoint_names[kpt_idx]] = {
                                    "x": float(x),
                                    "y": float(y),
                                    "confidence": float(conf)
                                }

                        # Calculate bounding box from keypoints
                        if keypoints:
                            valid_kpts = [(kp["x"], kp["y"]) for kp in keypoints.values()]
                            if valid_kpts:
                                x_coords, y_coords = zip(*valid_kpts)
                                x1, y1 = min(x_coords), min(y_coords)
                                x2, y2 = max(x_coords), max(y_coords)

                                frame_poses.append({
                                    "keypoints": keypoints,
                                    "confidence": float(np.mean([kp["confidence"] for kp in keypoints.values()])),
                                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                                })

                batch_poses.append(frame_poses)

            all_poses.extend(batch_poses)

    except Exception as e:
        logger.error(f"Error in YOLO-NAS pose detection: {e}")
        all_poses = [[] for _ in frames]

    return all_poses

def detect_high_accuracy_objects(frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
    """Detect objects using YOLO-NAS object detection model"""
    if yolo_nas_object_model is None:
        logger.warning("YOLO-NAS object model not loaded")
        return [[] for _ in frames]

    all_objects = []
    batch_size = 8
    PADEL_CLASSES = {0: "person", 32: "sports ball", 38: "tennis racket"}

    try:
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]

            with torch.no_grad():
                results = yolo_nas_object_model.predict(batch, half=True)

            batch_objects = []
            for result in results:
                frame_objects = []

                # Extract bounding boxes and classes
                if (hasattr(result.prediction, 'bboxes_xyxy') and
                    hasattr(result.prediction, 'labels') and
                    hasattr(result.prediction, 'confidence')):

                    bboxes = result.prediction.bboxes_xyxy
                    labels = result.prediction.labels
                    confidences = result.prediction.confidence

                    for bbox, label, conf in zip(bboxes, labels, confidences):
                        class_id = int(label)
                        confidence = float(conf)

                        # Filter for padel-relevant classes and confidence threshold
                        if class_id in PADEL_CLASSES and confidence > 0.3:
                            x1, y1, x2, y2 = bbox
                            frame_objects.append({
                                "class": PADEL_CLASSES[class_id],
                                "confidence": confidence,
                                "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                            })

                batch_objects.append(frame_objects)

            all_objects.extend(batch_objects)

    except Exception as e:
        logger.error(f"Error in YOLO-NAS object detection: {e}")
        all_objects = [[] for _ in frames]

    return all_objects

# Drawing Functions
def draw_poses_on_frame(frame: np.ndarray, poses: List[Dict[str, Any]]) -> np.ndarray:
    """Draw poses with skeleton connections"""
    annotated_frame = frame.copy()

    connections = [
        ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"),
        ("right_eye", "right_ear"), ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle")
    ]

    for pose in poses:
        keypoints = pose["keypoints"]

        # Draw bounding box
        if "bbox" in pose:
            bbox = pose["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints
        for keypoint_name, keypoint_data in keypoints.items():
            x, y = int(keypoint_data["x"]), int(keypoint_data["y"])
            confidence = keypoint_data["confidence"]

            if confidence > 0.5:
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                cv2.circle(annotated_frame, (x, y), 5, color, -1)

        # Draw connections
        for start_point_name, end_point_name in connections:
            if (start_point_name in keypoints and end_point_name in keypoints and
                keypoints[start_point_name]["confidence"] > 0.5 and
                keypoints[end_point_name]["confidence"] > 0.5):

                start_x, start_y = int(keypoints[start_point_name]["x"]), int(keypoints[start_point_name]["y"])
                end_x, end_y = int(keypoints[end_point_name]["x"]), int(keypoints[end_point_name]["y"])

                avg_confidence = (keypoints[start_point_name]["confidence"] + keypoints[end_point_name]["confidence"]) / 2
                color = (0, 255, int(255 * (1 - avg_confidence)))
                thickness = max(1, int(3 * avg_confidence))

                cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), color, thickness)

    return annotated_frame

def draw_objects_on_frame(frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    """Draw objects with bounding boxes"""
    annotated_frame = frame.copy()

    colors = {"person": (0, 255, 0), "sports ball": (0, 0, 255), "tennis racket": (255, 0, 0)}

    for obj in objects:
        class_name = obj["class"]
        confidence = obj["confidence"]
        bbox = obj["bbox"]

        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])

        color = colors.get(class_name, (255, 255, 255))

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

        for frame in frames:
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.write(frame.tobytes())
                except (IOError, BrokenPipeError):
                    break

        if process.stdin and not process.stdin.closed:
            process.stdin.close()

        try:
            stdout, stderr = process.communicate(timeout=120)
            if process.returncode == 0:
                logger.info("Video created successfully")
                return await upload_to_gcs(output_video_path)
            else:
                logger.error(f"FFMPEG failed with code {process.returncode}")
                return None
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("FFMPEG timed out")
            return None
    finally:
        if output_video_path and os.path.exists(output_video_path):
            os.unlink(output_video_path)

# API Endpoints
@app.get("/healthz")
async def health_check():
    models_status = {
        "pose_model_loaded": yolo_nas_pose_model is not None,
        "object_model_loaded": yolo_nas_object_model is not None,
        "super_gradients_available": SUPER_GRADIENTS_AVAILABLE
    }

    if not any([yolo_nas_pose_model, yolo_nas_object_model]):
        return JSONResponse(
            content={
                "status": "unhealthy",
                "models": models_status,
                "model_info": model_info
            },
            status_code=503
        )

    return {
        "status": "healthy",
        "models": models_status,
        "model_info": model_info
    }

@app.post("/yolo-nas/pose")
async def yolo_nas_pose_detection(payload: VideoAnalysisURLRequest, request: Request):
    """
    YOLO-NAS High-Accuracy Pose Detection

    Uses YOLO-NAS architecture for high-precision pose estimation.
    Optimized for batch processing with half-precision on GPU.

    Returns 17-keypoint pose data with confidence scores and skeleton visualization.
    """
    logger.info("YOLO-NAS pose detection request received")

    if yolo_nas_pose_model is None:
        raise HTTPException(status_code=503, detail="YOLO-NAS pose model not available")

    return await process_pose_detection(payload)

@app.post("/yolo-nas/object")
async def yolo_nas_object_detection(payload: VideoAnalysisURLRequest, request: Request):
    """
    YOLO-NAS High-Accuracy Object Detection

    Uses YOLO-NAS architecture for high-precision object detection.
    Detects padel-specific objects: person, sports ball, tennis racket.

    Returns bounding box data with confidence scores and annotated visualization.
    """
    logger.info("YOLO-NAS object detection request received")

    if yolo_nas_object_model is None:
        raise HTTPException(status_code=503, detail="YOLO-NAS object model not available")

    return await process_object_detection(payload)

async def process_pose_detection(payload: VideoAnalysisURLRequest):
    """Common pose detection processing"""
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

        # Process poses
        all_poses = detect_high_accuracy_poses(frames)
        annotated_frames = []

        if payload.video:
            for i, frame in enumerate(frames):
                if i < len(all_poses):
                    annotated_frames.append(draw_poses_on_frame(frame, all_poses[i]))

        # Prepare response
        response_data = {}
        if payload.data:
            response_data["data"] = {"poses_per_frame": all_poses}

        if payload.video:
            video_url = await create_video_from_frames(annotated_frames, video_info)
            response_data["video_url"] = video_url

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO-NAS pose error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"YOLO-NAS pose error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)

async def process_object_detection(payload: VideoAnalysisURLRequest):
    """Common object detection processing"""
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

        # Process objects
        all_objects = detect_high_accuracy_objects(frames)
        annotated_frames = []

        if payload.video:
            for i, frame in enumerate(frames):
                if i < len(all_objects):
                    annotated_frames.append(draw_objects_on_frame(frame, all_objects[i]))

        # Prepare response
        response_data = {}
        if payload.data:
            response_data["data"] = {"objects_per_frame": all_objects}

        if payload.video:
            video_url = await create_video_from_frames(annotated_frames, video_info)
            response_data["video_url"] = video_url

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO-NAS object error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"YOLO-NAS object error: {str(e)}")
    finally:
        if temp_downloaded_path and os.path.exists(temp_downloaded_path):
            os.unlink(temp_downloaded_path)

if __name__ == "__main__":
    logger.info("Starting YOLO-NAS service on port 8004")
    if not yolo_nas_pose_model and not yolo_nas_object_model:
        logger.critical("No YOLO-NAS models loaded - service will be unhealthy")
    else:
        logger.info(f"YOLO-NAS service starting with {model_info}")
    uvicorn.run(app, host="0.0.0.0", port=8004, log_config=None)