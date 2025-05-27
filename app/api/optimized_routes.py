from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import httpx
import tempfile
import os
import uuid
from datetime import datetime
from google.cloud import storage

router = APIRouter()

# GCS Configuration
GCS_BUCKET_NAME = "padel-ai"

class VideoRequest(BaseModel):
    video_url: HttpUrl
    video: bool = True      # Return annotated video (includes annotations automatically)
    data: bool = True       # Return detection data
    confidence: float = 0.3 # Confidence threshold

class PoseRequest(VideoRequest):
    model: str = Query(default="yolo11", regex="^(yolo11|mmpose|yolo-nas)$")

class ObjectRequest(VideoRequest):
    model: str = Query(default="yolov8", regex="^(yolov8|yolo11)$")

async def upload_to_gcs(file_path: str, folder: str) -> str:
    """Upload file to GCS and return public URL"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = os.path.basename(file_path)
        object_name = f"{folder}/video_{timestamp}_{unique_id}_{filename}"
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(file_path)
        blob.make_public()
        
        return blob.public_url
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCS upload failed: {str(e)}")

def get_service_url(model: str, endpoint_type: str) -> str:
    """Get service URL based on model and endpoint type"""
    service_mapping = {
        "pose": {
            "yolo11": "http://localhost:8001/yolo11",
            "mmpose": "http://localhost:8003/mmpose", 
            "yolo-nas": "http://localhost:8004/yolo-nas"
        },
        "object": {
            "yolov8": "http://localhost:8002/yolov8",
            "yolo11": "http://localhost:8001/yolo11"
        }
    }
    
    if endpoint_type not in service_mapping or model not in service_mapping[endpoint_type]:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}' for {endpoint_type}")
    
    return service_mapping[endpoint_type][model]

@router.post("/pose")
async def pose_estimation(request: PoseRequest):
    """
    Pose estimation endpoint with model selection and video annotations
    
    Models available:
    - yolo11: YOLO11 pose (fast, good accuracy) - detects person + 17 keypoints
    - mmpose: MMPose (high precision) - person pose with 17 keypoints
    - yolo-nas: YOLO-NAS pose (alternative) - person pose with 17 keypoints
    
    Simplified Input:
    {
        "video_url": "https://example.com/video.mp4",
        "video": true,         // Returns annotated video with pose overlays
        "data": true,          // Returns pose keypoint data
        "confidence": 0.3,     // Detection confidence threshold
        "model": "yolo11"      // Model to use (yolo11|mmpose|yolo-nas)
    }
    
    Returns:
        - Pose keypoints data (if data=True)
        - Annotated video with skeleton overlay (if video=True, uploaded to GCS)
    """
    
    try:
        # Get service URL
        service_url = get_service_url(request.model, "pose")
        
        # Simplified payload - service handles all annotation defaults
        payload = {
            "video_url": str(request.video_url),
            "video": request.video,
            "data": request.data,
            "confidence": request.confidence
        }
        
        # Call specific model service
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(service_url, json=payload)
            response.raise_for_status()
            result = response.json()
        
        # Enhanced response with model info
        enhanced_result = {
            "status": "success",
            "model_used": request.model,
            "endpoint": "pose",
            "processing_info": {
                "confidence_threshold": request.confidence,
                "video_annotations": request.video,
                "target_classes": ["person"],
                "keypoints_detected": 17
            },
            "data": result.get("data") if request.data else None,
            "video_url": result.get("video_url") if request.video else None,
            "gcs_bucket": GCS_BUCKET_NAME if request.video else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=enhanced_result)
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Model service error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pose estimation failed: {str(e)}")

@router.post("/object")
async def object_detection(request: ObjectRequest):
    """
    Object detection endpoint with model selection and video annotations
    
    Models available:
    - yolov8: YOLOv8 medium (proven performance for padel objects)
    - yolo11: YOLO11 (latest version)
    
    Padel-specific classes detected:
    - person (class 0)
    - sports ball (class 32)
    - tennis racket (class 38)
    
    Simplified Input:
    {
        "video_url": "https://example.com/video.mp4",
        "video": true,         // Returns annotated video with bounding boxes
        "data": true,          // Returns object detection data
        "confidence": 0.3,     // Detection confidence threshold
        "model": "yolov8"      // Model to use (yolov8|yolo11)
    }
    
    Returns:
        - Object detection data (if data=True) - players, ball, rackets
        - Annotated video with bounding boxes (if video=True, uploaded to GCS)
    """
    
    try:
        # Get service URL
        service_url = get_service_url(request.model, "object")
        
        # Simplified payload - service handles all annotation defaults
        payload = {
            "video_url": str(request.video_url),
            "video": request.video,
            "data": request.data,
            "confidence": request.confidence
        }
        
        # Call specific model service
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(service_url, json=payload)
            response.raise_for_status()
            result = response.json()
        
        # Enhanced response with model info
        enhanced_result = {
            "status": "success",
            "model_used": request.model,
            "endpoint": "object",
            "processing_info": {
                "confidence_threshold": request.confidence,
                "video_annotations": request.video,
                "target_classes": {
                    "person": 0,
                    "sports ball": 32,
                    "tennis racket": 38
                }
            },
            "data": result.get("data") if request.data else None,
            "video_url": result.get("video_url") if request.video else None,
            "gcs_bucket": GCS_BUCKET_NAME if request.video else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=enhanced_result)
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Model service error: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")

@router.get("/models/status")
async def get_models_status():
    """Get status of all available models for testing"""
    models_status = {}
    
    # Check each service
    services = {
        "yolo11": "http://localhost:8001/healthz",
        "yolov8": "http://localhost:8002/healthz", 
        "mmpose": "http://localhost:8003/healthz",
        "yolo-nas": "http://localhost:8004/healthz"
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for model, health_url in services.items():
            try:
                response = await client.get(health_url)
                models_status[model] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_code": response.status_code,
                    "capabilities": get_model_capabilities(model)
                }
            except Exception as e:
                models_status[model] = {
                    "status": "unavailable",
                    "error": str(e),
                    "capabilities": get_model_capabilities(model)
                }
    
    return {
        "models": models_status,
        "endpoints_available": {
            "pose": ["/pose?model=yolo11", "/pose?model=mmpose", "/pose?model=yolo-nas"],
            "object": ["/object?model=yolov8", "/object?model=yolo11"]
        },
        "gcs_bucket": GCS_BUCKET_NAME
    }

def get_model_capabilities(model: str) -> Dict[str, Any]:
    """Get capabilities of each model for testing"""
    capabilities = {
        "yolo11": {
            "type": ["pose", "object"],
            "classes": ["person", "sports ball", "tennis racket"],
            "keypoints": 17,
            "performance": "fast",
            "accuracy": "good"
        },
        "yolov8": {
            "type": ["object"],
            "classes": ["person", "sports ball", "tennis racket"], 
            "keypoints": 0,
            "performance": "medium",
            "accuracy": "high"
        },
        "mmpose": {
            "type": ["pose"],
            "classes": ["person"],
            "keypoints": 17,
            "performance": "slow",
            "accuracy": "very high"
        },
        "yolo-nas": {
            "type": ["pose"],
            "classes": ["person"],
            "keypoints": 17,
            "performance": "medium",
            "accuracy": "high"
        }
    }
    
    return capabilities.get(model, {"type": "unknown"})

@router.get("/test/examples")
async def get_test_examples():
    """Get simplified example requests for testing different models"""
    return {
        "pose_examples": {
            "yolo11_pose": {
                "url": "/pose",
                "method": "POST",
                "body": {
                    "video_url": "https://example.com/padel_match.mp4",
                    "model": "yolo11",
                    "video": True,
                    "data": True,
                    "confidence": 0.3
                }
            },
            "mmpose_high_precision": {
                "url": "/pose",
                "method": "POST",
                "body": {
                    "video_url": "https://example.com/padel_match.mp4",
                    "model": "mmpose",
                    "video": True,
                    "data": True,
                    "confidence": 0.5
                }
            },
            "yolo_nas_pose": {
                "url": "/pose",
                "method": "POST",
                "body": {
                    "video_url": "https://example.com/padel_match.mp4",
                    "model": "yolo-nas",
                    "video": True,
                    "data": True,
                    "confidence": 0.4
                }
            }
        },
        "object_examples": {
            "yolov8_padel_objects": {
                "url": "/object",
                "method": "POST",
                "body": {
                    "video_url": "https://example.com/padel_match.mp4",
                    "model": "yolov8",
                    "video": True,
                    "data": True,
                    "confidence": 0.3
                }
            },
            "yolo11_padel_objects": {
                "url": "/object",
                "method": "POST",
                "body": {
                    "video_url": "https://example.com/padel_match.mp4",
                    "model": "yolo11",
                    "video": True,
                    "data": True,
                    "confidence": 0.3
                }
            }
        },
        "data_only_examples": {
            "pose_data_only": {
                "url": "/pose",
                "method": "POST",
                "body": {
                    "video_url": "https://example.com/padel_match.mp4",
                    "model": "yolo11",
                    "video": False,
                    "data": True,
                    "confidence": 0.3
                }
            },
            "object_data_only": {
                "url": "/object",
                "method": "POST",
                "body": {
                    "video_url": "https://example.com/padel_match.mp4",
                    "model": "yolov8",
                    "video": False,
                    "data": True,
                    "confidence": 0.3
                }
            }
        },
        "note": "Simplified input format - only video_url, model, video, data, and confidence parameters needed"
    }