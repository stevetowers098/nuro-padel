from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import uvicorn
from typing import Dict, Any, Union, Optional
import asyncio
import io
import base64
import json
from pydantic import BaseModel, HttpUrl

app = FastAPI(title="NuroPadel AI", version="1.0.0")

SERVICES = {
    "yolo11": "http://localhost:8001",
    "yolov8": "http://localhost:8005", 
    "yolo_nas": "http://localhost:8002",
    "mmpose": "http://localhost:8003",
    "combined": "http://localhost:8004"
}

# Define request model
class VideoRequest(BaseModel):
    video_url: HttpUrl
    return_video: bool = False
    return_both: bool = False
    service: str = "combined"

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "NuroPadel AI - All Models Ready"}

@app.post("/analyze")
async def full_analysis(
    request: VideoRequest = Body(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Analyze a video using the specified service.
    
    Args:
        request: The request body containing the video URL and parameters
        file: Optional file upload (for backward compatibility)
        
    Returns:
        If return_both is True, returns both JSON data and video content.
        If return_video is True, returns the annotated video as a StreamingResponse.
        If both are False, returns the analysis as JSON.
    """
    # Get parameters from request
    video_url = request.video_url
    return_video = request.return_video
    return_both = request.return_both
    service = request.service
    
    if service not in SERVICES:
        raise HTTPException(status_code=400, detail=f"Invalid service: {service}")
    
    # Determine the endpoint based on the service
    endpoint = "analyze"
    if service == "yolo11" or service == "yolo_nas":
        endpoint = "pose"
    elif service == "yolov8":
        endpoint = "track"
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Prepare the request payload
        json_payload = {
            "video_url": str(video_url),
            "return_video": return_video,
            "return_both": return_both
        }
        
        # If return_both is True, we need to get both JSON and video
        if return_both:
            # Call the service with return_both=true
            response = await client.post(
                f"{SERVICES[service]}/{endpoint}",
                json=json_payload
            )
            
            # Return the combined response
            return response.json()
        else:
            # Call the service with the return_video parameter
            response = await client.post(
                f"{SERVICES[service]}/{endpoint}",
                json=json_payload
            )
            
            # If return_video is True, return the video as a StreamingResponse
            if return_video:
                return StreamingResponse(
                    io.BytesIO(response.content),
                    media_type="video/mp4"
                )
            else:
                # Otherwise, return the JSON response
                return response.json()

@app.post("/pose")
async def pose_estimation(
    request: VideoRequest = Body(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Proxy to YOLO11 pose service.
    
    Args:
        request: The request body containing the video URL and parameters
        file: Optional file upload (for backward compatibility)
    """
    # Create a new request with the updated service parameter
    request = request.copy(update={"service": "yolo11"})
    # Reuse the full_analysis function
    return await full_analysis(request=request, file=file)

@app.post("/track")
async def object_tracking(
    request: VideoRequest = Body(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Proxy to YOLOv8 object tracking service.
    
    Args:
        request: The request body containing the video URL and parameters
        file: Optional file upload (for backward compatibility)
    """
    # Create a new request with the updated service parameter
    request = request.copy(update={"service": "yolov8"})
    # Reuse the full_analysis function
    return await full_analysis(request=request, file=file)

@app.post("/mmpose")
async def biomechanical_analysis(
    request: VideoRequest = Body(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Proxy to MMPose service.
    
    Args:
        request: The request body containing the video URL and parameters
        file: Optional file upload (for backward compatibility)
    """
    # Create a new request with the updated service parameter
    request = request.copy(update={"service": "mmpose"})
    # Reuse the full_analysis function
    return await full_analysis(request=request, file=file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
