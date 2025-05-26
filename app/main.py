from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
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

# Define request model - simplified for direct use without request wrapper
class VideoRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False  # True = output video
    data: bool = False   # True = output data
    service: str = "combined"

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "NuroPadel AI - All Models Ready"}

@app.post("/analyze")
async def full_analysis(
    video_url: HttpUrl,
    video: bool = False,
    data: bool = False,
    service: str = "combined",
    file: Optional[UploadFile] = File(None)
):
    """
    Analyze a video using the specified service.
    
    Args:
        video_url: URL of the video to analyze
        video: Set to true to output video
        data: Set to true to output data
        service: Which service to use (default is "combined")
        file: Optional file upload (for backward compatibility)
        
    Returns:
        If both video and data are true, returns both JSON data and video content.
        If only video is true, returns the annotated video as a StreamingResponse.
        If only data is true or both are false, returns the analysis as JSON.
    """
    # Simplified parameter handling
    return_video = video
    return_both = video and data
    
    if service not in SERVICES:
        raise HTTPException(status_code=400, detail=f"Invalid service: {service}")
    
    # Determine the endpoint based on the service
    endpoint = "analyze"
    if service == "yolo11" or service == "yolo_nas":
        endpoint = "pose"
    elif service == "yolov8":
        endpoint = "track"
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Prepare the request payload for the service
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
    video_url: HttpUrl,
    video: bool = False,
    data: bool = False,
    file: Optional[UploadFile] = File(None)
):
    """
    Proxy to YOLO11 pose service.
    
    Args:
        video_url: URL of the video to analyze
        video: Set to true to output video
        data: Set to true to output data
        file: Optional file upload (for backward compatibility)
    """
    # Reuse the full_analysis function with yolo11 service
    return await full_analysis(
        video_url=video_url,
        video=video,
        data=data,
        service="yolo11",
        file=file
    )

@app.post("/track")
async def object_tracking(
    video_url: HttpUrl,
    video: bool = False,
    data: bool = False,
    file: Optional[UploadFile] = File(None)
):
    """
    Proxy to YOLOv8 object tracking service.
    
    Args:
        video_url: URL of the video to analyze
        video: Set to true to output video
        data: Set to true to output data
        file: Optional file upload (for backward compatibility)
    """
    # Reuse the full_analysis function with yolov8 service
    return await full_analysis(
        video_url=video_url,
        video=video,
        data=data,
        service="yolov8",
        file=file
    )

@app.post("/mmpose")
async def biomechanical_analysis(
    video_url: HttpUrl,
    video: bool = False,
    data: bool = False,
    file: Optional[UploadFile] = File(None)
):
    """
    Proxy to MMPose service.
    
    Args:
        video_url: URL of the video to analyze
        video: Set to true to output video
        data: Set to true to output data
        file: Optional file upload (for backward compatibility)
    """
    # Reuse the full_analysis function with mmpose service
    return await full_analysis(
        video_url=video_url,
        video=video,
        data=data,
        service="mmpose",
        file=file
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
