from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
from typing import Dict, Any
import asyncio
import io

app = FastAPI(title="NuroPadel AI", version="1.0.0")

SERVICES = {
    "yolo11": "http://localhost:8001",
    "yolov8": "http://localhost:8005", 
    "yolo_nas": "http://localhost:8002",
    "mmpose": "http://localhost:8003",
    "combined": "http://localhost:8004"
}

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "NuroPadel AI - All Models Ready"}

@app.post("/analyze")
async def full_analysis(file: UploadFile = File(...), return_video: bool = False, service: str = "combined"):
    """
    Analyze a video using the specified service.
    
    Args:
        file: The input video file
        return_video: Whether to return the annotated video (default: False)
        service: The service to use for analysis (default: "combined")
        
    Returns:
        If return_video is False, returns the analysis as JSON.
        If return_video is True, returns the annotated video as a StreamingResponse.
    """
    if service not in SERVICES:
        raise HTTPException(status_code=400, detail=f"Invalid service: {service}")
    
    # Determine the endpoint based on the service
    endpoint = "analyze"
    if service == "yolo11" or service == "yolo_nas":
        endpoint = "pose"
    elif service == "yolov8":
        endpoint = "track"
    
    # Read the file content
    file_content = await file.read()
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Call the service with the return_video parameter
        response = await client.post(
            f"{SERVICES[service]}/{endpoint}?return_video={return_video}",
            files={"file": ("video.mp4", file_content, "video/mp4")}
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
