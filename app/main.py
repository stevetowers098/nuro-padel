from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
from typing import Dict, Any, Optional
import io
from pydantic import BaseModel, HttpUrl

app = FastAPI(title="NuroPadel AI - Combined Service", version="1.0.0")

# Updated service URLs with consistent port numbering and endpoints
SERVICES = {
    "yolo11": "http://localhost:8001/yolo11",
    "yolov8": "http://localhost:8002/yolov8",  # Changed from 8005 to 8002
    "yolo_nas": "http://localhost:8004/yolo-nas",  # Changed to port 8004
    "mmpose": "http://localhost:8003/mmpose"
}

class VideoRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False  # True = output video
    data: bool = False   # True = output data

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "NuroPadel AI - Combined Service Ready"}

@app.post("/main")  # Changed from /analyze to /main
async def main_analysis(
    video_url: HttpUrl,
    video: bool = False,
    data: bool = False,
    file: Optional[UploadFile] = File(None)
):
    """
    Analyze a video using all available services and combine the results.
    
    Args:
        video_url: URL of the video to analyze
        video: Set to true to output video
        data: Set to true to output data
        file: Optional file upload (for backward compatibility)
        
    Returns:
        Combined results from all services.
    """
    results = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Call each service and collect results
        for service_name, service_url in SERVICES.items():
            try:
                # Use consistent parameter names (video, data)
                response = await client.post(
                    service_url,
                    json={
                        "video_url": str(video_url),
                        "video": video,
                        "data": data
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    results[service_name] = response.json()
                else:
                    results[service_name] = {
                        "error": f"Service returned status {response.status_code}"
                    }
            except Exception as e:
                results[service_name] = {"error": str(e)}
    
    # Process and combine results
    # This would be where you implement your combined analysis logic
    combined_result = {
        "combined_analysis": "Combined analysis results would go here",
        "individual_results": results
    }
    
    return combined_result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)