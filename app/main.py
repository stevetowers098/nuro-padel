from fastapi import FastAPI, UploadFile, File, HTTPException
import httpx
import uvicorn
from typing import Dict, Any
import asyncio

app = FastAPI(title="NuroPadel AI", version="1.0.0")

SERVICES = {
    "yolo11": "http://localhost:8001",
    "yolov8": "http://localhost:8005", 
    "yolo_nas": "http://localhost:8002",
    "mmpose": "http://localhost:8003"
}

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "NuroPadel AI - All Models Ready"}

@app.post("/analyze")
async def full_analysis(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=300.0) as client:
        tasks = [
            client.post(f"{SERVICES['yolo11']}/pose", files={"file": file.file}),
            client.post(f"{SERVICES['yolov8']}/track", files={"file": file.file})
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return {"video_analysis": {"poses": results[0].json() if not isinstance(results[0], Exception) else None}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
