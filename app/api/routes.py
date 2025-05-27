from fastapi import APIRouter
import httpx

router = APIRouter()

@router.get("/models/status")
async def get_models_status():
    services = {
        "yolo11": "http://localhost:8001",
        "yolov8": "http://localhost:8002",
        "yolo_nas": "http://localhost:8004",
        "mmpose": "http://localhost:8003"
    }
    
    status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services.items():
            try:
                response = await client.get(f"{url}/healthz")
                status[name] = response.json()
            except:
                status[name] = {"status": "unavailable"}
    return status
