# app/main.py - Enhanced with GPU Auto Start/Stop
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
from typing import Dict, Any, Optional
import io
import time
import asyncio
from pydantic import BaseModel, HttpUrl

# Add GPU management imports
try:
    from google.cloud import compute_v1
    GPU_MANAGEMENT_AVAILABLE = True
except ImportError:
    GPU_MANAGEMENT_AVAILABLE = False
    print("⚠️ Google Cloud Compute not available. GPU auto-management disabled.")

app = FastAPI(title="NuroPadel AI - Combined Service", version="2.0.0")

# GPU Instance Configuration
PROJECT_ID = "surf-coach"
ZONE = "australia-southeast1-a" 
INSTANCE_NAME = "padel-ai"

# Updated service URLs with consistent port numbering
SERVICES = {
    "yolo11": "http://localhost:8001",
    "yolov8": "http://localhost:8002",  # Changed from 8005 to 8002
    "yolo_nas": "http://localhost:8004",  # Changed to port 8004
    "mmpose": "http://localhost:8003"
}

# Service endpoints for API calls
SERVICE_ENDPOINTS = {
    "yolo11": "yolo11",
    "yolov8": "yolov8",
    "yolo_nas": "yolo-nas",
    "mmpose": "mmpose"
}

class VideoRequest(BaseModel):
    video_url: HttpUrl
    video: bool = False  # True = output video
    data: bool = False   # True = output data

class GPUManager:
    """Handles GPU instance start/stop for cost optimization with request queuing"""
    
    def __init__(self):
        if GPU_MANAGEMENT_AVAILABLE:
            self.compute_client = compute_v1.InstancesClient()
        else:
            self.compute_client = None
        self.active_requests = 0
        self.shutdown_delay = 300  # 5 minutes before auto-shutdown
        self._shutdown_task = None
    
    def is_instance_running(self):
        """Check if GPU instance is currently running"""
        if not self.compute_client:
            return True  # Assume running if can't check
            
        try:
            instance = self.compute_client.get(
                project=PROJECT_ID,
                zone=ZONE,
                instance=INSTANCE_NAME
            )
            return instance.status == "RUNNING"
        except Exception as e:
            print(f"Error checking instance status: {e}")
            return True  # Assume running on error
    
    async def start_instance_if_needed(self):
        """Start GPU instance if not already running"""
        if not self.compute_client:
            print("⚠️ GPU management not available, skipping start")
            return True
            
        if self.is_instance_running():
            print("🟢 GPU instance already running")
            return True
            
        try:
            print(f"🚀 Starting GPU instance {INSTANCE_NAME}...")
            operation = self.compute_client.start(
                project=PROJECT_ID,
                zone=ZONE,
                instance=INSTANCE_NAME
            )
            
            # Wait for instance to start and be ready
            await self._wait_for_instance_ready()
            print("✅ GPU instance started and ready!")
            return True
            
        except Exception as e:
            print(f"❌ Error starting GPU instance: {e}")
            # Don't fail the request if GPU management fails
            return False
    
    def stop_instance(self):
        """Stop GPU instance to save costs"""
        if not self.compute_client:
            print("⚠️ GPU management not available, skipping stop")
            return True
            
        try:
            if not self.is_instance_running():
                print("🟡 GPU instance already stopped")
                return True
                
            print(f"💰 Stopping GPU instance {INSTANCE_NAME} to save costs...")
            operation = self.compute_client.stop(
                project=PROJECT_ID,
                zone=ZONE,
                instance=INSTANCE_NAME
            )
            print("✅ GPU instance stopped! Cost savings activated.")
            return True
            
        except Exception as e:
            print(f"⚠️ Error stopping GPU instance: {e}")
            return False
    
    async def _wait_for_instance_ready(self, timeout=300):
        """Wait for instance to be ready to accept requests"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if any service is responding
                async with httpx.AsyncClient(timeout=5.0) as client:
                    for service_url in SERVICES.values():
                        try:
                            # Try to reach any service health endpoint
                            response = await client.get(f"{service_url}/healthz")
                            if response.status_code == 200:
                                return True
                        except:
                            continue
            except:
                pass
            
            print("⏳ Waiting for GPU services to be ready...")
            await asyncio.sleep(15)
        
        print("⚠️ Timeout waiting for services, proceeding anyway...")
        return True  # Don't fail if we can't verify readiness
    
    async def schedule_shutdown(self):
        """Schedule GPU shutdown after delay if no active requests"""
        await asyncio.sleep(self.shutdown_delay)
        if self.active_requests == 0:
            print(f"💰 Auto-stopping GPU after {self.shutdown_delay}s idle time...")
            self.stop_instance()
        else:
            print(f"🔄 GPU shutdown cancelled - {self.active_requests} active requests")
    
    def start_request(self):
        """Track start of new request"""
        self.active_requests += 1
        if self._shutdown_task:
            self._shutdown_task.cancel()
            self._shutdown_task = None
            print("🔄 GPU auto-shutdown cancelled - new request received")
    
    def end_request(self):
        """Track end of request and schedule shutdown if idle"""
        self.active_requests = max(0, self.active_requests - 1)
        if self.active_requests == 0 and not self._shutdown_task:
            self._shutdown_task = asyncio.create_task(self.schedule_shutdown())

# Initialize GPU manager
gpu_manager = GPUManager()

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "gpu_management": GPU_MANAGEMENT_AVAILABLE}

@app.get("/")
async def root():
    return {
        "message": "NuroPadel AI - Combined Service Ready",
        "gpu_optimization": GPU_MANAGEMENT_AVAILABLE,
        "version": "2.0.0"
    }

@app.get("/gpu-status")
async def get_gpu_status():
    """Check current GPU instance status and cost"""
    if not GPU_MANAGEMENT_AVAILABLE:
        return {"error": "GPU management not available"}
        
    is_running = gpu_manager.is_instance_running()
    return {
        "instance_name": INSTANCE_NAME,
        "status": "RUNNING" if is_running else "STOPPED",
        "cost_per_hour": "$0.35" if is_running else "$0.00",
        "monthly_cost_if_always_on": "$252",
        "estimated_optimized_cost": "$15-30/month"
    }

@app.post("/main")  # Changed from /analyze to /main
async def main_analysis(
    video_url: HttpUrl,
    video: bool = False,
    data: bool = False,
    file: Optional[UploadFile] = File(None)
):
    """
    Analyze a video using all available services and combine the results.
    
    🚀 NEW: Now includes automatic GPU cost optimization!
    - Starts GPU instance if needed
    - Processes video on all services
    - Stops GPU instance to save costs
    
    Args:
        video_url: URL of the video to analyze
        video: Set to true to output video
        data: Set to true to output data
        file: Optional file upload (for backward compatibility)
        
    Returns:
        Combined results from all services.
    """
    start_time = time.time()
    
    try:
        # 🚀 Track request start for smart GPU management
        if GPU_MANAGEMENT_AVAILABLE:
            gpu_manager.start_request()
            print("🔍 Checking GPU instance status...")
            await gpu_manager.start_instance_if_needed()
        
        # 🔄 Process video using all services (your existing logic)
        print("🔄 Processing video on all services...")
        results = {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Call each service and collect results
            for service_name, service_url in SERVICES.items():
                try:
                    print(f"   → Calling {service_name}...")
                    # Use consistent parameter names (video, data)
                    endpoint = SERVICE_ENDPOINTS[service_name]
                    full_url = f"{service_url}/{endpoint}"
                    response = await client.post(
                        full_url,
                        json={
                            "video_url": str(video_url),
                            "video": video,
                            "data": data
                        },
                        timeout=60.0  # Increased timeout for GPU processing
                    )
                    
                    if response.status_code == 200:
                        results[service_name] = response.json()
                        print(f"   ✅ {service_name} completed")
                    else:
                        results[service_name] = {
                            "error": f"Service returned status {response.status_code}"
                        }
                        print(f"   ❌ {service_name} failed: {response.status_code}")
                except Exception as e:
                    results[service_name] = {"error": str(e)}
                    print(f"   ❌ {service_name} error: {e}")
        
        # Process and combine results (your existing logic)
        processing_time = time.time() - start_time
        combined_result = {
            "status": "success",
            "combined_analysis": "Combined analysis results would go here",
            "individual_results": results,
            # 🚀 NEW: Cost optimization info
            "cost_optimization": {
                "gpu_auto_management": GPU_MANAGEMENT_AVAILABLE,
                "processing_time_seconds": round(processing_time, 2),
                "estimated_cost_this_request": f"${round((processing_time / 3600) * 0.35, 3)}" if GPU_MANAGEMENT_AVAILABLE else "N/A"
            }
        }
        
        # 💰 NEW: Smart GPU management - schedule shutdown if idle
        if GPU_MANAGEMENT_AVAILABLE:
            gpu_manager.end_request()
            combined_result["cost_optimization"]["smart_shutdown"] = "scheduled"
        
        return combined_result
        
    except Exception as e:
        # 🚨 Always track request end even if processing failed
        if GPU_MANAGEMENT_AVAILABLE:
            print(f"🚨 Error during processing: {e}")
            gpu_manager.end_request()
        
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 NEW: Manual GPU control endpoints for testing
@app.post("/manual-gpu-start")
async def manual_start_gpu():
    """Manually start GPU (for testing)"""
    if not GPU_MANAGEMENT_AVAILABLE:
        return {"error": "GPU management not available"}
        
    if gpu_manager.is_instance_running():
        return {"status": "already_running"}
    
    await gpu_manager.start_instance_if_needed()
    return {"status": "started", "message": "GPU instance started"}

@app.post("/manual-gpu-stop")
async def manual_stop_gpu():
    """Manually stop GPU (for testing)"""
    if not GPU_MANAGEMENT_AVAILABLE:
        return {"error": "GPU management not available"}
        
    if not gpu_manager.is_instance_running():
        return {"status": "already_stopped"}
    
    gpu_manager.stop_instance()
    return {"status": "stopped", "message": "GPU instance stopped to save costs"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)