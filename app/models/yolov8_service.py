import logging
import uvicorn
from fastapi import FastAPI
import sys # For checking Python path if needed

# Configure basic logging to ensure it goes to stdout/stderr
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG for more info
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Explicitly use StreamHandler to stdout
)
logger = logging.getLogger(__name__)

logger.info("--- YOLOV8 TEST SCRIPT STARTED ---")
logger.info(f"Python sys.path: {sys.path}") # Log python path

app = FastAPI(title="Test YOLOv8 Logging Service")
logger.info("Test YOLOv8 FastAPI app object created.")

@app.get("/healthz")
async def health_check():
    logger.info("Test /healthz endpoint called.")
    return {"status": "healthy", "model": "test_yolov8_logging"}

@app.post("/yolov8") # Keep an endpoint with the same name as your original
async def test_endpoint():
    logger.info("Test /yolov8 endpoint called.")
    logger.info("Test /yolov8 endpoint finishing.")
    return {"message": "Test YOLOv8 response from simplified script"}

if __name__ == "__main__":
    logger.info("Inside __main__ block. Preparing to start Uvicorn for Test YOLOv8 service on port 8002.")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None) # log_config=None to prevent Uvicorn from overriding our basicConfig
    except Exception as e:
        logger.error(f"Uvicorn failed to start: {e}", exc_info=True)
        # Consider sys.exit(1) here if Uvicorn failure is critical
        raise
else:
    logger.info("Script is NOT being run as __main__.") # Should not happen with `python -m models.yolov8_service`
