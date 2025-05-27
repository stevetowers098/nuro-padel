# 🎯 How to Call the 3 YOLO Models Individually

## 🚀 Current Service Architecture

Each YOLO model has its own dedicated service and API endpoint:
- **Separate environments:** Each model type has optimized dependencies
- **Individual services:** Each YOLO variant has its own service endpoint
- **Dedicated APIs:** Clean separation for each model type

## 📡 API Endpoints (Individual Services)

### **Current Services:**
```bash
yolo11-service     - Port 8001 - http://35.189.53.46:8001
yolov8-service     - Port 8002 - http://35.189.53.46:8002
mmpose-service     - Port 8003 - http://35.189.53.46:8003
yolo-nas-service   - Port 8004 - http://35.189.53.46:8004
```

### **API Calls:**
```bash
# YOLOv8 predictions
curl -X POST "http://35.189.53.46:8002/predict" \
  -F "file=@image.jpg" \
  -F "model=yolov8n"

# YOLO11 predictions
curl -X POST "http://35.189.53.46:8001/predict" \
  -F "file=@image.jpg" \
  -F "model=yolo11n"

# YOLO-NAS predictions (dedicated service)
curl -X POST "http://35.189.53.46:8004/predict" \
  -F "file=@image.jpg" \
  -F "model=yolo_nas_s"
```

## 🐍 Python Code (Programmatic Usage)

### **Method 1: Direct Model Loading**
```python
from ultralytics import YOLO

# Load different YOLO models individually
yolov8_model = YOLO('yolov8n.pt')
yolo11_model = YOLO('yolo11n.pt') 
yolo_nas_model = YOLO('yolo_nas_s.pt')

# Use each model individually
image_path = 'test_image.jpg'

# YOLOv8 inference
yolov8_results = yolov8_model(image_path)
print("YOLOv8 Results:", yolov8_results[0].boxes.data)

# YOLO11 inference
yolo11_results = yolo11_model(image_path)
print("YOLO11 Results:", yolo11_results[0].boxes.data)

# YOLO-NAS inference
yolo_nas_results = yolo_nas_model(image_path)
print("YOLO-NAS Results:", yolo_nas_results[0].boxes.data)
```

### **Method 2: Model Factory Pattern**
```python
from ultralytics import YOLO

class YOLOModelManager:
    def __init__(self):
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all YOLO models once"""
        self.models = {
            'yolov8': YOLO('yolov8n.pt'),
            'yolo11': YOLO('yolo11n.pt'),
            'yolo_nas': YOLO('yolo_nas_s.pt')
        }
    
    def predict(self, model_name: str, image_path: str):
        """Predict using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        return self.models[model_name](image_path)

# Usage
manager = YOLOModelManager()

# Call each model individually
yolov8_results = manager.predict('yolov8', 'image.jpg')
yolo11_results = manager.predict('yolo11', 'image.jpg')
yolo_nas_results = manager.predict('yolo_nas', 'image.jpg')
```

## 🔧 Service Configuration Options

### **Option 1: Keep Current Architecture (Recommended)**
Keep separate services for different use cases:
```python
# app/api/routes.py - Different endpoints for different models
from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()

# Load models once at startup
yolov8_model = YOLO('yolov8n.pt')
yolo11_model = YOLO('yolo11n.pt')
yolo_nas_model = YOLO('yolo_nas_s.pt')

@app.post("/predict/yolov8")
async def predict_yolov8(file: UploadFile):
    return yolov8_model(file)

@app.post("/predict/yolo11") 
async def predict_yolo11(file: UploadFile):
    return yolo11_model(file)

@app.post("/predict/yolo_nas")
async def predict_yolo_nas(file: UploadFile):
    return yolo_nas_model(file)
```

### **Option 2: Single Service with Model Parameter**
```python
@app.post("/predict")
async def predict(file: UploadFile, model_type: str = "yolov8"):
    models = {
        'yolov8': yolov8_model,
        'yolo11': yolo11_model, 
        'yolo_nas': yolo_nas_model
    }
    
    if model_type not in models:
        raise HTTPException(400, "Invalid model type")
    
    return models[model_type](file)
```

## 🚀 CLI Usage (All Models)

```bash
# Activate the consolidated YOLO environment
source /opt/padel/yolo/venv/bin/activate

# Use any YOLO model via CLI
yolo predict model=yolov8n.pt source=image.jpg
yolo predict model=yolo11n.pt source=image.jpg  
yolo predict model=yolo_nas_s.pt source=image.jpg

# Export any model to different formats
yolo export model=yolov8n.pt format=onnx
yolo export model=yolo11n.pt format=tensorrt
yolo export model=yolo_nas_s.pt format=onnx
```

## 🎯 Recommended Service Structure

### **Individual API Endpoints:**
```
POST http://35.189.53.46:8001/predict    # YOLO11 service
POST http://35.189.53.46:8002/predict    # YOLOv8 service
POST http://35.189.53.46:8003/predict    # MMPose service
POST http://35.189.53.46:8004/predict    # YOLO-NAS service
```

### **Each Service is Completely Separate:**
- **Dedicated environments** with optimized dependencies
- **Individual model loading** for best performance
- **Clean API separation** for easy integration
- **Independent scaling** and monitoring

## 📋 Example Usage Scripts

### **Test All Models:**
```python
import requests

# Test all YOLO models at their individual endpoints
models_to_test = [
    ("yolov8", "http://35.189.53.46:8002/predict"),
    ("yolo11", "http://35.189.53.46:8001/predict"),
    ("yolo_nas", "http://35.189.53.46:8004/predict")  # Dedicated YOLO-NAS service
]

for model_name, endpoint in models_to_test:
    with open('test_image.jpg', 'rb') as f:
        files = {'file': f}
        if model_name == "yolo_nas":
            data = {'model': 'yolo_nas_s'}
        else:
            data = {'model': f'{model_name}n'}
        
        response = requests.post(endpoint, files=files, data=data)
        print(f"{model_name} Results:", response.json())
```

## 🔍 Current Status Check

```bash
# Check which services are running
sudo systemctl status yolo11-service yolov8-service mmpose-service yolo-nas-service

# Test endpoints
curl http://35.189.53.46:8001/healthz  # YOLO11
curl http://35.189.53.46:8002/healthz  # YOLOv8
curl http://35.189.53.46:8003/healthz  # MMPose
curl http://35.189.53.46:8004/healthz  # YOLO-NAS
```

**Each YOLO model has its own dedicated service and API endpoint** for clean separation and individual access!