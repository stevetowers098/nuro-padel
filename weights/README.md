# NuroPadel AI Models - Manual Management

All models are stored in `/opt/padel/app/weights/` on the VM.

## Required Models:
- `yolo11n-pose.pt` (6MB) - YOLO11 pose detection
- `yolov8m.pt` (52MB) - YOLOv8 object detection  
- `rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth` (53MB) - MMPose poses
- `yolo_nas_pose_m_coco_pose.pth` (TBD) - YOLO-NAS poses (optional)

## Download Commands (VM only):
```bash
cd /opt/padel/app/weights/
wget -O yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
wget -O yolov8m.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt
wget -O rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
