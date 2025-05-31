"""
🚀 MEDIUM PRIORITY: FP16 and INT8 Quantization Manager
Advanced quantization with padel-specific calibration
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

try:
    from super_gradients.conversion import ExportQuantizationMode
    from super_gradients.training import models
    SUPER_GRADIENTS_AVAILABLE = True
except ImportError:
    SUPER_GRADIENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class QuantizationManager:
    """🚀 MEDIUM PRIORITY: FP16 and INT8 Quantization with Padel Optimization"""
    
    def __init__(self, weights_dir: str = "/app/weights"):
        self.weights_dir = weights_dir
        self.quantized_models_dir = os.path.join(weights_dir, "quantized")
        os.makedirs(self.quantized_models_dir, exist_ok=True)
        
        logger.info("Quantization Manager initialized")
    
    def apply_fp16_quantization(self, model, model_name: str) -> torch.nn.Module:
        """🚀 MEDIUM PRIORITY: FP16 Quantization - 2x faster inference with minimal accuracy loss"""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, skipping FP16 quantization")
                return model
            
            logger.info(f"Applying FP16 quantization to {model_name}")
            
            # Move model to GPU and apply half precision
            model = model.cuda().half()
            
            # Enable optimization for FP16
            model.eval()
            
            # Verify FP16 conversion
            param_dtype = next(model.parameters()).dtype
            logger.info(f"Model {model_name} quantized to {param_dtype}")
            
            # Save quantized model
            quantized_path = os.path.join(self.quantized_models_dir, f"{model_name}_fp16.pth")
            torch.save(model.state_dict(), quantized_path)
            logger.info(f"FP16 model saved to {quantized_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"FP16 quantization failed for {model_name}: {e}")
            return model
    
    def export_fp16_onnx(self, model, model_name: str, input_shape: Tuple[int, ...] = (1, 3, 640, 640)) -> str:
        """🚀 MEDIUM PRIORITY: Export FP16 ONNX model"""
        try:
            if not SUPER_GRADIENTS_AVAILABLE:
                logger.error("Super Gradients not available for ONNX export")
                return ""
            
            onnx_path = os.path.join(self.quantized_models_dir, f"{model_name}_fp16.onnx")
            
            logger.info(f"Exporting {model_name} to FP16 ONNX format")
            
            # Export with FP16 quantization
            export_result = model.export(
                onnx_path,
                quantization_mode=ExportQuantizationMode.FP16,
                input_image_shape=input_shape[2:],  # (height, width)
                preprocessing=True,
                postprocessing=True
            )
            
            if os.path.exists(onnx_path):
                logger.info(f"✅ FP16 ONNX model exported to {onnx_path}")
                return onnx_path
            else:
                logger.error(f"❌ Failed to export FP16 ONNX model")
                return ""
                
        except Exception as e:
            logger.error(f"FP16 ONNX export failed: {e}")
            return ""
    
    def create_padel_calibration_dataset(self, video_frames: List[np.ndarray], dataset_size: int = 100) -> List[np.ndarray]:
        """🚀 MEDIUM PRIORITY: Create calibration dataset from padel videos for INT8"""
        logger.info(f"Creating padel calibration dataset with {dataset_size} samples")
        
        if not video_frames:
            logger.warning("No video frames provided for calibration dataset")
            return []
        
        # Sample frames evenly from the video
        if len(video_frames) > dataset_size:
            step = len(video_frames) // dataset_size
            calibration_frames = video_frames[::step][:dataset_size]
        else:
            calibration_frames = video_frames
        
        # Preprocess frames for calibration
        processed_frames = []
        for frame in calibration_frames:
            # Resize to model input size
            resized_frame = self._preprocess_frame_for_calibration(frame)
            processed_frames.append(resized_frame)
        
        logger.info(f"Created calibration dataset with {len(processed_frames)} frames")
        return processed_frames
    
    def apply_int8_quantization_with_calibration(
        self, 
        model, 
        model_name: str, 
        calibration_frames: List[np.ndarray]
    ) -> str:
        """🚀 MEDIUM PRIORITY: Advanced INT8 Quantization with Padel Calibration"""
        try:
            if not SUPER_GRADIENTS_AVAILABLE:
                logger.error("Super Gradients not available for INT8 quantization")
                return ""
            
            if not calibration_frames:
                logger.error("No calibration frames provided for INT8 quantization")
                return ""
            
            logger.info(f"Applying INT8 quantization to {model_name} with {len(calibration_frames)} calibration samples")
            
            # Create calibration data loader
            calibration_loader = self._create_calibration_dataloader(calibration_frames)
            
            # Export path
            int8_onnx_path = os.path.join(self.quantized_models_dir, f"{model_name}_int8.onnx")
            
            # Export with INT8 quantization and calibration
            export_result = model.export(
                int8_onnx_path,
                quantization_mode=ExportQuantizationMode.INT8,
                calibration_loader=calibration_loader,
                preprocessing=True,
                postprocessing=True
            )
            
            if os.path.exists(int8_onnx_path):
                logger.info(f"✅ INT8 ONNX model exported to {int8_onnx_path}")
                return int8_onnx_path
            else:
                logger.error(f"❌ Failed to export INT8 ONNX model")
                return ""
                
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            return ""
    
    def export_complete_pipeline_onnx(
        self, 
        model, 
        model_name: str, 
        include_preprocessing: bool = True,
        include_postprocessing: bool = True,
        output_format: str = "batch"
    ) -> str:
        """🎨 ADVANCED: ONNX Export with Preprocessing Pipeline"""
        try:
            if not SUPER_GRADIENTS_AVAILABLE:
                logger.error("Super Gradients not available for pipeline export")
                return ""
            
            pipeline_path = os.path.join(self.quantized_models_dir, f"{model_name}_complete_pipeline.onnx")
            
            logger.info(f"Exporting complete pipeline for {model_name}")
            
            # Determine output format mode
            if output_format == "batch":
                try:
                    from super_gradients.training.utils.predict import DetectionOutputFormatMode
                    output_format_mode = DetectionOutputFormatMode.BATCH_FORMAT
                except ImportError:
                    output_format_mode = None
            else:
                output_format_mode = None
            
            # Export complete pipeline
            export_result = model.export(
                pipeline_path,
                preprocessing=include_preprocessing,
                postprocessing=include_postprocessing,
                output_predictions_format=output_format_mode,
                quantization_mode=ExportQuantizationMode.FP16  # Default to FP16 for compatibility
            )
            
            if os.path.exists(pipeline_path):
                logger.info(f"✅ Complete pipeline exported to {pipeline_path}")
                return pipeline_path
            else:
                logger.error(f"❌ Failed to export complete pipeline")
                return ""
                
        except Exception as e:
            logger.error(f"Pipeline export failed: {e}")
            return ""
    
    def _preprocess_frame_for_calibration(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess frame for calibration dataset"""
        import cv2
        
        # Resize frame
        resized = cv2.resize(frame, target_size)
        
        # Normalize to 0-1 range
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert BGR to RGB if needed
        if len(normalized.shape) == 3 and normalized.shape[2] == 3:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        
        return normalized
    
    def _create_calibration_dataloader(self, calibration_frames: List[np.ndarray]):
        """Create data loader for calibration"""
        try:
            import torch.utils.data as data_utils
            
            # Convert frames to torch tensors
            tensors = []
            for frame in calibration_frames:
                # Convert to tensor and add batch dimension
                tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                tensors.append(tensor)
            
            # Create dataset and dataloader
            dataset = data_utils.TensorDataset(torch.cat(tensors, dim=0))
            dataloader = data_utils.DataLoader(dataset, batch_size=1, shuffle=False)
            
            return dataloader
            
        except Exception as e:
            logger.error(f"Failed to create calibration dataloader: {e}")
            return None
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """Get information about available quantized models"""
        info = {
            "quantized_models_dir": self.quantized_models_dir,
            "available_models": [],
            "quantization_methods": ["FP16", "INT8"],
            "export_formats": ["ONNX", "TensorRT"]
        }
        
        if os.path.exists(self.quantized_models_dir):
            for file in os.listdir(self.quantized_models_dir):
                if file.endswith(('.onnx', '.pth', '.engine')):
                    info["available_models"].append(file)
        
        return info

class CustomNMSManager:
    """🚀 MEDIUM PRIORITY: Custom NMS Parameters for Padel"""
    
    def __init__(self):
        # Padel-optimized NMS parameters
        self.padel_nms_config = {
            "confidence_threshold": 0.3,      # Lower for player detection
            "nms_threshold": 0.5,             # Reduce overlapping detections  
            "max_predictions_per_image": 10,  # Max realistic player count
            "class_agnostic": False,          # Class-specific NMS
            "multi_label": True               # Allow multiple labels per detection
        }
        
        logger.info("Custom NMS Manager initialized with padel-optimized parameters")
    
    def get_nms_config(self) -> Dict[str, Any]:
        """Get current NMS configuration"""
        return self.padel_nms_config.copy()
    
    def update_nms_config(self, new_config: Dict[str, Any]):
        """Update NMS configuration"""
        self.padel_nms_config.update(new_config)
        logger.info(f"NMS configuration updated: {self.padel_nms_config}")