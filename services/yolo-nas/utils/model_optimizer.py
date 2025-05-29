"""
Model Optimization Utilities for NuroPadel AI Services
Supports PyTorch -> ONNX -> TensorRT optimization pipeline
"""

import os
import logging
import torch
import numpy as np
from typing import Optional, Union, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional imports for optimization
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("✅ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("⚠️ ONNX Runtime not available")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    logger.info("✅ TensorRT available")
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("⚠️ TensorRT not available")

class ModelOptimizer:
    """Handles model optimization and inference with fallback support"""
    
    def __init__(self, weights_dir: str = "/app/weights"):
        self.weights_dir = Path(weights_dir)
        self.onnx_session = None
        self.trt_engine = None
        self.pytorch_model = None
        self.current_backend = None
        
    def load_optimized_model(self, model_name: str, pytorch_model=None) -> Tuple[str, object]:
        """
        Load the best available optimized model with fallback
        Returns: (backend_name, model_object)
        Priority: TensorRT -> ONNX -> PyTorch
        """
        base_path = self.weights_dir / "optimized" / model_name
        
        # Try TensorRT first (best performance)
        if TENSORRT_AVAILABLE:
            trt_path = base_path.with_suffix('.engine')
            if trt_path.exists():
                try:
                    trt_model = self._load_tensorrt_engine(str(trt_path))
                    if trt_model:
                        logger.info(f"🚀 Using TensorRT engine: {trt_path}")
                        return "tensorrt", trt_model
                except Exception as e:
                    logger.warning(f"Failed to load TensorRT engine: {e}")
        
        # Try ONNX next (good performance, portable)
        if ONNX_AVAILABLE:
            onnx_path = base_path.with_suffix('.onnx')
            if onnx_path.exists():
                try:
                    onnx_session = self._load_onnx_model(str(onnx_path))
                    if onnx_session:
                        logger.info(f"⚡ Using ONNX model: {onnx_path}")
                        return "onnx", onnx_session
                except Exception as e:
                    logger.warning(f"Failed to load ONNX model: {e}")
        
        # Fallback to PyTorch (always available)
        if pytorch_model:
            logger.info(f"🔄 Using PyTorch model (fallback)")
            return "pytorch", pytorch_model
        
        logger.error("No suitable model backend found")
        return "none", None
    
    def _load_onnx_model(self, onnx_path: str):
        """Load ONNX model with GPU acceleration"""
        try:
            providers = []
            if torch.cuda.is_available():
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            session = ort.InferenceSession(onnx_path, providers=providers)
            logger.debug(f"ONNX session created with providers: {session.get_providers()}")
            return session
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            return None
    
    def _load_tensorrt_engine(self, engine_path: str):
        """Load TensorRT engine (simplified - would need full TRT implementation)"""
        try:
            # This is a placeholder - full TensorRT implementation would be more complex
            # Would involve creating TRT runtime, engine, execution context, etc.
            logger.info(f"TensorRT engine loading not fully implemented yet: {engine_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return None
    
    def predict_optimized(self, model_info: Tuple[str, object], input_data: np.ndarray):
        """Run prediction with the optimized model"""
        backend, model = model_info
        
        if backend == "onnx" and model:
            return self._predict_onnx(model, input_data)
        elif backend == "tensorrt" and model:
            return self._predict_tensorrt(model, input_data)
        elif backend == "pytorch" and model:
            return self._predict_pytorch(model, input_data)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _predict_onnx(self, session, input_data: np.ndarray):
        """Run ONNX inference"""
        try:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_data})
            return outputs
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise
    
    def _predict_tensorrt(self, engine, input_data: np.ndarray):
        """Run TensorRT inference (placeholder)"""
        try:
            # Full TensorRT inference implementation would go here
            logger.info("TensorRT inference not fully implemented yet")
            return None
        except Exception as e:
            logger.error(f"TensorRT inference failed: {e}")
            raise
    
    def _predict_pytorch(self, model, input_data: np.ndarray):
        """Run PyTorch inference (fallback)"""
        try:
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data)
                else:
                    input_tensor = input_data
                
                if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                    input_tensor = input_tensor.cuda()
                
                outputs = model(input_tensor)
                return outputs
        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}")
            raise

def export_pytorch_to_onnx(pytorch_model, input_shape: Tuple[int, ...], 
                          output_path: str, dynamic_axes: Optional[dict] = None):
    """Export PyTorch model to ONNX format"""
    try:
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        if torch.cuda.is_available() and next(pytorch_model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        # Set model to eval mode
        pytorch_model.eval()
        
        # Default dynamic axes for batch dimension
        if dynamic_axes is None:
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,  # Compatible with most ONNX Runtime versions
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"✅ PyTorch model exported to ONNX: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to export PyTorch model to ONNX: {e}")
        return False

def export_onnx_to_tensorrt(onnx_path: str, engine_path: str, 
                           max_batch_size: int = 8, fp16: bool = True):
    """Export ONNX model to TensorRT engine (requires trtexec or TensorRT Python API)"""
    try:
        if not TENSORRT_AVAILABLE:
            logger.error("TensorRT not available for export")
            return False
        
        # Using trtexec command (simpler than Python API)
        import subprocess
        
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--maxBatch={max_batch_size}"
        ]
        
        if fp16:
            cmd.append("--fp16")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ ONNX model exported to TensorRT: {engine_path}")
            return True
        else:
            logger.error(f"❌ TensorRT export failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to export ONNX to TensorRT: {e}")
        return False

# Example usage for model export script
def create_model_export_script():
    """Generate a script to export models to optimized formats"""
    script_content = '''#!/bin/bash
# Model Export Script for NuroPadel AI Services
# Run this on the VM with models to create optimized versions

echo "🚀 Starting model optimization for NuroPadel services..."

WEIGHTS_DIR="/opt/padel-docker/weights"
OPTIMIZED_DIR="$WEIGHTS_DIR/optimized"
mkdir -p "$OPTIMIZED_DIR"

echo "📦 Exporting models to ONNX and TensorRT..."

# Export YOLO models (requires running Python export script)
python3 export_models.py

echo "✅ Model optimization complete!"
echo "📁 Optimized models saved to: $OPTIMIZED_DIR"
'''
    return script_content