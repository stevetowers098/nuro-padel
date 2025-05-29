"""
TrackNet Inference Engine
Ball tracking logic with YOLO integration
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
import logging

from .model import TrackNet, load_tracknet_model
from .utils import (
    preprocess_frames_for_tracknet, 
    postprocess_tracknet_output,
    smooth_ball_trajectory,
    merge_yolo_tracknet_detections
)

logger = logging.getLogger(__name__)


class TrackNetInference:
    """
    TrackNet inference engine for ball tracking
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.frame_buffer = []
        self.ball_trajectory = []
        self.is_available = False
        
        # Try to load model
        try:
            self.model = load_tracknet_model(model_path, device)
            if self.model is not None:
                self.is_available = True
                logger.info("TrackNet inference engine initialized successfully")
            else:
                logger.warning("TrackNet model not available, ball tracking disabled")
        except Exception as e:
            logger.error(f"Failed to initialize TrackNet: {e}")
            self.is_available = False
    
    def reset(self):
        """Reset frame buffer and trajectory"""
        self.frame_buffer = []
        self.ball_trajectory = []
    
    def add_frame(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Add frame to buffer and perform ball tracking if possible
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Ball position dict or None
        """
        if not self.is_available:
            return None
        
        self.frame_buffer.append(frame)
        
        # Keep only last 3 frames
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
        
        # Need exactly 3 frames for tracking
        if len(self.frame_buffer) == 3:
            return self._track_ball()
        
        return None
    
    def _track_ball(self) -> Optional[Dict[str, float]]:
        """Internal method to track ball using current frame buffer"""
        try:
            if not self.is_available or len(self.frame_buffer) != 3:
                return None
            
            # Preprocess frames
            input_tensor = preprocess_frames_for_tracknet(self.frame_buffer)
            
            if self.device == 'cuda' and torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Forward pass
            with torch.no_grad():
                heatmap = self.model(input_tensor)
            
            # Postprocess output
            ball_position = postprocess_tracknet_output(heatmap)
            
            return ball_position
            
        except Exception as e:
            logger.error(f"Error in TrackNet ball tracking: {e}")
            return None
    
    def process_video_frames(self, frames: List[np.ndarray]) -> List[Optional[Dict[str, float]]]:
        """
        Process entire video for ball tracking
        
        Args:
            frames: List of video frames
            
        Returns:
            List of ball positions (one per frame)
        """
        if not self.is_available:
            return [None] * len(frames)
        
        self.reset()
        ball_positions = []
        
        # Process each frame
        for frame in frames:
            ball_pos = self.add_frame(frame)
            ball_positions.append(ball_pos)
        
        # Apply smoothing to trajectory
        smoothed_positions = smooth_ball_trajectory(ball_positions)
        
        return smoothed_positions
    
    def enhance_yolo_detections(self, frames: List[np.ndarray], 
                              yolo_objects_per_frame: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """
        Enhance YOLO object detections with TrackNet ball tracking
        
        Args:
            frames: Video frames
            yolo_objects_per_frame: YOLO detections for each frame
            
        Returns:
            Enhanced object detections with refined ball tracking
        """
        if not self.is_available:
            logger.info("TrackNet not available, returning original YOLO detections")
            return yolo_objects_per_frame
        
        # Get TrackNet ball positions
        ball_positions = self.process_video_frames(frames)
        
        # Merge with YOLO detections
        enhanced_detections = []
        
        for i, (yolo_objects, ball_pos) in enumerate(zip(yolo_objects_per_frame, ball_positions)):
            enhanced_objects = merge_yolo_tracknet_detections(yolo_objects, ball_pos)
            enhanced_detections.append(enhanced_objects)
        
        logger.info(f"Enhanced {len(enhanced_detections)} frames with TrackNet ball tracking")
        return enhanced_detections


# Global TrackNet instance (optional)
_tracknet_instance = None


def get_tracknet_instance(model_path: Optional[str] = None, device: str = 'cpu') -> TrackNetInference:
    """Get or create global TrackNet instance"""
    global _tracknet_instance
    
    if _tracknet_instance is None:
        _tracknet_instance = TrackNetInference(model_path, device)
    
    return _tracknet_instance


def is_tracknet_available() -> bool:
    """Check if TrackNet is available for use"""
    try:
        instance = get_tracknet_instance()
        return instance.is_available
    except Exception:
        return False