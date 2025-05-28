"""
TrackNet Utilities
Preprocessing and postprocessing functions for ball tracking
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def preprocess_frames_for_tracknet(frames: List[np.ndarray], target_size: Tuple[int, int] = (640, 360)) -> torch.Tensor:
    """
    Preprocess 3 consecutive frames for TrackNet input
    
    Args:
        frames: List of 3 consecutive frames (BGR format)
        target_size: Target resolution (width, height)
    
    Returns:
        torch.Tensor: Preprocessed tensor (1, 9, H, W)
    """
    if len(frames) != 3:
        raise ValueError("TrackNet requires exactly 3 consecutive frames")
    
    processed_frames = []
    
    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Convert to CHW format
        chw_frame = np.transpose(normalized, (2, 0, 1))
        processed_frames.append(chw_frame)
    
    # Concatenate 3 frames along channel dimension (3*3 = 9 channels)
    input_tensor = np.concatenate(processed_frames, axis=0)
    
    # Add batch dimension and convert to torch tensor
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
    
    return input_tensor


def postprocess_tracknet_output(heatmap: torch.Tensor, confidence_threshold: float = 0.5) -> Optional[Dict[str, float]]:
    """
    Extract ball position from TrackNet heatmap output
    
    Args:
        heatmap: TrackNet output heatmap (1, 1, H, W)
        confidence_threshold: Minimum confidence for ball detection
    
    Returns:
        Dict with ball position and confidence, or None if no ball detected
    """
    try:
        # Remove batch and channel dimensions
        heatmap_np = heatmap.squeeze().cpu().numpy()
        
        # Find maximum value and position
        max_conf = float(np.max(heatmap_np))
        
        if max_conf < confidence_threshold:
            return None
        
        # Find coordinates of maximum
        max_pos = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
        y, x = max_pos
        
        # Convert to original frame coordinates (assuming 640x360 input)
        ball_x = float(x * (640 / heatmap_np.shape[1]))
        ball_y = float(y * (360 / heatmap_np.shape[0]))
        
        return {
            "x": ball_x,
            "y": ball_y,
            "confidence": max_conf,
            "heatmap_size": heatmap_np.shape
        }
        
    except Exception as e:
        logger.error(f"Error processing TrackNet output: {e}")
        return None


def smooth_ball_trajectory(ball_positions: List[Optional[Dict[str, float]]], 
                         window_size: int = 5) -> List[Optional[Dict[str, float]]]:
    """
    Apply temporal smoothing to ball trajectory
    
    Args:
        ball_positions: List of ball positions from TrackNet
        window_size: Size of smoothing window
    
    Returns:
        Smoothed ball positions
    """
    if not ball_positions:
        return ball_positions
    
    smoothed = []
    
    for i, current_pos in enumerate(ball_positions):
        if current_pos is None:
            smoothed.append(None)
            continue
        
        # Collect valid positions in window
        window_start = max(0, i - window_size // 2)
        window_end = min(len(ball_positions), i + window_size // 2 + 1)
        
        valid_positions = []
        for j in range(window_start, window_end):
            if ball_positions[j] is not None:
                valid_positions.append(ball_positions[j])
        
        if not valid_positions:
            smoothed.append(current_pos)
            continue
        
        # Calculate weighted average (more weight to center)
        weights = []
        x_coords = []
        y_coords = []
        
        for pos in valid_positions:
            weights.append(pos["confidence"])
            x_coords.append(pos["x"])
            y_coords.append(pos["y"])
        
        total_weight = sum(weights)
        if total_weight > 0:
            smoothed_x = sum(x * w for x, w in zip(x_coords, weights)) / total_weight
            smoothed_y = sum(y * w for y, w in zip(y_coords, weights)) / total_weight
            
            smoothed_pos = {
                "x": smoothed_x,
                "y": smoothed_y,
                "confidence": current_pos["confidence"],
                "smoothed": True
            }
            smoothed.append(smoothed_pos)
        else:
            smoothed.append(current_pos)
    
    return smoothed


def merge_yolo_tracknet_detections(yolo_objects: List[Dict[str, Any]], 
                                 tracknet_ball: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Merge YOLO object detections with TrackNet ball tracking
    
    Args:
        yolo_objects: YOLO detected objects
        tracknet_ball: TrackNet ball position
    
    Returns:
        Enhanced object list with refined ball tracking
    """
    enhanced_objects = []
    ball_found_in_yolo = False
    
    # Process existing YOLO detections
    for obj in yolo_objects:
        if obj.get("class") == "sports ball" and tracknet_ball is not None:
            # Replace YOLO ball detection with TrackNet refinement
            enhanced_ball = {
                "class": "sports ball",
                "confidence": float(tracknet_ball["confidence"]),
                "bbox": {
                    "x1": max(0, tracknet_ball["x"] - 15),
                    "y1": max(0, tracknet_ball["y"] - 15),
                    "x2": min(640, tracknet_ball["x"] + 15),
                    "y2": min(360, tracknet_ball["y"] + 15)
                },
                "center": {
                    "x": tracknet_ball["x"],
                    "y": tracknet_ball["y"]
                },
                "tracking_method": "tracknet_refined"
            }
            enhanced_objects.append(enhanced_ball)
            ball_found_in_yolo = True
        else:
            enhanced_objects.append(obj)
    
    # Add TrackNet ball if not found in YOLO
    if not ball_found_in_yolo and tracknet_ball is not None:
        tracknet_only_ball = {
            "class": "sports ball",
            "confidence": float(tracknet_ball["confidence"]),
            "bbox": {
                "x1": max(0, tracknet_ball["x"] - 15),
                "y1": max(0, tracknet_ball["y"] - 15),
                "x2": min(640, tracknet_ball["x"] + 15),
                "y2": min(360, tracknet_ball["y"] + 15)
            },
            "center": {
                "x": tracknet_ball["x"],
                "y": tracknet_ball["y"]
            },
            "tracking_method": "tracknet_only"
        }
        enhanced_objects.append(tracknet_only_ball)
    
    return enhanced_objects