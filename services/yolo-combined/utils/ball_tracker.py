import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from scipy import interpolate
import logging

logger = logging.getLogger(__name__)

class KalmanBallTracker:
    """Enhanced ball tracker with Kalman filtering and physics priors"""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Kalman filter for ball tracking
        # State: [x, y, vx, vy] - position and velocity
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (constant velocity model with gravity)
        self.kalman.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        q = 0.1
        self.kalman.processNoiseCov = q * np.eye(4, dtype=np.float32)
        
        # Measurement noise covariance
        r = 1.0
        self.kalman.measurementNoiseCov = r * np.eye(2, dtype=np.float32)
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 10
        self.trajectory = []
        
    def initialize(self, x: float, y: float):
        """Initialize tracker with first detection"""
        self.kalman.statePre = np.array([x, y, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([x, y, 0, 0], dtype=np.float32)
        self.initialized = True
        self.lost_frames = 0
        self.trajectory = [(x, y)]
        logger.debug(f"Kalman tracker initialized at ({x:.1f}, {y:.1f})")
        
    def predict(self) -> Tuple[float, float]:
        """Predict next position"""
        if not self.initialized:
            return None, None
            
        prediction = self.kalman.predict()
        return float(prediction[0]), float(prediction[1])
        
    def update(self, detection: Optional[Tuple[float, float]]):
        """Update tracker with detection (None if no detection)"""
        if not self.initialized:
            if detection:
                self.initialize(detection[0], detection[1])
            return
            
        if detection:
            # We have a detection - update Kalman filter
            measurement = np.array([[detection[0]], [detection[1]]], dtype=np.float32)
            self.kalman.correct(measurement)
            self.lost_frames = 0
            self.trajectory.append(detection)
        else:
            # No detection - just predict
            self.lost_frames += 1
            pred_x, pred_y = self.predict()
            if pred_x is not None:
                self.trajectory.append((pred_x, pred_y))
            
    def is_lost(self) -> bool:
        """Check if tracker is lost"""
        return self.lost_frames > self.max_lost_frames
        
    def get_state(self) -> Dict[str, float]:
        """Get current state"""
        if not self.initialized:
            return {}
            
        state = self.kalman.statePost
        return {
            "x": float(state[0]),
            "y": float(state[1]),
            "vx": float(state[2]),
            "vy": float(state[3]),
            "confidence": max(0.0, 1.0 - (self.lost_frames / self.max_lost_frames))
        }
        
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get trajectory points"""
        return self.trajectory[-50:]  # Last 50 points for visualization


def smooth_ball_trajectory(ball_detections: List[List[Dict[str, Any]]], fps: float = 30.0) -> List[List[Dict[str, Any]]]:
    """
    Enhanced ball tracking with Kalman filtering and trajectory smoothing
    
    Args:
        ball_detections: List of frame detections, each frame containing list of ball objects
        fps: Frames per second for proper physics modeling
        
    Returns:
        Enhanced detections with smoothed trajectories and filled gaps
    """
    if not ball_detections:
        return ball_detections
        
    tracker = KalmanBallTracker(fps=fps)
    enhanced_detections = []
    
    # Track balls through frames
    for frame_idx, frame_objects in enumerate(ball_detections):
        # Find sports balls in current frame
        balls = [obj for obj in frame_objects if obj.get("class") == "sports ball"]
        
        if balls:
            # Use highest confidence ball for tracking
            best_ball = max(balls, key=lambda x: x.get("confidence", 0))
            bbox = best_ball["bbox"]
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2
            
            tracker.update((center_x, center_y))
        else:
            # No ball detected, predict position
            tracker.update(None)
        
        # Get current state
        state = tracker.get_state()
        
        # Create enhanced frame data
        enhanced_frame = []
        
        # Add non-ball objects unchanged
        for obj in frame_objects:
            if obj.get("class") != "sports ball":
                enhanced_frame.append(obj)
        
        # Add enhanced ball detection
        if state and not tracker.is_lost():
            # Calculate bbox from predicted center
            bbox_size = 20  # Default ball size in pixels
            enhanced_ball = {
                "class": "sports ball",
                "confidence": float(state["confidence"]),
                "bbox": {
                    "x1": float(state["x"] - bbox_size/2),
                    "y1": float(state["y"] - bbox_size/2),
                    "x2": float(state["x"] + bbox_size/2),
                    "y2": float(state["y"] + bbox_size/2)
                },
                "tracking": {
                    "velocity_x": float(state.get("vx", 0)),
                    "velocity_y": float(state.get("vy", 0)),
                    "trajectory": tracker.get_trajectory(),
                    "tracked": True,
                    "interpolated": len(balls) == 0  # Mark if this was interpolated
                }
            }
            enhanced_frame.append(enhanced_ball)
        
        enhanced_detections.append(enhanced_frame)
    
    # Apply polynomial smoothing for final trajectory refinement
    enhanced_detections = apply_polynomial_smoothing(enhanced_detections)
    
    return enhanced_detections


def apply_polynomial_smoothing(detections: List[List[Dict[str, Any]]], window_size: int = 5) -> List[List[Dict[str, Any]]]:
    """
    Apply polynomial interpolation for smooth ball trajectories
    
    Args:
        detections: Frame detections with ball tracking data
        window_size: Number of frames to use for smoothing window
        
    Returns:
        Detections with smoothed ball trajectories
    """
    if len(detections) < window_size:
        return detections
    
    # Extract ball positions
    ball_positions = []
    ball_indices = []
    
    for frame_idx, frame in enumerate(detections):
        for obj in frame:
            if obj.get("class") == "sports ball" and obj.get("tracking", {}).get("tracked"):
                bbox = obj["bbox"]
                center_x = (bbox["x1"] + bbox["x2"]) / 2
                center_y = (bbox["y1"] + bbox["y2"]) / 2
                ball_positions.append((center_x, center_y))
                ball_indices.append(frame_idx)
                break
    
    if len(ball_positions) < 3:
        return detections  # Need at least 3 points for smoothing
    
    # Create smoothed trajectory using polynomial interpolation
    try:
        positions = np.array(ball_positions)
        indices = np.array(ball_indices)
        
        # Smooth X and Y coordinates separately
        if len(positions) >= 3:
            # Use polynomial degree based on number of points
            degree = min(3, len(positions) - 1)
            
            # Interpolate X coordinates
            poly_x = np.polyfit(indices, positions[:, 0], degree)
            smooth_x = np.polyval(poly_x, indices)
            
            # Interpolate Y coordinates  
            poly_y = np.polyfit(indices, positions[:, 1], degree)
            smooth_y = np.polyval(poly_y, indices)
            
            # Update detections with smoothed positions
            for i, frame_idx in enumerate(ball_indices):
                for obj in detections[frame_idx]:
                    if obj.get("class") == "sports ball" and obj.get("tracking", {}).get("tracked"):
                        bbox_size = obj["bbox"]["x2"] - obj["bbox"]["x1"]
                        obj["bbox"] = {
                            "x1": float(smooth_x[i] - bbox_size/2),
                            "y1": float(smooth_y[i] - bbox_size/2),
                            "x2": float(smooth_x[i] + bbox_size/2),
                            "y2": float(smooth_y[i] + bbox_size/2)
                        }
                        obj["tracking"]["smoothed"] = True
                        break
    
    except Exception as e:
        logger.warning(f"Polynomial smoothing failed: {e}")
    
    return detections


def draw_enhanced_ball_trajectory(frame: np.ndarray, ball_objects: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw enhanced ball visualization with trajectory
    
    Args:
        frame: Video frame
        ball_objects: Ball objects with tracking information
        
    Returns:
        Annotated frame with trajectory visualization
    """
    annotated_frame = frame.copy()
    
    for obj in ball_objects:
        if obj.get("class") != "sports ball":
            continue
            
        bbox = obj["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        confidence = obj.get("confidence", 0)
        tracking = obj.get("tracking", {})
        
        # Choose color based on tracking status
        if tracking.get("interpolated"):
            color = (0, 255, 255)  # Yellow for interpolated
            thickness = 1
        elif tracking.get("tracked"):
            color = (0, 255, 0)  # Green for tracked
            thickness = 2
        else:
            color = (0, 0, 255)  # Red for raw detection
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw confidence and tracking info
        label = f"Ball {confidence:.2f}"
        if tracking.get("tracked"):
            vx = tracking.get("velocity_x", 0)
            vy = tracking.get("velocity_y", 0)
            speed = np.sqrt(vx**2 + vy**2)
            label += f" v:{speed:.1f}"
        
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        # Draw trajectory
        trajectory = tracking.get("trajectory", [])
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                alpha = i / len(trajectory)  # Fade older points
                traj_color = tuple(int(c * alpha) for c in color)
                cv2.line(annotated_frame, pt1, pt2, traj_color, 1)
    
    return annotated_frame