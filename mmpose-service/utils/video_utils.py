import cv2
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def get_video_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file in get_video_info: {video_path}")
        return {
            "fps": 0, "frame_count": 0, "width": 0,
            "height": 0, "duration": 0, "error": "Could not open video"
        }
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }

def extract_frames(video_path: str, num_frames_to_extract: int = -1, start_frame: int = 0) -> List[np.ndarray]:
    """
    Extracts a specified number of consecutive frames from a video.
    Args:
        video_path: Path to the video file.
        num_frames_to_extract: How many consecutive frames to extract. 
                               -1 means all frames from start_frame.
        start_frame: The frame index to start extraction from.
    Returns:
        A list of OpenCV frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file in extract_frames: {video_path}")
        return []

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: List[np.ndarray] = []
    frames_read_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"End of video or cannot read frame at index approx {start_frame + frames_read_count}.")
            break
        
        frames.append(frame)
        frames_read_count += 1
        
        if num_frames_to_extract != -1 and frames_read_count >= num_frames_to_extract:
            logger.debug(f"Reached num_frames_to_extract: {num_frames_to_extract}.")
            break
            
    cap.release()
    logger.info(f"Extracted {len(frames)} frames starting from frame {start_frame}.")
    return frames