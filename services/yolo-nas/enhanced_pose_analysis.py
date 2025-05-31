"""
Enhanced YOLO-NAS Pose Analysis Module
Implements all HIGH PRIORITY and ADVANCED features for padel pose analysis
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class JointAnalysis:
    """Enhanced joint analysis with confidence and stability tracking"""
    joint_id: int
    joint_name: str
    x: float
    y: float
    confidence: float
    stability_score: float
    quality_score: float
    is_high_quality: bool
    padel_relevance: float

@dataclass
class PoseQualityMetrics:
    """Comprehensive pose quality assessment"""
    overall_score: float
    visible_joints: int
    total_joints: int
    joint_distribution_score: float
    stability_score: float
    confidence_scores: List[float]
    padel_relevance_score: float
    quality_category: str  # "excellent", "good", "fair", "poor"

@dataclass
class BatchPredictions:
    """🎯 HIGH PRIORITY: Structured batch format output"""
    num_detections: int
    pred_boxes: List[List[float]]
    pred_scores: List[float]
    pred_joints: List[List[Tuple[float, float, float]]]
    pred_labels: List[int]
    batch_metadata: Dict[str, Any]

# 🎨 ADVANCED: 17-Keypoint Joint Analysis - COCO keypoints mapping
COCO_KEYPOINTS = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

# 🎨 ADVANCED: Padel-Specific Joint Mapping
PADEL_JOINTS = {
    'serving_arm': [5, 7, 9],      # Right shoulder, elbow, wrist
    'balance_leg': [11, 13, 15],   # Hip, knee, ankle  
    'racket_grip': [9, 10],        # Both wrists for grip analysis
    'core_stability': [5, 6, 11, 12],  # Shoulders and hips
    'power_chain': [6, 8, 10],     # Right shoulder->elbow->wrist
    'balance_chain': [5, 7, 9],    # Left shoulder->elbow->wrist
    'lower_body': [11, 12, 13, 14, 15, 16],  # Hip to ankle chains
    'head_tracking': [0, 1, 2, 3, 4]  # Head orientation
}

class JointStabilityTracker:
    """🎯 HIGH PRIORITY: Enhanced Joint Confidence Analysis - Track joint stability over time"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.joint_history = defaultdict(lambda: deque(maxlen=window_size))
        self.confidence_history = defaultdict(lambda: deque(maxlen=window_size))
        self.quality_history = defaultdict(lambda: deque(maxlen=window_size))
        
    def track_joint_stability_over_time(self, joint_index: int, confidence: float):
        """Track joint stability over time as requested"""
        self.confidence_history[joint_index].append(confidence)
        
        # Calculate moving average stability
        if len(self.confidence_history[joint_index]) >= 3:
            recent_confidences = list(self.confidence_history[joint_index])
            variance = np.var(recent_confidences)
            stability = max(0.0, 1.0 - variance)
            self.quality_history[joint_index].append(stability)
        
    def analyze_joint_quality(self, joint_index: int, x: float, y: float, confidence: float) -> JointAnalysis:
        """🎯 HIGH PRIORITY: Per-joint confidence scoring and filtering"""
        joint_name = COCO_KEYPOINTS.get(joint_index, f"joint_{joint_index}")
        
        # Update tracking
        self.track_joint_stability_over_time(joint_index, confidence)
        self.joint_history[joint_index].append((x, y))
        
        # Calculate stability score
        stability_score = self.get_stability_score(joint_index)
        
        # Calculate padel relevance for this joint
        padel_relevance = self._calculate_joint_padel_relevance(joint_index)
        
        # Calculate overall quality score
        quality_score = (
            confidence * 0.5 +           # Base confidence
            stability_score * 0.3 +      # Stability over time
            padel_relevance * 0.2        # Padel-specific importance
        )
        
        # Determine if high quality based on threshold
        joint_confidence_threshold = 0.5  # Configurable threshold
        is_high_quality = quality_score > joint_confidence_threshold
        
        return JointAnalysis(
            joint_id=joint_index,
            joint_name=joint_name,
            x=x,
            y=y,
            confidence=confidence,
            stability_score=stability_score,
            quality_score=quality_score,
            is_high_quality=is_high_quality,
            padel_relevance=padel_relevance
        )
        
    def get_stability_score(self, joint_index: int) -> float:
        """Calculate joint stability score from history"""
        if joint_index not in self.confidence_history or len(self.confidence_history[joint_index]) < 3:
            return 0.0
            
        # Position stability
        if len(self.joint_history[joint_index]) >= 3:
            positions = list(self.joint_history[joint_index])[-3:]  # Last 3 positions
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_variance = np.var(x_coords)
            y_variance = np.var(y_coords)
            position_stability = max(0.0, 1.0 - (x_variance + y_variance) / 1000.0)
        else:
            position_stability = 0.0
            
        # Confidence stability
        confidences = list(self.confidence_history[joint_index])[-3:]  # Last 3 confidences
        confidence_variance = np.var(confidences)
        confidence_stability = max(0.0, 1.0 - confidence_variance)
        
        # Combined stability
        return (position_stability * 0.6 + confidence_stability * 0.4)
    
    def _calculate_joint_padel_relevance(self, joint_index: int) -> float:
        """Calculate how important this joint is for padel analysis"""
        # High importance joints for padel
        high_importance = [5, 6, 7, 8, 9, 10]  # Arms and shoulders
        medium_importance = [11, 12, 13, 14]   # Core and upper legs
        low_importance = [15, 16]              # Ankles
        minimal_importance = [0, 1, 2, 3, 4]   # Head region
        
        if joint_index in high_importance:
            return 1.0
        elif joint_index in medium_importance:
            return 0.8
        elif joint_index in low_importance:
            return 0.6
        elif joint_index in minimal_importance:
            return 0.3
        else:
            return 0.5

class EnhancedPoseAnalyzer:
    """Main enhanced pose analyzer with all requested features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.joint_tracker = JointStabilityTracker()
        
        # 🚀 MEDIUM PRIORITY: Custom NMS Parameters for Padel
        self.padel_nms_config = {
            "confidence_threshold": 0.3,      # Lower for player detection
            "nms_threshold": 0.5,             # Reduce overlapping detections  
            "max_predictions_per_image": 10   # Max realistic player count
        }
        
        logger.info("Enhanced Pose Analyzer initialized with padel-optimized parameters")
    
    def calculate_pose_quality(self, pred_joints: List[Tuple[float, float, float]]) -> PoseQualityMetrics:
        """🎨 ADVANCED: Dynamic Pose Quality Scoring"""
        
        # Count visible joints (confidence > 0.5)
        visible_joints = sum(1 for _, _, conf in pred_joints if conf > 0.5)
        total_joints = len(COCO_KEYPOINTS)
        
        # Calculate joint distribution score
        visible_positions = [(x, y) for x, y, conf in pred_joints if conf > 0.5]
        joint_distribution_score = self._analyze_joint_spacing(visible_positions)
        
        # Calculate average confidence and stability
        confidence_scores = [conf for _, _, conf in pred_joints if conf > 0.5]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Calculate stability across joints
        stability_scores = []
        for i, (_, _, conf) in enumerate(pred_joints):
            if conf > 0.5:
                stability = self.joint_tracker.get_stability_score(i)
                stability_scores.append(stability)
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        # Calculate padel relevance
        padel_relevance = self._calculate_pose_padel_relevance(pred_joints)
        
        # Overall pose quality score
        coverage_score = visible_joints / total_joints
        overall_score = (
            coverage_score * 0.25 +      # Joint coverage
            avg_confidence * 0.35 +      # Average confidence
            joint_distribution_score * 0.15 +  # Joint distribution
            avg_stability * 0.1 +        # Stability
            padel_relevance * 0.15       # Padel relevance
        )
        
        # Categorize quality
        if overall_score >= 0.8:
            quality_category = "excellent"
        elif overall_score >= 0.6:
            quality_category = "good"
        elif overall_score >= 0.4:
            quality_category = "fair"
        else:
            quality_category = "poor"
        
        return PoseQualityMetrics(
            overall_score=overall_score,
            visible_joints=visible_joints,
            total_joints=total_joints,
            joint_distribution_score=joint_distribution_score,
            stability_score=avg_stability,
            confidence_scores=confidence_scores,
            padel_relevance_score=padel_relevance,
            quality_category=quality_category
        )
    
    def _analyze_joint_spacing(self, positions: List[Tuple[float, float]]) -> float:
        """Analyze joint distribution quality"""
        if len(positions) < 3:
            return 0.0
        
        positions = np.array(positions)
        x_var = np.var(positions[:, 0])
        y_var = np.var(positions[:, 1])
        
        # Normalize variance to 0-1 score (higher variance = better distribution)
        total_var = x_var + y_var
        normalized_score = min(1.0, total_var / 10000.0)
        
        return normalized_score
    
    def _calculate_pose_padel_relevance(self, pred_joints: List[Tuple[float, float, float]]) -> float:
        """Calculate how relevant this pose is for padel analysis"""
        padel_scores = []
        
        for joint_group_name, joint_indices in PADEL_JOINTS.items():
            group_visibility = 0
            total_confidence = 0
            
            for joint_idx in joint_indices:
                if joint_idx < len(pred_joints):
                    _, _, conf = pred_joints[joint_idx]
                    if conf > 0.5:
                        group_visibility += 1
                        total_confidence += conf
            
            if len(joint_indices) > 0:
                group_score = (group_visibility / len(joint_indices)) * 0.6
                if group_visibility > 0:
                    avg_confidence = total_confidence / group_visibility
                    group_score += avg_confidence * 0.4
                
                # Weight by importance for padel
                if joint_group_name in ['serving_arm', 'racket_grip']:
                    padel_scores.append(group_score * 1.0)
                elif joint_group_name in ['power_chain', 'core_stability']:
                    padel_scores.append(group_score * 0.9)
                elif joint_group_name in ['balance_leg', 'lower_body']:
                    padel_scores.append(group_score * 0.8)
                else:
                    padel_scores.append(group_score * 0.6)
        
        return np.mean(padel_scores) if padel_scores else 0.0