"""
TrackNet Model Architecture
Lightweight implementation for ball tracking in padel videos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrackNet(nn.Module):
    """
    Simplified TrackNet for ball tracking
    Input: 3 consecutive frames (640x360)
    Output: Gaussian heatmap with ball position
    """
    
    def __init__(self, input_height: int = 360, input_width: int = 640):
        super(TrackNet, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        
        # Encoder (3 frames = 9 channels)
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x1_pool = self.pool(x1)
        
        x2 = self.relu(self.conv2(x1_pool))
        x2_pool = self.pool(x2)
        
        x3 = self.relu(self.conv3(x2_pool))
        x3_pool = self.pool(x3)
        
        x4 = self.relu(self.conv4(x3_pool))
        x4_pool = self.pool(x4)
        
        # Decoder
        d1 = self.relu(self.deconv1(x4_pool))
        d2 = self.relu(self.deconv2(d1))
        d3 = self.relu(self.deconv3(d2))
        d4 = self.sigmoid(self.deconv4(d3))
        
        return d4


def load_tracknet_model(model_path: str, device: str = 'cpu') -> Optional[TrackNet]:
    """Load TrackNet model with error handling"""
    try:
        model = TrackNet()
        if torch.cuda.is_available() and device == 'cuda':
            model = model.cuda()
        
        # Try to load weights if available
        try:
            if model_path and torch.cuda.is_available():
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(f"TrackNet weights loaded from {model_path}")
            else:
                logger.warning("TrackNet running with random weights (no pre-trained model)")
        except Exception as e:
            logger.warning(f"Could not load TrackNet weights: {e}")
        
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"Failed to create TrackNet model: {e}")
        return None