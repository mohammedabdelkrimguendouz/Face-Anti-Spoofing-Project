"""
Frame Buffer
------------
Buffers frames for feature extraction and sequence processing.
"""

import numpy as np
from typing import List, Optional
from collections import deque
import cv2

# =============================================================
# Import config with fallback
# =============================================================
try:
    from frontend.config import config
except ImportError:
    # Fallback config
    class Config:
        FEATURE_SIZE = 1408
        IMAGE_SIZE = 288
        NUM_FRAMES = 150
    config = Config()


class FrameBuffer:
    """
    Buffer for storing frames and extracting features.
    """
    
    def __init__(self, max_size: int = None):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum number of frames to store
        """
        if max_size is None:
            max_size = getattr(config, 'NUM_FRAMES', 150)
        
        self.max_size = max_size
        self.frames = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        
    def add(self, frame: np.ndarray) -> None:
        """
        Add a frame to the buffer.
        
        Args:
            frame: numpy array - Input frame
        """
        if frame is not None:
            self.frames.append(frame.copy())
        
    def clear(self) -> None:
        """Clear the buffer."""
        self.frames.clear()
        self.timestamps.clear()
    
    def is_full(self) -> bool:
        """
        Check if buffer is full.
        
        Returns:
            bool: True if buffer has max_size frames
        """
        return len(self.frames) == self.max_size
    
    def size(self) -> int:
        """
        Get current buffer size.
        
        Returns:
            int: Number of frames in buffer
        """
        return len(self.frames)
    
    def get_frames(self) -> Optional[List[np.ndarray]]:
        """
        Get all frames from buffer.
        
        Returns:
            Optional[List[np.ndarray]]: List of frames or None
        """
        if len(self.frames) == 0:
            return None
        return list(self.frames)
    
    def get_features(self) -> Optional[List[List[float]]]:
        """
        Extract features from buffered frames.
        
        This is a placeholder - in production, you would:
        1. Preprocess each frame
        2. Extract features using EfficientNet-B2
        3. Return the feature sequence
        
        Returns:
            Optional[List[List[float]]]: Feature sequence [T, 1408] or None
        """
        if len(self.frames) < self.max_size:
            return None
        
        feature_size = getattr(config, 'FEATURE_SIZE', 1408)
        features = []
        
        for frame in self.frames:
            try:
                # Preprocess frame
                processed = self._preprocess_frame(frame)
                # Placeholder feature extraction
                feature = np.random.randn(feature_size).astype(np.float32)
                features.append(feature.tolist())
            except Exception as e:
                # Skip problematic frames
                continue
        
        if len(features) == 0:
            return None
            
        return features
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.
        
        Args:
            frame: numpy array - Input frame
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        image_size = getattr(config, 'IMAGE_SIZE', 288)
        
        # Resize
        resized = cv2.resize(
            frame,
            (image_size, image_size)
        )
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to RGB if needed
        if len(normalized.shape) == 2:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        elif normalized.shape[2] == 4:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_BGRA2RGB)
        elif normalized.shape[2] == 3:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        
        return normalized
    
    def get_frame_sequence(self) -> Optional[np.ndarray]:
        """
        Get frames as a sequence tensor.
        
        Returns:
            Optional[np.ndarray]: Frame sequence [T, H, W, C] or None
        """
        if len(self.frames) < self.max_size:
            return None
        
        sequence = np.array(self.frames)
        return sequence