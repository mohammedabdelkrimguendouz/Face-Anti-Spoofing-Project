"""
Camera Capture
--------------
Handles video capture from camera.
"""

import cv2
import threading
import numpy as np
from typing import Optional, Tuple
import logging

from frontend.config import config

# =============================================================
# Setup
# =============================================================
logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Camera capture class with threading support.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        
        logger.info(f"Camera initialized: ID={camera_id}, {width}x{height}@{fps}fps")
    
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            bool: True if started successfully
        """
        if self.is_running:
            logger.warning("Camera already running")
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop(self) -> None:
        """Stop camera capture."""
        self.is_running = False
        
        if self.thread is not None:
            self.thread.join(timeout=2)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        logger.info("Camera stopped")
    
    def _capture_loop(self) -> None:
        """Capture loop running in separate thread."""
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                logger.warning("Failed to read frame")
                break
    
    def read(self) -> Optional[np.ndarray]:
        """
        Read the latest frame.
        
        Returns:
            Optional[np.ndarray]: Latest frame or None
        """
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def is_opened(self) -> bool:
        """
        Check if camera is opened.
        
        Returns:
            bool: True if camera is opened
        """
        return self.cap is not None and self.cap.isOpened()
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get frame size.
        
        Returns:
            Tuple[int, int]: (width, height)
        """
        return (self.width, self.height)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()