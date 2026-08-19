"""
Camera View Component
---------------------
Component for displaying camera feed with overlays.
"""

import streamlit as st
import cv2
import numpy as np
from typing import Optional

from frontend.camera.capture import CameraCapture


class CameraView:
    """
    Camera view UI component.
    """
    
    def __init__(self):
        """Initialize camera view."""
        self.placeholder = None
    
    def render(
        self,
        camera: CameraCapture,
        is_running: bool
    ) -> Optional[np.ndarray]:
        """
        Render camera view.
        
        Args:
            camera: Camera capture instance
            is_running: Whether camera is running
            
        Returns:
            Optional[np.ndarray]: Current frame or None
        """
        # Create placeholder for video
        if self.placeholder is None:
            self.placeholder = st.empty()
        
        frame = None
        
        if is_running and camera.is_opened():
            # Read frame
            frame = camera.read()
            
            if frame is not None:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                self.placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )
                
                return frame
            else:
                self.placeholder.warning("⏳ جاري تحميل الكاميرا...")
        else:
            # Show placeholder when camera is off
            if not is_running:
                self.placeholder.info(
                    "📷 الكاميرا متوقفة\n\n"
                    "اضغط 'بدء' في الشريط الجانبي لتشغيل الكاميرا"
                )
            else:
                self.placeholder.warning("❌ الكاميرا غير متاحة")
        
        return None