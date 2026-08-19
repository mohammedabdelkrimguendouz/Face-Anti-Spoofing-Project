"""
Frontend Configuration
----------------------
Configuration for Streamlit frontend.
"""

import os
import streamlit as st
from dataclasses import dataclass
from typing import Optional


@dataclass
class FrontendConfig:
    """Frontend configuration."""
    
    # =========================================================
    # API Configuration - مهم للنشر على Streamlit Cloud
    # =========================================================
    @property
    def API_URL(self) -> str:
        """
        Get API URL - يمكن تعديله حسب البيئة.
        """
        # للاستخدام المحلي
        if os.getenv("STREAMLIT_CLOUD", "false").lower() == "true":
            # في Streamlit Cloud، استخدم عنوان الخادم البعيد
            return os.getenv("API_URL", "https://your-backend-url.com")
        else:
            # محلياً
            return os.getenv("API_URL", "http://localhost:8000")
    
    # =========================================================
    # Model Configuration
    # =========================================================
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "bilstm_vae")
    LSTM_AE_THRESHOLD: float = float(os.getenv("LSTM_AE_THRESHOLD", "0.5"))
    BILSTM_VAE_THRESHOLD: float = float(os.getenv("BILSTM_VAE_THRESHOLD", "0.5"))
    
    # =========================================================
    # Camera Configuration
    # =========================================================
    CAMERA_ID: int = int(os.getenv("CAMERA_ID", "0"))
    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "640"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "480"))
    FPS: int = int(os.getenv("FPS", "30"))
    NUM_FRAMES: int = int(os.getenv("NUM_FRAMES", "150"))
    
    # =========================================================
    # Feature Extraction
    # =========================================================
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "288"))
    FEATURE_SIZE: int = int(os.getenv("FEATURE_SIZE", "1408"))
    
    # =========================================================
    # UI Configuration
    # =========================================================
    THEME: str = os.getenv("THEME", "dark")
    REFRESH_INTERVAL: float = float(os.getenv("REFRESH_INTERVAL", "0.1"))
    SHOW_HEATMAP: bool = os.getenv("SHOW_HEATMAP", "true").lower() == "true"
    SHOW_STATS: bool = os.getenv("SHOW_STATS", "true").lower() == "true"
    
    # =========================================================
    # Results
    # =========================================================
    SAVE_RESULTS: bool = os.getenv("SAVE_RESULTS", "true").lower() == "true"
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", "results")


# Create singleton instance
config = FrontendConfig()