"""
Face Anti-Spoofing - Main Application
-------------------------------------
Streamlit application for real-time face anti-spoofing detection.
"""

import streamlit as st
import sys
import os

# =============================================================
# إضافة المسارات
# =============================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================
# استيراد OpenCV مع معالجة الأخطاء
# =============================================================
import cv2

import numpy as np
from PIL import Image
import time
from datetime import datetime

from frontend.config import config
from frontend.api.client import APIClient
from frontend.camera.capture import CameraCapture
from frontend.camera.frame_buffer import FrameBuffer
from frontend.components.model_selector import ModelSelector
from frontend.components.camera_view import CameraView
from frontend.components.result_view import ResultView
from frontend.components.heatmap_view import HeatmapView

# =============================================================
# Page Configuration
# =============================================================
st.set_page_config(
    page_title="Face Anti-Spoofing System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# Custom CSS
# =============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .real-box {
        background-color: rgba(46, 204, 113, 0.2);
        border: 2px solid #2ECC71;
    }
    .spoof-box {
        background-color: rgba(231, 76, 60, 0.2);
        border: 2px solid #E74C3C;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================
# Session State Initialization
# =============================================================
def init_session_state():
    """Initialize session state variables."""
    if "api_client" not in st.session_state:
        try:
            st.session_state.api_client = APIClient(config.API_URL)
        except Exception as e:
            st.error(f"⚠️ فشل تهيئة عميل API: {str(e)}")
            st.session_state.api_client = None
    
    if "camera" not in st.session_state:
        st.session_state.camera = CameraCapture(camera_id=config.CAMERA_ID)
    
    if "frame_buffer" not in st.session_state:
        st.session_state.frame_buffer = FrameBuffer(max_size=config.NUM_FRAMES)
    
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = config.DEFAULT_MODEL
    
    if "threshold" not in st.session_state:
        st.session_state.threshold = config.BILSTM_VAE_THRESHOLD
    
    if "show_heatmap" not in st.session_state:
        st.session_state.show_heatmap = config.SHOW_HEATMAP
    
    if "fps" not in st.session_state:
        st.session_state.fps = 0
    
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = True  # Demo mode enabled by default


# =============================================================
# Sidebar
# =============================================================
def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.markdown("## 🎯 التحكم")
        
        # =========================================================
        # وضع العرض
        # =========================================================
        st.markdown("### 📱 وضع التشغيل")
        demo_mode = st.toggle(
            "🔄 وضع العرض (Demo)",
            value=st.session_state.demo_mode,
            help="استخدم بيانات وهمية بدلاً من الكاميرا"
        )
        st.session_state.demo_mode = demo_mode
        
        # =========================================================
        # Model Selection
        # =========================================================
        st.markdown("### 🤖 النموذج")
        model_selector = ModelSelector()
        selected_model = model_selector.render()
        st.session_state.selected_model = selected_model
        
        # =========================================================
        # Threshold
        # =========================================================
        st.markdown("### ⚙️ العتبة")
        threshold = st.slider(
            "عتبة الكشف",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.01,
            help="كلما انخفضت العتبة، زادت الحساسية للكشف عن التزييف"
        )
        st.session_state.threshold = threshold
        
        # =========================================================
        # Heatmap toggle
        # =========================================================
        st.markdown("### 📊 العرض")
        show_heatmap = st.checkbox(
            "عرض خريطة الحرارة",
            value=st.session_state.show_heatmap
        )
        st.session_state.show_heatmap = show_heatmap
        
        # =========================================================
        # Camera controls
        # =========================================================
        st.markdown("### 📷 الكاميرا")
        
        if not st.session_state.demo_mode:
            col1, col2 = st.columns(2)
            with col1:
                start_button = st.button(
                    "▶️ بدء",
                    use_container_width=True,
                    type="primary"
                )
            with col2:
                stop_button = st.button(
                    "⏹️ إيقاف",
                    use_container_width=True,
                    type="secondary"
                )
            
            if start_button:
                st.session_state.is_running = True
                st.session_state.camera.start()
            
            if stop_button:
                st.session_state.is_running = False
                st.session_state.camera.stop()
                st.session_state.frame_buffer.clear()
        else:
            st.info("🔄 وضع العرض: يتم استخدام بيانات وهمية")
            st.session_state.is_running = True
        
        # =========================================================
        # Status
        # =========================================================
        st.markdown("### 📊 الحالة")
        if st.session_state.demo_mode:
            status_text = "🟢 وضع العرض"
        else:
            status_text = "🟢 يعمل" if st.session_state.is_running else "🔴 متوقف"
        st.info(status_text)
        
        if st.session_state.fps > 0:
            st.metric("FPS", f"{st.session_state.fps:.1f}")
        
        # =========================================================
        # Statistics
        # =========================================================
        if len(st.session_state.history) > 0:
            st.markdown("### 📈 الإحصائيات")
            real_count = sum(1 for r in st.session_state.history if r.get("prediction") == 0)
            spoof_count = len(st.session_state.history) - real_count
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🟢 حقيقي", real_count)
            with col2:
                st.metric("🔴 مزيف", spoof_count)
            
            if len(st.session_state.history) > 0:
                avg_confidence = sum(r.get("confidence", 0) for r in st.session_state.history) / len(st.session_state.history)
                st.metric("متوسط الثقة", f"{avg_confidence:.2%}")
        
        # =========================================================
        # Clear history
        # =========================================================
        if st.button("🧹 مسح التاريخ", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        # =========================================================
        # Check API connection
        # =========================================================
        st.markdown("### 🔌 الاتصال بالخادم")
        if st.session_state.api_client:
            try:
                health = st.session_state.api_client.health_check()
                if health.get("status") == "healthy":
                    st.success("✅ الخادم متصل")
                else:
                    st.warning("⚠️ الخادم غير صحي")
            except:
                st.error("❌ الخادم غير متصل")
        else:
            st.error("❌ عميل API غير مهيأ")


# =============================================================
# Main Content
# =============================================================
def render_main():
    """Render main content area."""
    st.markdown('<h1 class="main-header">🎭 نظام كشف تزييف الوجه</h1>', unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera view
        if st.session_state.demo_mode:
            # Demo mode: show placeholder
            placeholder = st.empty()
            
            # Generate demo frames
            if st.session_state.is_running:
                try:
                    frame = generate_demo_frame()
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        process_frame(frame)
                except Exception as e:
                    placeholder.error(f"⚠️ خطأ في وضع العرض: {str(e)}")
            else:
                placeholder.info("⏳ وضع العرض متوقف")
        else:
            # Camera mode
            try:
                camera_view = CameraView()
                frame = camera_view.render(
                    camera=st.session_state.camera,
                    is_running=st.session_state.is_running
                )
                if frame is not None:
                    process_frame(frame)
            except Exception as e:
                st.error(f"⚠️ خطأ في الكاميرا: {str(e)}")
    
    with col2:
        # Results
        result_view = ResultView()
        result_view.render(st.session_state.current_result)
        
        # Heatmap
        if st.session_state.show_heatmap:
            heatmap_view = HeatmapView()
            heatmap_view.render(st.session_state.current_result)


def generate_demo_frame() -> np.ndarray:
    """
    Generate a demo frame with random content.
    
    Returns:
        np.ndarray: Demo frame
    """
    height = 480
    width = 640
    
    # Create random frame
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some structure
    cv2.rectangle(frame, (100, 100), (540, 380), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "DEMO MODE",
        (200, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3
    )
    
    return frame


def process_frame(frame):
    """
    Process a single frame for anti-spoofing detection.
    
    Args:
        frame: numpy array - Input frame
    """
    try:
        # Add frame to buffer
        st.session_state.frame_buffer.add(frame)
        
        # Check if buffer is full
        if st.session_state.frame_buffer.is_full():
            # Get feature sequence
            features = st.session_state.frame_buffer.get_features()
            
            if features is not None and st.session_state.api_client:
                # Measure inference time
                start_time = time.time()
                
                try:
                    # Get prediction from API
                    result = st.session_state.api_client.predict(
                        features=features,
                        model_name=st.session_state.selected_model,
                        threshold=st.session_state.threshold
                    )
                    
                    # Calculate FPS
                    inference_time = time.time() - start_time
                    st.session_state.fps = 1.0 / inference_time if inference_time > 0 else 0
                    
                    # Add timestamp
                    result["timestamp"] = datetime.now().isoformat()
                    
                    # Update current result
                    st.session_state.current_result = result
                    
                    # Add to history
                    st.session_state.history.append(result)
                    
                    # Keep history limited
                    if len(st.session_state.history) > 100:
                        st.session_state.history = st.session_state.history[-100:]
                    
                except Exception as e:
                    # Use mock result if API fails
                    if st.session_state.demo_mode:
                        mock_result = {
                            "prediction": np.random.choice([0, 1]),
                            "status": "real" if np.random.random() > 0.5 else "spoof",
                            "anomaly_score": np.random.random(),
                            "confidence": 0.5 + np.random.random() * 0.5,
                            "reconstruction_error": np.random.random() * 0.1,
                            "threshold": st.session_state.threshold,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.current_result = mock_result
                        st.session_state.history.append(mock_result)
                    else:
                        st.error(f"⚠️ خطأ في التنبؤ: {str(e)}")
                
                # Clear buffer for next sequence
                st.session_state.frame_buffer.clear()
                
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")


# =============================================================
# Main App
# =============================================================
def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_main()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            Face Anti-Spoofing System v1.0.0 | 
            Developed with ❤️ using Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()