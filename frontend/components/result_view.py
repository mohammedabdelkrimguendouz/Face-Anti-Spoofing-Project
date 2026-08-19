"""
Result View Component
---------------------
Component for displaying prediction results.
"""

import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime


class ResultView:
    """
    Result view UI component.
    """
    
    def __init__(self):
        """Initialize result view."""
        pass
    
    def render(self, result: Optional[Dict[str, Any]]) -> None:
        """
        Render prediction results.
        
        Args:
            result: Prediction result dictionary
        """
        st.markdown("### 📊 النتائج")
        
        if result is None:
            st.info("⏳ انتظار النتائج...")
            return
        
        # Check for error
        if "error" in result:
            st.error(f"❌ خطأ: {result['error']}")
            return
        
        # Extract values
        prediction = result.get("prediction", -1)
        status = result.get("status", "unknown")
        anomaly_score = result.get("anomaly_score", 0.0)
        confidence = result.get("confidence", 0.0)
        reconstruction_error = result.get("reconstruction_error", 0.0)
        threshold = result.get("threshold", 0.5)
        timestamp = result.get("timestamp", datetime.now().isoformat())
        
        # Display prediction
        if prediction == 0:
            st.markdown(
                """
                <div class="result-box real-box">
                    <h2 style="color: #2ECC71; margin: 0;">✅ حقيقي</h2>
                    <p style="margin: 0;">هذا الوجه حقيقي</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif prediction == 1:
            st.markdown(
                """
                <div class="result-box spoof-box">
                    <h2 style="color: #E74C3C; margin: 0;">❌ مزيف</h2>
                    <p style="margin: 0;">تم الكشف عن هجوم تزييف!</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ لم يتم الحصول على نتيجة صالحة")
        
        # Metrics
        st.markdown("### 📈 المقاييس")
        
        col1, col2 = st.columns(2)
        with col1:
            # Anomaly score with gauge
            st.metric(
                "درجة الشذوذ",
                f"{anomaly_score:.2%}",
                delta=None
            )
            
            # Progress bar for anomaly score
            st.progress(
                min(1.0, anomaly_score),
                text=f"العتبة: {threshold:.2%}"
            )
        
        with col2:
            st.metric(
                "الثقة",
                f"{confidence:.2%}",
                delta=None
            )
            
            st.metric(
                "خطأ إعادة البناء",
                f"{reconstruction_error:.4f}",
                delta=None
            )
        
        # Additional info
        with st.expander("ℹ️ معلومات إضافية"):
            st.markdown(f"""
            - **التوقيت:** {timestamp}
            - **النموذج:** {result.get('model_name', 'N/A')}
            - **العتبة:** {threshold:.4f}
            - **الحالة:** {status}
            """)
        
        # History
        if st.session_state.history:
            st.markdown("### 📜 تاريخ التنبؤات")
            
            # Create history dataframe
            history_df = pd.DataFrame(st.session_state.history[-10:])
            
            if not history_df.empty:
                # Format columns
                display_df = history_df[["timestamp", "prediction", "status", "anomaly_score", "confidence"]].copy()
                
                # Map predictions to labels
                display_df["prediction"] = display_df["prediction"].map({
                    0: "🟢 حقيقي",
                    1: "🔴 مزيف",
                    -1: "❓ غير معروف"
                })
                
                display_df.columns = ["التوقيت", "التنبؤ", "الحالة", "درجة الشذوذ", "الثقة"]
                
                # Display
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=200
                )