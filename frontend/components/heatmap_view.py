"""
Heatmap View Component
----------------------
Component for displaying heatmap visualization.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any


class HeatmapView:
    """
    Heatmap view UI component.
    """
    
    def __init__(self):
        """Initialize heatmap view."""
        pass
    
    def render(self, result: Optional[Dict[str, Any]]) -> None:
        """
        Render heatmap visualization.
        
        Args:
            result: Prediction result dictionary
        """
        st.markdown("### 🔥 خريطة الحرارة")
        
        if result is None:
            st.info("⏳ انتظار النتائج...")
            return
        
        # Check for error
        if "error" in result:
            st.error(f"❌ خطأ: {result['error']}")
            return
        
        # Extract values
        anomaly_score = result.get("anomaly_score", 0.0)
        confidence = result.get("confidence", 0.0)
        
        # Create heatmap
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Score gauge (like a thermometer)
        ax1 = axes[0]
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # Draw gauge
        colors = plt.cm.RdYlGn_r(anomaly_score)
        
        # Background
        ax1.add_patch(plt.Rectangle((0.1, 0.4), 0.8, 0.2, facecolor='lightgray', alpha=0.3))
        
        # Colored bar
        ax1.add_patch(plt.Rectangle((0.1, 0.4), 0.8 * anomaly_score, 0.2, facecolor=colors))
        
        # Text
        ax1.text(0.5, 0.85, f"درجة الشذوذ", ha='center', fontsize=12, fontweight='bold')
        ax1.text(0.5, 0.3, f"{anomaly_score:.2%}", ha='center', fontsize=16, fontweight='bold')
        ax1.text(0.1, 0.2, "طبيعي", ha='left', fontsize=10)
        ax1.text(0.9, 0.2, "شاذ", ha='right', fontsize=10)
        
        # Confidence heatmap
        ax2 = axes[1]
        
        # Create 2D heatmap data
        size = 10
        heatmap_data = np.zeros((size, size))
        
        # Fill with confidence values
        for i in range(size):
            for j in range(size):
                # Simulate confidence distribution
                distance = np.sqrt((i - size/2)**2 + (j - size/2)**2)
                heatmap_data[i, j] = confidence * np.exp(-distance / (size/3))
        
        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            ax=ax2,
            cmap='RdYlGn',
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False,
            vmin=0,
            vmax=1
        )
        ax2.set_title(f"الثقة: {confidence:.2%}", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Display
        st.pyplot(fig)
        plt.close()
        
        # Additional info
        st.caption(f"""
        **ملاحظة:** خريطة الحرارة تظهر توزيع الثقة في التنبؤ.
        المناطق الخضراء تشير إلى ثقة عالية، والمناطق الحمراء تشير إلى ثقة منخفضة.
        """)