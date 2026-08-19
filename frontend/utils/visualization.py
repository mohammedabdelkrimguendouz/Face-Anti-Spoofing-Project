"""
Visualization Utilities
-----------------------
Helper functions for visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import streamlit as st


class VisualizationUtils:
    """
    Utility class for visualization functions.
    """
    
    @staticmethod
    def draw_prediction_box(
        frame: np.ndarray,
        prediction: int,
        anomaly_score: float,
        confidence: float
    ) -> np.ndarray:
        """
        Draw prediction box on frame.
        
        Args:
            frame: Input frame
            prediction: 0=Real, 1=Spoof
            anomaly_score: Anomaly score (0-1)
            confidence: Confidence score (0-1)
            
        Returns:
            np.ndarray: Frame with overlay
        """
        img = frame.copy()
        h, w = img.shape[:2]
        
        # Determine color
        if prediction == 0:  # Real
            color = (0, 255, 0)  # Green
            label = "REAL"
        else:  # Spoof
            color = (0, 0, 255)  # Red
            label = "SPOOF"
        
        # Draw border
        thickness = 3
        cv2.rectangle(img, (10, 10), (w - 10, h - 10), color, thickness)
        
        # Draw label background
        text = f"{label} | {anomaly_score:.2%} | {confidence:.2%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        
        # Background rectangle
        cv2.rectangle(
            img,
            (10, 10),
            (10 + text_w + 20, 10 + text_h + 20),
            (0, 0, 0),
            -1
        )
        
        # Text
        cv2.putText(
            img,
            text,
            (20, 10 + text_h + 10),
            font,
            font_scale,
            color,
            text_thickness
        )
        
        return img
    
    @staticmethod
    def create_gauge_chart(value: float, max_value: float = 1.0) -> plt.Figure:
        """
        Create a gauge chart.
        
        Args:
            value: Current value
            max_value: Maximum value
            
        Returns:
            plt.Figure: Gauge chart figure
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Normalize value
        norm_value = min(value / max_value, 1.0)
        
        # Create gauge
        colors = plt.cm.RdYlGn_r(norm_value)
        
        # Background
        ax.add_patch(plt.Rectangle((0.1, 0.4), 0.8, 0.2, facecolor='lightgray', alpha=0.3))
        
        # Colored bar
        ax.add_patch(plt.Rectangle((0.1, 0.4), 0.8 * norm_value, 0.2, facecolor=colors))
        
        # Text
        ax.text(0.5, 0.85, f"Score: {value:.2%}", ha='center', fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig
    
    @staticmethod
    def create_confidence_heatmap(confidence: float, size: int = 10) -> plt.Figure:
        """
        Create a confidence heatmap.
        
        Args:
            confidence: Confidence score
            size: Heatmap size
            
        Returns:
            plt.Figure: Heatmap figure
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # Create data
        data = np.zeros((size, size))
        center = size // 2
        radius = size // 3
        
        for i in range(size):
            for j in range(size):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                data[i, j] = confidence * np.exp(-distance / radius)
        
        # Plot
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f"Confidence: {confidence:.2%}")
        ax.axis('off')
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        return fig
    
    @staticmethod
    def create_anomaly_distribution(
        anomaly_scores: List[float],
        threshold: float
    ) -> plt.Figure:
        """
        Create anomaly score distribution plot.
        
        Args:
            anomaly_scores: List of anomaly scores
            threshold: Threshold value
            
        Returns:
            plt.Figure: Distribution plot
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Histogram
        ax.hist(
            anomaly_scores,
            bins=30,
            alpha=0.7,
            color='blue',
            edgecolor='black'
        )
        
        # Threshold line
        ax.axvline(
            threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold: {threshold:.2%}'
        )
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Anomaly Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def overlay_heatmap(
        frame: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay heatmap on frame.
        
        Args:
            frame: Input frame
            heatmap: Heatmap data
            alpha: Transparency factor
            
        Returns:
            np.ndarray: Frame with heatmap overlay
        """
        # Normalize heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_norm * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Resize to match frame
        h, w = frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap_colored, (w, h))
        
        # Overlay
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_resized, alpha, 0)
        
        return overlay