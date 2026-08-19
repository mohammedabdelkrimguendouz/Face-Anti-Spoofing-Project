"""
Model Selector Component
------------------------
Component for selecting and managing models.
"""

import streamlit as st
from typing import Optional, List, Dict, Any

from frontend.api.client import APIClient
from frontend.config import config


class ModelSelector:
    """
    Model selector UI component.
    """
    
    def __init__(self):
        """Initialize model selector."""
        self.api_client = APIClient()
        self.models = self._load_models()
    
    def _load_models(self) -> List[Dict[str, Any]]:
        """
        Load available models from API.
        
        Returns:
            List[Dict]: List of model information
        """
        try:
            models = self.api_client.list_models()
            return models
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            return []
    
    def render(self) -> str:
        """
        Render the model selector component.
        
        Returns:
            str: Selected model name
        """
        st.markdown("#### 🤖 اختر النموذج")
        
        if not self.models:
            st.warning("لا توجد نماذج متاحة")
            return config.DEFAULT_MODEL
        
        # Create model options
        model_options = {}
        for model in self.models:
            name = model.get("name", model.get("model_type", "Unknown"))
            model_type = model.get("model_type", "unknown")
            is_loaded = model.get("is_loaded", False)
            
            # Add status indicator
            status = "🟢" if is_loaded else "🔴"
            label = f"{status} {name}"
            model_options[label] = model_type
        
        # Selectbox
        selected_label = st.selectbox(
            "النموذج",
            options=list(model_options.keys()),
            index=0,
            key="model_selector_selectbox"
        )
        
        selected_model = model_options[selected_label]
        
        # Show model info
        if selected_model:
            st.caption(f"**النموذج:** {selected_model}")
            
            # Find model info
            model_info = next(
                (m for m in self.models if m.get("model_type") == selected_model),
                None
            )
            
            if model_info:
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"حجم المدخلات: {model_info.get('input_size', 'N/A')}")
                    st.caption(f"الطبقات المخفية: {model_info.get('hidden_size', 'N/A')}")
                with col2:
                    st.caption(f"العتبة: {model_info.get('threshold', 'N/A')}")
                    st.caption(f"الحالة: {'✅ محمّل' if model_info.get('is_loaded') else '❌ غير محمّل'}")
        
        return selected_model