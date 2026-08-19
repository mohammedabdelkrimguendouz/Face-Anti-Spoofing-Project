"""
API Client for Face Anti-Spoofing System
"""

import requests
from typing import List, Dict, Any, Optional
import logging
import json

# =============================================================
# Setup
# =============================================================
logger = logging.getLogger(__name__)


class APIClient:
    """
    Client for Face Anti-Spoofing API.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API
        """
        # استيراد config هنا لتجنب المشاكل
        try:
            from frontend.config import config
            self.base_url = base_url or config.API_URL
        except:
            self.base_url = base_url or "http://localhost:8000"
        
        self.timeout = 30
        self.session = requests.Session()
        
        logger.info(f"API Client initialized with base URL: {self.base_url}")
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make an API request.
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                data=data,
                json=json_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        return self._request("GET", "/api/v1/health")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        result = self._request("GET", "/api/v1/models")
        if isinstance(result, list):
            return result
        return []
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        return self._request("GET", f"/api/v1/models/{model_name}")
    
    def load_model(self, model_name: str) -> Dict[str, str]:
        """Load a specific model."""
        return self._request("POST", f"/api/v1/models/{model_name}/load")
    
    def predict(
        self,
        features: List[List[List[float]]],
        model_name: str = "bilstm_vae",
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run prediction on features.
        """
        data = {
            "features": features,
            "model_name": model_name
        }
        
        if threshold is not None:
            data["threshold"] = threshold
        
        result = self._request(
            "POST",
            "/api/v1/predict/batch",
            json_data=data
        )
        
        # إذا كان هناك خطأ، نعيد نتيجة افتراضية
        if "error" in result:
            return {
                "predictions": [],
                "statuses": [],
                "anomaly_scores": [],
                "reconstruction_errors": [],
                "model_name": model_name,
                "threshold": threshold or 0.5,
                "error": result["error"]
            }
        
        return result
    
    def predict_single(
        self,
        features: List[List[float]],
        model_name: str = "bilstm_vae",
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run prediction on a single sample.
        """
        data = {
            "features": features,
            "model_name": model_name
        }
        
        if threshold is not None:
            data["threshold"] = threshold
        
        return self._request(
            "POST",
            "/api/v1/predict",
            json_data=data
        )