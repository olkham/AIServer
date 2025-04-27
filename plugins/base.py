# plugins/base.py - Define the base plugin interface
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class InferencePlugin(ABC):
    """
    Base class for all inference plugins.
    Each plugin must implement these methods.
    """
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load the model from the specified path"""
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """Preprocess the input image for inference"""
        pass
    
    @abstractmethod
    def infer(self, preprocessed_data: Any) -> Any:
        """Run inference on preprocessed data"""
        pass
    
    @abstractmethod
    def postprocess(self, inference_result: Any) -> Dict[str, Any]:
        """Postprocess inference results to standardized output format"""
        pass
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Complete inference pipeline: preprocess -> infer -> postprocess
        This is the main method that will be called by the server
        """
        preprocessed = self.preprocess(image)
        inference_result = self.infer(preprocessed)
        return self.postprocess(inference_result)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the plugin"""
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """Return a list of supported task types (e.g., 'detection', 'segmentation')"""
        pass
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the plugin (can be overridden)"""
        return {
            "name": self.name,
            "supported_tasks": self.supported_tasks
        }