# plugins/openvino_plugin.py - Example OpenVINO plugin implementation
import numpy as np
from typing import Dict, Any, List
from .base import InferencePlugin
import cv2
from geti_sdk.deployment import Deployment

class GetiPlugin(InferencePlugin):
    """Implementation of InferencePlugin for OpenVINO models"""
    
    def __init__(self):
        self.model = None
        self.deployment = None
        self._task_type = "detection"  # Default, can be set during model loading
    
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load OpenVINO model"""
        self.deployment = Deployment.from_folder(model_path)
        self.deployment.load_inference_models(device=kwargs.get('device', 'CPU'))

        # Set task type if provided
        # if 'task_type' in kwargs:
            # self._task_type = kwargs['task_type']
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Pass through the image without preprocessing since Geti handles preprocessing internally"""
        return image
    
    def infer(self, preprocessed_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference using OpenVINO"""
        result = self.model({self.input_name: preprocessed_data})
        return result
    
    def postprocess(self, inference_result: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Pass through the inference_result without postprocessing since Geti handles postprocessing internally"""
        return inference_result
    
    @property
    def name(self) -> str:
        return "Geti"
    
    @property
    def supported_tasks(self) -> List[str]:
        return ["detection", "classification", "instance segmentation", "segmentation", "anomaly", "keypoint", "task chain"]
