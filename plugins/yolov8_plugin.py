# plugins/yolov8_plugin.py - Example YOLOv8 plugin implementation
import numpy as np
from typing import Dict, Any, List
from .base import InferencePlugin
import cv2

class YOLOv8Plugin(InferencePlugin):
    """Implementation of InferencePlugin for Ultralytics YOLOv8 models"""
    
    def __init__(self):
        self.model = None
        self.conf_threshold = 0.25
        self._task_type = "detection"
    
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load YOLOv8 model using ultralytics"""
        try:
            # Import is placed here to make this optional dependency
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            
            # Set task type based on model type
            if hasattr(self.model, 'task') and self.model.task:
                self._task_type = self.model.task
            elif 'task_type' in kwargs:
                self._task_type = kwargs['task_type']
                
            # Set confidence threshold if provided
            if 'conf_threshold' in kwargs:
                self.conf_threshold = kwargs['conf_threshold']
                
        except ImportError:
            raise ImportError("Ultralytics package is required for YOLOv8Plugin. Install it with 'pip install ultralytics'")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """YOLOv8 handles preprocessing internally, just pass the image"""
        return image
    
    def infer(self, preprocessed_data: np.ndarray) -> Any:
        """Run inference using YOLOv8"""
        results = self.model(preprocessed_data, conf=self.conf_threshold)
        return results
    
    def postprocess(self, inference_result: Any) -> Dict[str, Any]:
        """Convert YOLOv8 results to standardized format"""
        if self._task_type == "detection":
            detections = []
            for result in inference_result:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i].xyxy[0].cpu().numpy()  # xyxy format (x1, y1, x2, y2)
                    conf = float(boxes[i].conf[0].cpu().numpy())
                    cls = int(boxes[i].cls[0].cpu().numpy())
                    
                    detections.append({
                        "label": cls,
                        "confidence": conf,
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                    })
            return {"detections": detections}
            
        elif self._task_type == "segmentation":
            segments = []
            for result in inference_result:
                if hasattr(result, 'masks') and result.masks is not None:
                    for i in range(len(result.masks)):
                        mask = result.masks[i].data.cpu().numpy()
                        cls = int(result.boxes[i].cls[0].cpu().numpy())
                        conf = float(result.boxes[i].conf[0].cpu().numpy())
                        
                        segments.append({
                            "label": cls,
                            "confidence": conf,
                            "mask": mask.tolist()  # Convert to list for JSON serialization
                        })
            return {"segments": segments}
            
        elif self._task_type == "classification":
            classifications = []
            for result in inference_result:
                if hasattr(result, 'probs'):
                    probs = result.probs.cpu().numpy()
                    top_indices = np.argsort(probs)[-5:][::-1]  # Top 5
                    
                    for idx in top_indices:
                        classifications.append({
                            "label": int(idx),
                            "confidence": float(probs[idx])
                        })
            return {"classifications": classifications}
            
        else:
            # Default fallback: return raw results
            return {"raw_output": str(inference_result)}
    
    @property
    def name(self) -> str:
        return "YOLOv8"
    
    @property
    def supported_tasks(self) -> List[str]:
        return ["detection", "segmentation", "classification", "pose"]