from pathlib import Path
from typing import List, Union
import numpy as np
from ultralytics import YOLO
from .schemas import Detection 

class YOLODetector:
    def __init__(self):
        """Initialize YOLO detector with model weights."""
        try:
            model_path = (
                Path(__file__).resolve().parent.parent.parent 
                / "models/yolo/weights/best.pt"
            )
            if not model_path.exists():
                raise FileNotFoundError(f"Model file missing: {model_path}")
            self.model = YOLO(str(model_path))
            if not hasattr(self.model, 'predict'):
                raise RuntimeError("Model failed to load properly")
        except Exception as e:
            raise RuntimeError(f"YOLO init failed: {str(e)}") from e

    def detect(
        self, 
        image: Union[str, np.ndarray], 
        conf_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Detect objects in image using YOLO model.
        
        Args:
            image: Either file path or numpy array of image
            conf_threshold: Minimum confidence threshold for detections
            
        Returns:
            List of Detection objects with converted coordinates
        """
        try:
            if isinstance(image, str) and not Path(image).exists():
                raise FileNotFoundError(f"Image not found: {image}")
            elif not isinstance(image, (str, np.ndarray)):
                raise ValueError("Image must be path string or numpy array")

            results = self.model.predict(image, conf=conf_threshold)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    try:
                        # Convert center coordinates to top-left coordinates
                        x_center = float(box.xywh[0][0])
                        y_center = float(box.xywh[0][1])
                        width = float(box.xywh[0][2])
                        height = float(box.xywh[0][3])
                        
                        # Calculate top-left coordinates
                        x = int(round(x_center - width/2))
                        y = int(round(y_center - height/2))
                        
                        detections.append(Detection(
                            label=str(result.names[int(box.cls[0])]),
                            x=max(0, x),  # Ensure within image bounds
                            y=max(0, y),
                            w=int(round(width)),
                            h=int(round(height)),
                            confidence=float(box.conf[0])
                        ))
                    except Exception as box_error:
                        print(f"Skipping invalid detection box: {box_error}")
                        continue
            
            return detections

        except Exception as e:
            print(f"Detection failed: {str(e)}")
            return []  # Return empty list on failure