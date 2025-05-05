from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

def load_yolo_model(model_path):
    """Loads the YOLO model from the specified path."""
    try:
        model = YOLO(model_path)
        logger.info(f"YOLO model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model from {model_path}: {e}")
        raise

def detect_objects(model, image, class_names_map, confidence_threshold=None):
    """
    Performs object detection using the YOLO model.
    Confidence threshold is optional; if None, uses model default or notebook hardcoded value if applicable.
    """
    detections_list = []
    if image is None:
        logger.error("Cannot perform detection on None image.")
        return detections_list

    try:
        # If confidence_threshold is None, use model default. If specified, pass it.
        predict_args = {'source': image, 'verbose': False}
        if confidence_threshold is not None:
            predict_args['conf'] = confidence_threshold

        results = model.predict(**predict_args)

        if not results or len(results) == 0:
            logger.warning("YOLO prediction returned no results.")
            return detections_list

        pred_result = results[0]  
        boxes = pred_result.boxes
        # Use the provided class_names_map from config
        if not class_names_map:
            logger.error("Class names map is required for detection processing.")
            # Fallback to model names
            class_names_map = pred_result.names
            logger.warning("Using class names map from model as fallback.")

        for i in range(len(boxes)):
            box_coords = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = class_names_map.get(class_id, f"Unknown_{class_id}")  

            detections_list.append({
                "box": box_coords,
                "confidence": confidence,
                "class_id": class_id,
                "name": class_name 
            })

        logger.info(f"Detected {len(detections_list)} objects.")
        return detections_list
    except Exception as e:
        logger.exception(f"Error during YOLO detection: {e}")  
        return []