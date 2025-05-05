from paddleocr import PaddleOCR
import logging
import numpy as np
import cv2
from . import parsing  

logger = logging.getLogger(__name__)


def initialize_ocr(use_gpu=False, lang='en', **kwargs):
    """Initializes the PaddleOCR engine."""
    try:
        ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False, **kwargs)
        logger.info(f"PaddleOCR engine initialized (GPU: {use_gpu}, Lang: {lang})")
        return ocr_engine
    except Exception as e:
        logger.exception(f"Error initializing PaddleOCR: {e}")
        raise

def read_text_with_paddleocr(paddle_ocr_engine, image):
    """Group vertically-aligned text boxes to form full labels"""
    X_ALIGNMENT_THRESHOLD = 30
    Y_GROUPING_THRESHOLD = 60
    result = paddle_ocr_engine.ocr(image, cls=True)
    if result is None or not result or not result[0]:
        return "No text detected", []
    line_items = []
    for line in result[0]:
        if line and isinstance(line, list) and len(line) == 2:
            points = line[0]
            text_conf = line[1]
            if isinstance(points, list) and len(points) == 4 and isinstance(text_conf, tuple) and len(text_conf) == 2:
                (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
                text, conf = text_conf
                center_x = (x0 + x1 + x2 + x3) / 4
                center_y = (y0 + y1 + y2 + y3) / 4
                line_items.append({'text': text, 'conf': conf, 'x': center_x, 'y': center_y, 'box': points})
            else:
                logger.warning(f"Skipping malformed line in OCR result: {line}")
        else:
            logger.warning(f"Skipping malformed block/line structure in OCR result: {line}")
    grouped_results = []
    line_items.sort(key=lambda item: item['x'])
    processed_indices = set()
    for i in range(len(line_items)):
        if i in processed_indices:
            continue
        base = line_items[i]
        current_group_items = [base]
        processed_indices.add(i)
        for j in range(i + 1, len(line_items)):
            if j in processed_indices:
                continue
            other = line_items[j]
            if abs(other['x'] - base['x']) < X_ALIGNMENT_THRESHOLD:
                if 0 < other['y'] - base['y'] < Y_GROUPING_THRESHOLD:
                    current_group_items.append(other)
                    processed_indices.add(j)
        current_group_items.sort(key=lambda g: g['y'])
        if current_group_items:
            full_text = " ".join(g['text'] for g in current_group_items)
            valid_confs = [g['conf'] for g in current_group_items if isinstance(g.get('conf'), (float, int))]
            avg_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
            grouped_results.append((full_text, avg_conf))
    combined_text = " ".join([text for text, _ in grouped_results]) if grouped_results else "No text groups formed"
    logger.debug(f"read_text_with_paddleocr combined text: {combined_text}")
    return combined_text, grouped_results


# Main OCR Function used by Processor
def extract_text_with_ocr(paddle_ocr_engine, image_full, bbox_xywh, region_type=None, known_app_names_list=None, config=None, confidence_threshold=0.4):
    """
    Extract text from a specific bounding box using PaddleOCR, apply corrections.
    """
    if image_full is None:
        return "-1", {"rejection_reason": "Input image is None"}
    if not bbox_xywh or len(bbox_xywh) != 4:
        return "-1", {"rejection_reason": "Invalid bounding box"}
    try:
        x, y, w, h = map(int, bbox_xywh)
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid bbox format: {bbox_xywh}. Error: {e}")
        return "-1", {"rejection_reason": f"Invalid bbox format: {bbox_xywh}"}
    img_h, img_w = image_full.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    if x1 >= x2 or y1 >= y2:
        return "-1", {"rejection_reason": f"Invalid ROI dimensions after clipping: Box={bbox_xywh}, Clipped={x1,y1,x2,y2}"}
    roi = image_full[y1:y2, x1:x2]
    roi_input = roi
    debug_info = {'original_text': None, 'all_lines': [], 'final_text': None, 'rejection_reason': None, 'applied_corrections': []}
    try:
        result = paddle_ocr_engine.ocr(roi_input, cls=True)
        if not result or not result[0]:
            debug_info['rejection_reason'] = "No text detected by OCR"
            logger.debug(f"OCR: No text detected in box {bbox_xywh}")
            return "-1", debug_info
        lines = []
        raw_lines_text = []
        for line in result[0]:
            if line and isinstance(line, list) and len(line) == 2:
                points = line[0]
                text_conf = line[1]
                if isinstance(points, list) and len(points) == 4 and isinstance(text_conf, tuple) and len(text_conf) == 2:
                    text, conf = text_conf
                    if conf >= confidence_threshold:
                        y0 = min(p[1] for p in points)
                        lines.append({'y': y0, 'text': text, 'conf': conf})
                        raw_lines_text.append(text)
                    else:
                        logger.debug(f"OCR: Line '{text}' rejected due to low confidence {conf:.2f} < {confidence_threshold}")
                else:
                    logger.warning(f"Skipping malformed line structure in OCR result: {line}")
            else:
                logger.warning(f"Skipping malformed block/line structure in OCR result: {line}")
        if not lines:
            debug_info['rejection_reason'] = "Text detected but confidence too low"
            logger.debug(f"OCR: Text below confidence threshold in box {bbox_xywh}")
            return "-1", debug_info
        lines.sort(key=lambda item: item['y'])
        combined_text = " ".join([item['text'] for item in lines])
        debug_info['original_text'] = combined_text
        debug_info['all_lines'] = [{'text': item['text'], 'conf': item['conf']} for item in lines]
        best_text = combined_text
        if region_type == "app_name":
            corrected_text = parsing.enhance_text_correction(best_text, "app_name", known_app_names_list, config)
            if corrected_text != best_text:
                debug_info['applied_corrections'].append(f"enhance: '{best_text}' -> '{corrected_text}'")
                logger.debug(f"OCR Correct (Name): '{best_text}' -> '{corrected_text}'")
            best_text = corrected_text
        elif region_type == "app_usage":
            corrected_text = parsing.validate_and_correct_usage(best_text)
            if corrected_text != best_text and corrected_text != "":
                debug_info['applied_corrections'].append(f"validate: '{best_text}' -> '{corrected_text}'")
                logger.debug(f"OCR Correct (Usage): '{best_text}' -> '{corrected_text}'")
            elif corrected_text == "":
                debug_info['rejection_reason'] = "Usage validation/correction failed"
                logger.warning(f"Usage validation failed for '{best_text}' in box {bbox_xywh}")
                debug_info['final_text'] = ""
                return "-1", debug_info
            best_text = corrected_text
        if not best_text or not best_text.strip():
            debug_info['rejection_reason'] = "Final text is empty"
            logger.debug(f"OCR: Final text empty for box {bbox_xywh}. Original: '{debug_info['original_text']}'")
            return "-1", debug_info
        debug_info['final_text'] = best_text
        logger.debug(f"OCR Success: Box {bbox_xywh}, Type '{region_type}', Result '{best_text}'")
        return best_text, debug_info
    except Exception as e:
        logger.exception(f"Error during OCR extraction for box {bbox_xywh}: {e}")
        debug_info['rejection_reason'] = f"Exception during OCR: {str(e)}"
        return "-1", debug_info
