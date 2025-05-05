import logging
import os
import re 
import json
import numpy as np
from . import image_utils, ocr, parsing, layout_analysis, detection

logger = logging.getLogger(__name__)

def extract_id_from_filename(image_path):
    """
    Extracts ID from filename if it contains 'OWL' in the format OWLXXXXX.
    Returns None if no valid ID is found in the filename.
    """
    filename = os.path.basename(image_path)
    owl_match = re.search(r'(OWL\d+)', filename, re.IGNORECASE)
    if owl_match:
        id_text = owl_match.group(1).upper()
        if re.fullmatch(r'OWL\d+', id_text):
            logger.info(f"Extracted ID '{id_text}' from filename: {filename}")
            return id_text
        else:
            logger.debug(f"Pattern 'OWL<digits>' found but cleaning invalid in {filename}. Extracted: {id_text}")
    logger.debug(f"No 'OWL<digits>' ID found in filename: {filename}")
    return None

# Extracts ID based on OCR detection result
def extract_id_from_ocr(detections, ocr_engine, image, id_class_name):
    id_text = "UNKNOWN_ID"
    id_confidence = 0.0
    id_box = None
    id_detections = [d for d in detections if d['name'] == id_class_name]
    if not id_detections:
        logger.warning("No ID region detected via OCR.")
        return id_text, id_confidence, id_box
    id_detections.sort(key=lambda d: d['confidence'], reverse=True)
    selected_id_det = id_detections[0]
    id_box = selected_id_det['box']
    id_crop = image_utils.crop_region(image, id_box)
    if id_crop is not None:
        try:
            result = ocr_engine.ocr(id_crop, cls=False)
            if result and result[0]:
                raw_id_text = " ".join([line[1][0] for line in result[0]])
                id_conf = np.mean([line[1][1] for line in result[0]])
            else:
                raw_id_text = ""
                id_conf = 0.0
        except Exception:
            raw_id_text = ""
            id_conf = 0.0
            logger.exception("Basic OCR failed for ID")

        id_text_cleaned = parsing.clean_text(raw_id_text)
        if not id_text_cleaned:
            id_text = "OCR_EMPTY"
            id_confidence = id_conf
        else:
            id_text = id_text_cleaned.upper()
            id_confidence = id_conf
            logger.info(f"Extracted ID via OCR: '{id_text}' (Conf: {id_confidence:.2f})")
    else:
        id_text = "CROP_FAILED"
    return id_text, id_confidence, id_box


def process_image_data(image_path, yolo_model, ocr_engine, config, known_app_names_list):
    """Processes a single image using the detailed layout analysis."""
    final_results = {
        "image_path": image_path, "id": "UNKNOWN_ID", "id_confidence": 0.0,
        "id_source": None, "status": "Failure", "extracted_data": [],
        "debug_info": {
            "raw_detections": [], "ocr_results_detail": {}, "id_box_ocr": None,
            "id_text_ocr": None, "id_text_filename": None, "reference_line": None,
            "reference_orientation": None, "icon_search_lines": [], "unmatched_app_names": [],
            "unmatched_app_usages": [], "unmatched_app_icons": [], "ids": [],
            "total_usage_str": "N/A", "total_usage_minutes": 0
        }
    }

    class_names = config['class_names']
    app_name_class = config['app_name_class_name']
    app_usage_class = config['app_usage_class_name']
    app_icon_class = config.get('app_icon_class_name', 'app_icon')
    id_class_ocr = config['id_class_name']
    yolo_conf = config.get('yolo_confidence_threshold', None)
    time_regex_hm = config['time_regex']

    image = image_utils.load_image(image_path)
    if image is None:
        return final_results

    all_detections_dict_list = detection.detect_objects(yolo_model, image, config['class_names'], yolo_conf)
    final_results["debug_info"]["raw_detections"] = all_detections_dict_list
    if not all_detections_dict_list:
        final_results["status"] = "No Detections"
        return final_results

    ocr_id, ocr_id_conf, ocr_id_box = extract_id_from_ocr(all_detections_dict_list, ocr_engine, image, id_class_ocr)
    final_results["debug_info"]["id_box_ocr"] = ocr_id_box
    final_results["debug_info"]["id_text_ocr"] = ocr_id
    filename_id = extract_id_from_filename(image_path)
    final_results["debug_info"]["id_text_filename"] = filename_id
    final_id, id_source, final_id_conf = "UNKNOWN_ID", "none", 0.0
    if filename_id:
        final_id, id_source, final_id_conf = filename_id, "filename", 1.0
    elif ocr_id not in ["UNKNOWN_ID", "OCR_EMPTY", "CROP_FAILED"]:
        final_id, id_source, final_id_conf = ocr_id, "ocr", ocr_id_conf
    else:
        logger.error(f"Failed to extract valid ID for {image_path}. OCR: '{ocr_id}', Filename: '{filename_id}'.")

    final_results["id"] = final_id
    final_results["id_source"] = id_source
    final_results["id_confidence"] = final_id_conf

    detections_for_layout = []
    ocr_debug_detail = {}

    for det in all_detections_dict_list:
        x1, y1, x2, y2 = map(int, det['box'])
        w, h = x2 - x1, y2 - y1
        label = det['name']
        conf = det['confidence']
        box_xywh = (x1, y1, w, h)

        internal_det = {
            "label": label, "x": x1, "y": y1, "w": w, "h": h,
            "conf": conf, "ocr_text": ""
        }

        if label in [app_name_class, app_usage_class]:
            region_type = "app_name" if label == app_name_class else "app_usage"
            corrected_text, ocr_debug_info = ocr.extract_text_with_ocr(
                ocr_engine, image, box_xywh,
                region_type=region_type,
                known_app_names_list=known_app_names_list,
                config=config
            )
            internal_det['ocr_text'] = corrected_text
            ocr_debug_detail[box_xywh] = ocr_debug_info

        elif label in [app_icon_class, id_class_ocr]:
            internal_det['ocr_text'] = ""

        if label in [app_name_class, app_usage_class, app_icon_class, id_class_ocr]:
            detections_for_layout.append(internal_det)

    final_results["debug_info"]["ocr_results_detail"] = ocr_debug_detail

    try:
        matched_pairs, updated_debug_info = layout_analysis.match_app_name_and_usage(
            detections_for_layout, image.shape, config
        )
        final_results["debug_info"].update(updated_debug_info)
    except Exception as e:
        logger.exception("Error during layout analysis and matching.")
        matched_pairs = []
        final_results["status"] = "Layout Analysis Failed"

    extracted_app_data = []
    for pair in matched_pairs:
        app_name_det = pair.get("app_name")
        app_usage_det = pair.get("app_usage")

        if app_name_det and app_usage_det:
            app_name_final = app_name_det.get('ocr_text', '')
            usage_time_str = app_usage_det.get('ocr_text', '')

            if app_name_final == "-1" or not app_name_final:
                app_name_final = "UNREADABLE_NAME"

            time_minutes = 0
            if usage_time_str != "-1" and usage_time_str:
                seconds = parsing.parse_time_hms_to_seconds(usage_time_str)
                if seconds is not None and seconds >= 0:
                    time_minutes = round(seconds / 60.0)
                    logger.debug(f"Parsed hms '{usage_time_str}' -> {seconds}s -> {time_minutes}min for '{app_name_final}'")
                else:
                    logger.warning(f"Failed to parse hms time string '{usage_time_str}' for app '{app_name_final}'. Treating as unreadable.")
                    usage_time_str = "-1"
                    time_minutes = 0
            else:
                time_minutes = 0
                if usage_time_str != "-1":
                    usage_time_str = "-1"

            extracted_app_data.append({
                "app_name": app_name_final,
                "time_str": usage_time_str,
                "time_minutes": time_minutes,
                "name_box_xywh": (app_name_det['x'], app_name_det['y'], app_name_det['w'], app_name_det['h']),
                "usage_box_xywh": (app_usage_det['x'], app_usage_det['y'], app_usage_det['w'], app_usage_det['h']),
                "name_confidence": app_name_det['conf'],
                "usage_confidence": app_usage_det['conf']
            })

        elif app_name_det and not app_usage_det:
            logger.debug(f"Found app name '{app_name_det.get('ocr_text', '')}' without matched usage.")

        elif not app_name_det and app_usage_det:
            logger.debug(f"Found usage time '{app_usage_det.get('ocr_text', '')}' without matched name.")

    final_results["extracted_data"] = extracted_app_data
    if final_results["status"] not in ["ID Extraction Failed", "Layout Analysis Failed"]:
        final_results["status"] = "Success" if extracted_app_data else "Success (No Apps Parsed)"

    total_mins = sum(app['time_minutes'] for app in extracted_app_data)
    final_results["debug_info"]["total_usage_str"] = parsing.format_minutes(total_mins)
    final_results["debug_info"]["total_usage_minutes"] = total_mins

    logger.info(f"Finished processing {image_path}. Status: {final_results['status']}, ID: {final_results['id']}, Apps: {len(extracted_app_data)}")
    return final_results