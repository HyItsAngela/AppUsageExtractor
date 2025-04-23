import os
import cv2
import logging
import re
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from detection import YOLODetector, GridProcessor, AppMatcher
from utils.file_utils import extract_id_from_filename
from .visualizer import visualize_debugging
from ocr import PaddleOCRWrapper, TextCorrector, OCRResult
from utils.config import load_config, AppConfig

logger = logging.getLogger(__name__)

@dataclass
class ProcessResult:
    id_text: Optional[str]
    extracted_data: List[Dict]
    debug_info: Dict

class ImageProcessor:
    def __init__(
        self, 
        detector: YOLODetector, 
        output_folder: str,
        ocr: PaddleOCRWrapper,
        corrector: TextCorrector 
    ):
        self.detector = detector
        self.output_folder = output_folder
        self.grid_processor = GridProcessor()
        self.matcher = AppMatcher()
        self.ocr = ocr
        self.corrector = corrector

    @classmethod
    def from_config(cls, config: AppConfig):
        return cls(
            detector=YOLODetector(config.yolo_model_path),
            output_folder=config.output.csv_path.parent,
            ocr=PaddleOCRWrapper(),
            corrector=TextCorrector(
                app_names_path=config.text_correction.app_names_json,
                min_match_score=config.text_correction.min_match_score,
                char_replacements=config.text_correction.char_replacements
            )
        )

    def process_image(self, image_path: str, debug: bool = True) -> ProcessResult:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        debug_info = {}
        extracted_data = []
        id_text = extract_id_from_filename(image_path)

        # Convert detections to original dict format
        detections = [self._detection_to_dict(d) for d in self.detector.detect(image, 0.3)]
        debug_info["raw_detections"] = [str(d) for d in detections]

        # Grid processing
        icon_centers = [
            (d["x"] + d["w"] // 2, d["y"] + d["h"] // 2)
            for d in detections if d["label"] == "app_icon"
        ]
        reference_line = self.grid_processor.find_reference_line(icon_centers, image.shape)
        debug_info["reference_line"] = reference_line

        match_results = self.matcher.match_app_components(detections, image.shape)
        matched_data = match_results["matched_data"]
        debug_info.update({
            "unmatched_names": match_results["unmatched"]["names"],
            "unmatched_usages": match_results["unmatched"]["usages"],
            "unmatched_icons": match_results["unmatched"]["icons"],
            "match_strategy": "reference_line" if reference_line else "fallback"
        })

        # OCR processing
        for pair in matched_data:
            app_data = {
                "app_name": "Unknown",
                "app_usage": "Unknown",
                "debug_info": {
                    "name": {"original": None, "corrections": []},
                    "usage": {"original": None, "corrections": []}
                }
            }

            if "app_name" in pair:
                name_data = pair["app_name"]
                ocr_result = self._process_ocr(image, name_data, "app_name")
                app_data["app_name"] = ocr_result["corrected"]
                app_data["debug_info"]["name"] = ocr_result["debug"]

            if "app_usage" in pair:
                usage_data = pair["app_usage"]
                ocr_result = self._process_ocr(image, usage_data, "app_usage")
                app_data["app_usage"] = ocr_result["corrected"]
                app_data["debug_info"]["usage"] = ocr_result["debug"]

            extracted_data.append(app_data)

        # ID extraction from image
        if not id_text:
            id_detections = [d for d in detections if d["label"] == "id"]
            if id_detections:
                id_data = id_detections[0]
                ocr_result = self._process_ocr(image, id_data, "id")
                id_text = ocr_result["corrected"]
                debug_info["id_recovery"] = ocr_result["debug"]

        if debug:
            self._save_debug_visualization(
                image, image_path, 
                matched_data, debug_info
            )

        return ProcessResult(
            id_text=id_text,
            extracted_data=extracted_data,
            debug_info=debug_info
        )

    def _process_ocr(self, image, bbox, region_type):
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        ocr_result = self.ocr.extract_text(image, (x, y, w, h)) or OCRResult(text="", confidence=0)
        
        raw_text = ocr_result.text.strip() or "Unknown"
        if region_type == "app_usage":
            raw_text = self._validate_time_format(raw_text)
        
        corrected, corrections = self.corrector.correct_text(raw_text, region_type)
        
        return {
            "corrected": corrected,
            "debug": {
                "original": raw_text,
                "corrections": corrections,
                "confidence": ocr_result.confidence,
                "bbox": (x, y, w, h),
                "ocr_debug": ocr_result.debug_info
            }
        }

    def _validate_time_format(self, text):
        if not re.match(r'^(\d+h)?(\d+m)?(\d+s)?$', text):
            return "00h00m00s"
        return text

    def _save_debug_visualization(self, image, image_path, matched_data, debug_info):
        debug_folder = os.path.join(self.output_folder, "debug_visualizations")
        os.makedirs(debug_folder, exist_ok=True)
        output_path = os.path.join(debug_folder, f"debug_{os.path.basename(image_path)}")
        
        visualize_debugging(
            image=image,
            matched_data=matched_data,
            unmatched_app_names=debug_info.get("unmatched_names", []),
            unmatched_app_usages=debug_info.get("unmatched_usages", []),
            unmatched_app_icons=debug_info.get("unmatched_icons", []),
            output_path=output_path,
            debug_info=debug_info
        )

    def _detection_to_dict(self, detection):
        return {
            "label": detection.label,
            "x": detection.x,
            "y": detection.y,
            "w": detection.w,
            "h": detection.h,
            "confidence": detection.confidence
        }