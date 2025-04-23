import argparse
import logging
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from src.pipeline import ImageProcessor
from src.output.handler import OutputHandler
from src.ocr import PaddleOCRWrapper, TextCorrector
from src.detection import YOLODetector
from src.utils.file_utils import FileManager
from src.utils.config import load_output_config

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.enable("src")

def main():
    parser = argparse.ArgumentParser(description='Process app usage screenshots')
    parser.add_argument('input_dir', type=str, help='Path to directory with screenshots')
    parser.add_argument('--config', default='configs/settings.yaml', help='Configuration file path')
    args = parser.parse_args()

    # Load configuration
    config = load_output_config(Path(args.config))
    
    # Initialize components with config
    file_manager = FileManager(config.output_root)
    detector = YOLODetector(config.yolo_model_path)  # Pass model path from config
    ocr = PaddleOCRWrapper(config.ocr_params)  # Pass OCR params if needed
    
    # Initialize TextCorrector with config
    corrector = TextCorrector(
        json_path=config.app_names_json,
        min_match_score=config.min_match_score
    )
    
    # Initialize CSVGenerator with config
    csv_generator = CSVGenerator(config.csv_settings)
    
    # Create pipeline processor
    processor = ImageProcessor(
        detector=detector,
        ocr=ocr,
        corrector=corrector,
        output_folder=config.output_root
    )

    # Process files
    image_paths = list(Path(args.input_dir).glob("*.png")) + list(Path(args.input_dir).glob("*.jpg"))
    
    for img_path in tqdm(image_paths, desc="Processing screenshots"):
        try:
            result = processor.process_image(str(img_path), debug=config.debug)
            
            # Organize output files
            file_manager.organize_by_id(
                result.id_text,
                str(img_path),
                result.extracted_data
            )
            
            # Save to output formats
            output_handler.save({
                result.id_text: result.extracted_data
            })
            
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {str(e)}")
            continue

if __name__ == "__main__":
    configure_logging()
    main()