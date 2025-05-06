import argparse
import logging
import os
import sys
import time
import json
import re
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config_loader, processor, detection, image_utils, ocr, output_handler, utils, parsing

# Setup logging
utils.setup_logging() 
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Process a single smartphone usage screenshot (High Fidelity).")
    parser.add_argument("--input-dir", required=True, help="Path to the input image file.")
    parser.add_argument("--output-dir", default=None, help="Directory to save output files (uses config default if not set).")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--debug", action='store_true', default=None, help="Enable debug mode (saves visualization image & detailed txt). Overrides config.")
    parser.add_argument("--no-debug", action='store_true', default=None, help="Disable debug mode. Overrides --debug and config.")
    parser.add_argument("--output-formats", default=None, help="Comma-separated list of output formats (e.g., 'csv,db,txt'). Overrides config.")

    args = parser.parse_args()

    # Configuration Loading
    try:
        config = config_loader.load_config(args.config)
    except Exception:
        logger.exception("Failed to load configuration. Exiting.")
        sys.exit(1)

    # Load Known App Names from JSON
    known_app_names_list = []
    app_names_path = config.get('app_names_json_path', 'data/scraped_app_names.json')
    try:
        with open(app_names_path, 'r') as f:
            known_app_names_list = json.load(f)
        if not isinstance(known_app_names_list, list):
            logger.warning(f"App names file {app_names_path} does not contain a valid JSON list. Using empty list.")
            known_app_names_list = []
        else:
            logger.info(f"Loaded {len(known_app_names_list)} known app names from {app_names_path}")
    except FileNotFoundError:
        logger.warning(f"App names JSON file not found at {app_names_path}. Proceeding without known app names list.")
        known_app_names_list = []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from app names file: {app_names_path}. Using empty list.")
        known_app_names_list = []
    except Exception as e:
        logger.error(f"An unexpected error occurred loading app names from {app_names_path}: {e}")
        known_app_names_list = []

    # Determine debug mode
    if args.no_debug:
        debug_mode = False
    elif args.debug is not None:  
        debug_mode = True
    else:
        debug_mode = config.get('debug_default', False)  # Fallback to config default

    # Determine output formats
    if args.output_formats:
        output_formats = [fmt.strip().lower() for fmt in args.output_formats.split(',')]
    else:
        output_formats = config.get('output_formats', ['csv', 'db', 'txt'])  # Default from config

    # Determine output directory
    results_dir = args.output_dir or config.get('results_dir', 'results')  # Use arg or config default

    # Find Input Files
    image_files = find_image_files(args.input_dir)
    if not image_files:
        logger.error(f"No image files found in directory: {args.input_dir}")
        sys.exit(1)
    logger.info(f"Found {len(image_files)} images to process in {args.input_dir}")

    # Model Initialization
    try:
        logger.info("Loading models...")
        yolo_model = detection.load_yolo_model(config['model_path'])
        ocr_engine = ocr.initialize_ocr(use_gpu=config.get('use_gpu_ocr', False), lang=config.get('ocr_lang', 'en'))
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize models: {e}. Exiting.")  
        sys.exit(1)

    # Batch Processing
    all_results = []
    logger.info("Starting batch processing...")
    total_start_time = time.time()

    for image_path in tqdm(image_files, desc="Processing Images"):
        logger.debug(f"Processing batch image: {image_path}")
        img_start_time = time.time()
        processed_data = processor.process_image_data(
            image_path, yolo_model, ocr_engine, config, known_app_names_list
        )
        img_end_time = time.time()
        logger.debug(f"Image {os.path.basename(image_path)} processed in {img_end_time - img_start_time:.2f} seconds.")
        all_results.append(processed_data)

        # Handle per-image outputs (Standard TXT, Debug Outputs)
        image_filename = os.path.basename(image_path)
        input_image_array = None 
        if 'txt' in output_formats:
             try:
                 input_image_array = image_utils.load_image(image_path)
                 if input_image_array is not None: output_handler.organize_output_by_id(results_dir, processed_data['id'], image_path, input_image_array, processed_data['extracted_data'])
                 else: logger.error(f"Could not load image {image_path} for standard TXT output.")
             except Exception as e: logger.error(f"Error saving standard TXT for {image_path}: {e}")
        if debug_mode:
            try: output_handler.save_debug_txt(processed_data, results_dir, image_filename)
            except Exception as e: logger.error(f"Error saving debug TXT for {image_path}: {e}")
            if processed_data['status'] != 'Failure':
                if input_image_array is None: input_image_array = image_utils.load_image(image_path)
                if input_image_array is not None:
                    try:
                        vis_image = image_utils.draw_debug_visualizations(input_image_array, processed_data["debug_info"], processed_data["extracted_data"])
                        debug_output_dir = os.path.join(results_dir, config.get('debug_dir', 'debug_visualizations')); os.makedirs(debug_output_dir, exist_ok=True)
                        debug_image_filename = f"debug_{os.path.splitext(image_filename)[0]}.jpg"; debug_image_path = os.path.join(debug_output_dir, debug_image_filename)
                        image_utils.save_image(vis_image, debug_image_path); logger.debug(f"Debug visualization saved to: {debug_image_path}")
                    except Exception as e: logger.error(f"Error saving debug visualization for {image_path}: {e}")
                else: logger.error(f"Could not load image {image_path} for debug visualization.")
        del input_image_array 

    total_end_time = time.time()
    logger.info(f"Batch processing loop finished in {total_end_time - total_start_time:.2f} seconds.")

    # Handle aggregated outputs (CSV, SQLite)
    logger.info("Aggregating results for final outputs...")
    aggregated_csv_data = defaultdict(list); valid_results_for_aggregation = 0
    for result in all_results:
        img_id = result.get('id', 'UNKNOWN_ID')
        if result['status'] != "Failure" and img_id not in ["PROCESSING_FAILED", "NO_DETECTIONS", "ID Extraction Failed", "UNKNOWN_ID", "OCR_EMPTY", "CROP_FAILED"]:
            app_list = result.get('extracted_data', [])
            if app_list: aggregated_csv_data[img_id].extend(app_list); valid_results_for_aggregation += 1
    if 'csv' in output_formats:
         if aggregated_csv_data:
             logger.info(f"Aggregated data for {len(aggregated_csv_data)} unique IDs from {valid_results_for_aggregation} valid processed images.")
             csv_path = os.path.join(results_dir, config.get('csv_filename', 'usage_report.csv')); output_handler.save_to_csv(aggregated_csv_data, csv_path)
         else: logger.warning("No valid data aggregated to save to CSV.")
    if 'db' in output_formats:
         valid_results_for_db = [r for r in all_results if r['status'] != "Failure" and r['id'] not in ["PROCESSING_FAILED", "NO_DETECTIONS", "ID Extraction Failed", "UNKNOWN_ID", "OCR_EMPTY", "CROP_FAILED"]]
         if valid_results_for_db: db_path = os.path.join(results_dir, config.get('db_filename', 'usage.db')); output_handler.save_aggregated_to_sqlite(valid_results_for_db, db_path)
         else: logger.warning("No valid results to save to SQLite DB.")
    logger.info("All processing finished.")

if __name__ == "__main__":
    main()
