import argparse
import logging
import os
import sys
import time
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config_loader, data_processor, detection, image_utils, ocr, output_handler, utils

# Setup logging
utils.setup_logging()  
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Process a single smartphone usage screenshot (High Fidelity).")
    parser.add_argument("--input", required=True, help="Path to the input image file.")
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

    # Model Initialization
    try:
        logger.info("Loading models...")
        yolo_model = detection.load_yolo_model(config['model_path'])
        ocr_engine = ocr.initialize_ocr(use_gpu=config.get('use_gpu_ocr', False), lang=config.get('ocr_lang', 'en'))
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize models: {e}. Exiting.")  # Log trace
        sys.exit(1)

    # Processing
    logger.info(f"Processing image: {args.input}")
    processed_data = data_processor.process_image_data(
        args.input,
        yolo_model,
        ocr_engine,
        config,
        known_app_names_list
    )

    # Output Saving
    image_filename = os.path.basename(args.input)
    all_results_list = [processed_data]  
    input_image_array = image_utils.load_image(args.input)  

    # Save standard TXT file
    if 'txt' in output_formats:
        output_handler.organize_output_by_id(
            results_dir,
            processed_data['id'],
            args.input,  
            input_image_array,  
            processed_data['extracted_data']
        )

    # Save aggregated formats
    if 'csv' in output_formats or 'db' in output_formats:
        aggregated_dict = {}
        if processed_data['status'] not in ["Failure", "ID Extraction Failed", "Layout Analysis Failed", "No Detections"] and processed_data['id'] != "UNKNOWN_ID":
            aggregated_dict[processed_data['id']] = processed_data['extracted_data']

        if 'csv' in output_formats:
            if aggregated_dict:  
                csv_path = os.path.join(results_dir, config.get('csv_filename', 'usage_report.csv'))
                output_handler.save_to_csv(aggregated_dict, csv_path)
            else:
                logger.warning("No valid data to save to CSV for this image.")

        if 'db' in output_formats:
            if processed_data['status'] != "Failure": 
                db_path = os.path.join(results_dir, config.get('db_filename', 'usage.db'))
                output_handler.save_aggregated_to_sqlite([processed_data], db_path)
            else:
                logger.warning("Skipping DB save due to processing failure.")

    # Debug Outputs (conditional on debug_mode)
    if debug_mode:
        logger.info("Debug mode enabled. Saving debug outputs.")
        # Save detailed debug text file
        output_handler.save_debug_txt(processed_data, results_dir, image_filename)

        # Save visualization image
        if processed_data['status'] != 'Failure':
            if input_image_array is None: 
                input_image_array = image_utils.load_image(args.input)

            if input_image_array is not None:
                vis_image = image_utils.draw_debug_visualizations(
                    input_image_array,  
                    processed_data["debug_info"]  
                )
                debug_output_dir = os.path.join(results_dir, config.get('debug_dir', 'debug_visualizations'))
                os.makedirs(debug_output_dir, exist_ok=True)
                debug_image_filename = f"debug_{os.path.splitext(image_filename)[0]}.jpg"
                debug_image_path = os.path.join(debug_output_dir, debug_image_filename)
                success = image_utils.save_image(vis_image, debug_image_path)
                if success:
                    logger.info(f"Debug visualization saved to: {debug_image_path}")
            else:
                logger.error("Could not load input image for debug visualization.")

    logger.info(f"Finished processing {args.input}")

if __name__ == "__main__":
    main()