import csv
import sqlite3
import pandas as pd
import os
import logging
from collections import defaultdict
import cv2  
import re

from .parsing import parse_time as parse_time_minutes, format_minutes

logger = logging.getLogger(__name__)

# Handles standard TXT output and saving original image
def organize_output_by_id(output_folder, id_text, image_path, image_array, extracted_data):
    """
    Organizes output by ID, saves original image, and simple OCR text file.
    Takes image_array (numpy array) to avoid re-reading.
    """
    # Create ID-based folder structure
    id_folders_root = os.path.join(output_folder, "id_folders")
    os.makedirs(id_folders_root, exist_ok=True)

    # Handle missing/unknown IDs
    clean_id = str(id_text).strip() if id_text else "unknown_id"
    clean_id = re.sub(r'[\\/*?:"<>|]', '_', clean_id)
    if not clean_id:
        clean_id = "unknown_id"

    id_folder = os.path.join(id_folders_root, f"ID_{clean_id}")
    os.makedirs(id_folder, exist_ok=True)

    # Save original image 
    image_filename = os.path.basename(image_path)
    output_image_path = os.path.join(id_folder, image_filename)
    if image_array is not None:
        try:
            cv2.imwrite(output_image_path, image_array)
            logger.debug(f"Original image saved to {output_image_path}")
        except Exception as e:
            logger.error(f"Error saving original image to {output_image_path}: {e}")
    else:
        logger.warning(f"No image array provided to save for {image_path}. Skipping image save.")

    # Save simple OCR results text file
    ocr_results_path = os.path.join(id_folder, f"ocr_results_{image_filename}.txt")
    try:
        with open(ocr_results_path, "w") as f:
            for data in extracted_data:
                app_name = data.get('app_name', 'N/A')
                app_usage = data.get('time_str', 'N/A')
                f.write(f"App Name: {app_name}\n")
                f.write(f"App Usage: {app_usage}\n\n")
        logger.info(f"Standard OCR results saved to {ocr_results_path}")
    except Exception as e:
        logger.error(f"Error saving standard OCR results to {ocr_results_path}: {e}")

    return id_folder

# Handles detailed DEBUG text output
def save_debug_txt(results, output_dir, image_filename):
    """Saves detailed debug processing results to a text file."""
    img_id = results.get('id', 'UNKNOWN_ID')
    id_source = results.get('id_source', 'none')
    if img_id in ["PROCESSING_FAILED", "NO_DETECTIONS", "ID Extraction Failed", "UNKNOWN_ID"]:
        img_id = f"FAILED_{img_id}"

    clean_id = re.sub(r'[\\/*?:"<>|]', '_', img_id)
    id_folder = os.path.join(output_dir, "id_folders", f"ID_{clean_id}")
    os.makedirs(id_folder, exist_ok=True)

    base_filename = os.path.splitext(image_filename)[0]
    output_txt_path = os.path.join(id_folder, f"{base_filename}_DEBUG_report.txt")

    try:
        with open(output_txt_path, 'w') as f:
            f.write(f"--- DEBUG Processing Report for: {image_filename} ---\n")
            f.write(f"Source Path: {results['image_path']}\n")
            f.write(f"Processing Status: {results['status']}\n")
            f.write(f"Final Extracted ID: {results['id']} (Source: {id_source}, Conf: {results['id_confidence']:.2f})\n")
            f.write(f"  OCR ID Attempt: {results['debug_info'].get('id_text_ocr', 'N/A')}\n")
            f.write(f"  Filename ID Attempt: {results['debug_info'].get('id_text_filename', 'N/A')}\n")
            if results["debug_info"].get("id_box_ocr"):
                f.write(f"  OCR ID Box: {results['debug_info']['id_box_ocr']}\n")

            raw_detections = results["debug_info"].get("raw_detections", [])
            f.write(f"\n--- Raw Detections ({len(raw_detections)}) ---\n")
            for det in raw_detections:
                f.write(f"  {det['name']} (Conf: {det['confidence']:.3f}) Box: {list(map(int, det['box']))} OCR: '{results['debug_info'].get('ocr_results', {}).get(tuple(det['box']), '')}'\n")

            if results["debug_info"].get("reference_line"):
                f.write(f"\n--- Reference Line ---\n")
                f.write(f"{results['debug_info']['reference_line']}\n")

            extracted_data = results.get('extracted_data', [])
            f.write(f"\n--- Successfully Parsed App Data ({len(extracted_data)}) ---\n")
            total_mins = results["debug_info"].get("total_usage_minutes", 0)
            total_str = results["debug_info"].get("total_usage_str", "N/A")
            f.write(f"Total Calculated Usage: {total_str} ({total_mins} minutes)\n\n")
            for app in extracted_data:
                f.write(f"App: {app['app_name']}\n")
                f.write(f"  Usage Str: {app['time_str']} -> Parsed Min: {app['time_minutes']}\n")

            f.write("\n--- Unmatched Items ---\n")
            for key in ["unmatched_app_names", "unmatched_app_usages", "unmatched_app_icons"]:
                items = results["debug_info"].get(key, [])
                f.write(f"{key.replace('_',' ').title()} ({len(items)}):\n")
                for item in items:
                    f.write(f"  - Label: {item.get('label')}, Box:({item.get('x')},{item.get('y')},{item.get('w')},{item.get('h')}), OCR: '{item.get('ocr_text')}'\n")

            f.write("\n--- End Report ---\n")

        logger.info(f"DEBUG report saved to {output_txt_path}")
    except Exception as e:
        logger.exception(f"Error saving DEBUG report to {output_txt_path}: {e}")

# Handles aggregated CSV output
def save_to_csv(aggregated_data_dict, output_csv_path):
    """
    Saves aggregated data to CSV. Expects aggregated_data_dict in format:
    {id_text: [list_of_app_data_dicts]}, where app_data_dict contains
    {'app_name': str, 'time_str': str, 'time_minutes': int}.
    Uses pre-calculated minutes for total, outputs time_str directly.
    """
    rows = []
    all_app_names = set()

    for id_text, app_data_list in aggregated_data_dict.items():
        for app in app_data_list:
            if app["app_name"] != "UNREADABLE_NAME":
                all_app_names.add(app["app_name"])

    sorted_app_names = sorted(list(all_app_names))

    for id_text, app_data_list in aggregated_data_dict.items():
        row = {"id": id_text}
        total_minutes = 0
        valid_entries_exist = False
        app_times_for_id = {}
        for app in app_data_list:
            app_name = app["app_name"]
            time_str = app["time_str"]
            time_minutes_val = app["time_minutes"]

            if app_name not in app_times_for_id:
                app_times_for_id[app_name] = time_str

            if time_minutes_val > 0:
                total_minutes += time_minutes_val
                valid_entries_exist = True
            elif time_minutes_val == 0 and time_str != "-1":
                valid_entries_exist = True

        for app_name in sorted_app_names:
            row[app_name] = app_times_for_id.get(app_name, '')

        if valid_entries_exist:
            row["total_usage"] = format_minutes(total_minutes)
        else:
            row["total_usage"] = "-1"

        rows.append(row)

    if not rows:
        logger.warning("No data to save to CSV after processing.")
        return

    try:
        df = pd.DataFrame(rows)
        cols_order = []
        if "id" in df.columns:
            cols_order.append("id")
        if "total_usage" in df.columns:
            cols_order.append("total_usage")
        other_cols = sorted([col for col in df.columns if col not in ["id", "total_usage"]])
        final_cols = cols_order + other_cols
        df = df.reindex(columns=final_cols, fill_value='')

        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Aggregated results saved to CSV: {output_csv_path}")
    except Exception as e:
        logger.exception(f"Error saving aggregated results to CSV {output_csv_path}: {e}")
        
def save_aggregated_to_sqlite(all_results, output_db_path):
    """
    Aggregates results and saves them into an SQLite database.
    Uses 'extracted_data' key instead of 'apps'.
    """
    if not all_results:
        logger.warning("No results to save to SQLite DB.")
        return

    app_usage_data = []
    processed_entries = set() 

    for result in all_results:
        # Filter results based on status and valid ID
        img_id = result.get('id', 'UNKNOWN_ID')
        if result.get('status') == "Failure" or img_id in ["PROCESSING_FAILED", "NO_DETECTIONS", "ID Extraction Failed", "UNKNOWN_ID", "OCR_EMPTY", "CROP_FAILED"]:
            continue 

        img_file_basename = os.path.basename(result.get('image_path', 'unknown_file'))

        for app in result.get('extracted_data', []): 
            app_name = app.get('app_name', 'UNKNOWN_APP')
            time_str = app.get('time_str', '-1') # Default to -1 if missing
            time_min = app.get('time_minutes', 0) # Default to 0 if missing

            db_time_minutes = time_min 

            entry_key = (img_id, app_name)
            if entry_key not in processed_entries:
                app_usage_data.append({
                    'id': img_id,
                    'app_name': app_name,
                    'usage_time_str': time_str,
                    'usage_time_minutes': db_time_minutes,
                    'source_image': img_file_basename
                })
                processed_entries.add(entry_key)

    if not app_usage_data:
        logger.warning("No valid app usage data found to save to SQLite DB after filtering.")
        return

    try:
        app_usage_df = pd.DataFrame(app_usage_data)
        os.makedirs(os.path.dirname(output_db_path), exist_ok=True)
        conn = sqlite3.connect(output_db_path)
        logger.info(f"Writing {len(app_usage_df)} rows to SQLite table 'app_usage' in {output_db_path}")
        app_usage_df.to_sql('app_usage', conn, if_exists='replace', index=False)
        conn.close()
        logger.info(f"Successfully saved results to SQLite DB.")

    except Exception as e:
        logger.exception(f"Error saving results to SQLite DB {output_db_path}: {e}") 
