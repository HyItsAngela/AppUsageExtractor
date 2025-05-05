import cv2
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

def load_image(image_path):
    """Loads an image from the specified path."""
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        logger.debug(f"Image loaded: {image_path} with shape {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def save_image(image, output_path):
    """Saves an image to the specified path."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.debug(f"Image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False

def crop_region(image, box):
    """Crops a region from the image based on the bounding box."""
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        logger.warning(f"Invalid crop dimensions for box {box}, skipping crop.")
        return None
    return image[y1:y2, x1:x2]

def draw_debug_visualizations(image, debug_info, extracted_data):
    """
    Draws enhanced debug visualizations including reference lines, search lines,
    and lines connecting matched name/usage pairs.

    Args:
        image: The original image (numpy array BGR).
        debug_info: Dictionary containing intermediate results like raw detections,
                    reference_line, icon_search_lines, unmatched items, etc.
        extracted_data: List of successfully parsed app data dicts, each containing
                        'app_name', 'time_str', 'time_minutes', 'name_box_xywh',
                        'usage_box_xywh'.
    """
    vis_image = image.copy()
    h, w = vis_image.shape[:2]

    colors = {
        'app_name': (255, 100, 100), # Light Blue/Red
        'app_usage': (100, 255, 100), # Light Green
        'id': (100, 100, 255),        # Light Red/Blue
        'app_icon': (255, 255, 0), # Cyan
        'other': (180, 180, 180), # Lighter Gray
        'reference_line': (220, 220, 220), # Very Light Gray
        'search_line': (0, 165, 255), # Orange
        'unmatched': (255, 0, 255), # Magenta
        'match_link': (255, 255, 255) # White for connecting lines
    }
    thickness_raw = 1
    thickness_matched = 2
    thickness_line = 1
    thickness_link = 2

    # Draw Raw Detections
    raw_detections = debug_info.get("raw_detections", [])
    if raw_detections:
        logger.debug(f"Drawing {len(raw_detections)} raw detections.")
        for det in raw_detections:
            try: 
                box = list(map(int, det['box']))
                label = f"{det['name']}:{det['confidence']:.2f}"
                color = colors.get(det['name'], colors['other'])
                cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), color, thickness_raw)
                ocr_text = debug_info.get('ocr_results_detail', {}).get(tuple(box), {}).get('final_text', '')
                if ocr_text and ocr_text != "-1": label += f" T:'{ocr_text}'"
                cv2.putText(vis_image, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness_raw)
            except Exception as e:
                logger.warning(f"Could not draw raw detection {det}: {e}")


    # Draw Reference Line
    ref_line = debug_info.get("reference_line")
    if ref_line:
        logger.debug("Drawing reference line.")
        try:
            _, start, end = ref_line
            cv2.line(vis_image, tuple(map(int,start)), tuple(map(int,end)), colors['reference_line'], thickness_line + 1) 
        except Exception as e:
             logger.warning(f"Could not draw reference line {ref_line}: {e}")

    # Draw Icon Search Lines
    search_lines = debug_info.get("icon_search_lines", [])
    if search_lines:
        logger.debug(f"Drawing {len(search_lines)} icon search lines.")
        for item in search_lines:
            try:
                search_line_tuple = item.get('search_line')
                if search_line_tuple and len(search_line_tuple) == 3:
                    _, start, end = search_line_tuple
                    cv2.line(vis_image, tuple(map(int,start)), tuple(map(int,end)), colors['search_line'], thickness_line)
                    cv2.circle(vis_image, tuple(map(int,start)), 3, colors['search_line'], -1) # Mark start point (icon center)
            except Exception as e:
                logger.warning(f"Could not draw search line {item}: {e}")

    # Draw Connecting Lines and Highlight Matched Pairs
    logger.debug(f"Drawing highlights/links for {len(extracted_data)} matched pairs.")
    for match in extracted_data:
        try:
            name_box_xywh = match.get('name_box_xywh')
            usage_box_xywh = match.get('usage_box_xywh')

            if name_box_xywh and usage_box_xywh:
                nx, ny, nw, nh = map(int, name_box_xywh)
                ux, uy, uw, uh = map(int, usage_box_xywh)

                cv2.rectangle(vis_image, (nx, ny), (nx + nw, ny + nh), colors['app_name'], thickness_matched)
                cv2.rectangle(vis_image, (ux, uy), (ux + uw, uy + uh), colors['app_usage'], thickness_matched)

                # Calculate centroids
                name_cx = nx + nw // 2
                name_cy = ny + nh // 2
                usage_cx = ux + uw // 2
                usage_cy = uy + uh // 2

                # Draw connecting line
                cv2.line(vis_image, (name_cx, name_cy), (usage_cx, usage_cy), colors['match_link'], thickness_link)
            else:
                logger.warning(f"Skipping match visualization due to missing box info: {match.get('app_name')}")
        except Exception as e:
            logger.warning(f"Could not draw match connection for {match.get('app_name')}: {e}")


    # Highlight Unmatched Items
    logger.debug("Drawing unmatched items.") 
    for key in ["unmatched_app_names", "unmatched_app_usages", "unmatched_app_icons"]:
        unmatched_items = debug_info.get(key, [])
        if unmatched_items:
            for item in unmatched_items:
                try:
                    x, y, w, h = item['x'], item['y'], item['w'], item['h']
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), colors['unmatched'], thickness_raw + 1)
                except Exception as e:
                    logger.warning(f"Could not draw unmatched item {item}: {e}")

    return vis_image