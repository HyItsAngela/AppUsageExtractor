import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

def visualize_debugging(
    image: np.ndarray,
    matched_data: List[Dict],  
    unmatched_app_names: List[Dict],
    unmatched_app_usages: List[Dict],
    unmatched_app_icons: List[Dict],
    output_path: str,
    debug_info: Dict
) -> None:
    debug_image = image.copy()

    # Draw reference line
    if debug_info.get("reference_line"):
        try:
            orientation, start, end = debug_info["reference_line"]
            cv2.line(debug_image, start, end, (0, 255, 255), 2)
            cv2.putText(debug_image, f"Reference Line ({orientation})", 
                       (start[0]+10, start[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        except Exception as e:
            print(f"Error drawing reference line: {e}")

    # Draw search lines
    if debug_info.get("icon_search_lines"):
        for icon, search_line in debug_info["icon_search_lines"]:
            try:
                _, start, end = search_line
                cv2.line(debug_image, start, end, (255,0,255), 1)
                cv2.circle(debug_image, start, 3, (255,0,255), -1)
                cv2.circle(debug_image, end, 3, (255,0,255), -1)
            except Exception as e:
                print(f"Error drawing search line: {e}")

    # Draw matched components
    for pair in matched_data:
        # App Usage (green)
        if "app_usage" in pair:
            usage = pair["app_usage"]
            cv2.rectangle(debug_image, 
                         (usage["x"], usage["y"]), 
                         (usage["x"]+usage["w"], usage["y"]+usage["h"]), 
                         (0,255,0), 2)
            cv2.putText(debug_image, "Usage", 
                       (usage["x"], usage["y"]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # App Icon (yellow)
        if "app_icon" in pair:
            icon = pair["app_icon"]
            cv2.rectangle(debug_image,
                         (icon["x"], icon["y"]),
                         (icon["x"]+icon["w"], icon["y"]+icon["h"]),
                         (0,255,255), 2)
            cv2.putText(debug_image, "Icon",
                       (icon["x"], icon["y"]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # App Name (blue)
        if "app_name" in pair:
            name = pair["app_name"]
            cv2.rectangle(debug_image,
                         (name["x"], name["y"]),
                         (name["x"]+name["w"], name["y"]+name["h"]),
                         (255,0,0), 2)
            cv2.putText(debug_image, "Name",
                       (name["x"], name["y"]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Connection logic
        if "app_icon" in pair and "app_name" in pair:
            icon_center = (
                pair["app_icon"]["x"] + pair["app_icon"]["w"]//2,
                pair["app_icon"]["y"] + pair["app_icon"]["h"]//2
            )
            name_center = (
                pair["app_name"]["x"] + pair["app_name"]["w"]//2,
                pair["app_name"]["y"] + pair["app_name"]["h"]//2
            )
            
            if debug_info.get("icon_search_lines"):
                for _, line in debug_info["icon_search_lines"]:
                    _, start, _ = line
                    cv2.line(debug_image, name_center, start, (0,255,0), 2)
                    break

        # Fallback connection
        if "app_name" in pair and "app_icon" not in pair and "app_usage" in pair:
            name_center = (
                pair["app_name"]["x"] + pair["app_name"]["w"]//2,
                pair["app_name"]["y"] + pair["app_name"]["h"]//2
            )
            usage_center = (
                pair["app_usage"]["x"] + pair["app_usage"]["w"]//2,
                pair["app_usage"]["y"] + pair["app_usage"]["h"]//2
            )
            cv2.line(debug_image, name_center, usage_center, (0,255,0), 2)

    # Draw unmatched components (original style)
    for det in unmatched_app_names:
        cv2.rectangle(debug_image, 
                     (det["x"], det["y"]), 
                     (det["x"]+det["w"], det["y"]+det["h"]), 
                     (0,0,255), 2)
        cv2.putText(debug_image, "Unmatched Name",
                   (det["x"], det["y"]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    for det in unmatched_app_usages:
        cv2.rectangle(debug_image,
                     (det["x"], det["y"]),
                     (det["x"]+det["w"], det["y"]+det["h"]),
                     (0,255,255), 2)
        cv2.putText(debug_image, "Unmatched Usage",
                   (det["x"], det["y"]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    for det in unmatched_app_icons:
        cv2.rectangle(debug_image,
                     (det["x"], det["y"]),
                     (det["x"]+det["w"], det["y"]+det["h"]),
                     (128,0,128), 2)
        cv2.putText(debug_image, "Unmatched Icon",
                   (det["x"], det["y"]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,0,128), 2)

    cv2.imwrite(output_path, debug_image)