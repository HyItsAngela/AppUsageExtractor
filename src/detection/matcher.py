from typing import List, Dict, Any, Tuple
import numpy as np
from .grid_processor import GridProcessor
from .utils import distance_to_segment, create_search_line

class AppMatcher:
    @staticmethod
    def match_app_components(
        detections: List[dict], 
        image_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        app_names = [d for d in detections if d["label"] == "app_name"]
        app_usages = [d for d in detections if d["label"] == "app_usage"]
        app_icons = [d for d in detections if d["label"] == "app_icon"]
        
        unmatched_app_names = app_names.copy()
        unmatched_app_usages = app_usages.copy()
        unmatched_app_icons = app_icons.copy()
        matched_data = []  
        icon_search_lines = []

        # Process icons if present
        if app_icons:
            icon_centers = [
                (d["x"] + d["w"] // 2, 
                 d["y"] + d["h"] // 2) 
                for d in app_icons
            ]
            reference_line = GridProcessor.find_reference_line(icon_centers, image_shape)
            
            if reference_line:
                ref_orientation = reference_line[0]
                image_width, image_height = image_shape[1], image_shape[0]
                all_icon_bboxes = [d for d in app_icons]  

                # Create search lines
                for icon in app_icons:
                    center = (
                        icon["x"] + icon["w"] // 2,
                        icon["y"] + icon["h"] // 2
                    )
                    search_line = create_search_line(
                        center, ref_orientation, 
                        reference_line, all_icon_bboxes,
                        image_width, image_height
                    )
                    icon_search_lines.append((icon, search_line))

                # Match usages to icons 
                for app_usage in app_usages:
                    usage_center = (
                        app_usage["x"] + app_usage["w"] // 2,
                        app_usage["y"] + app_usage["h"] // 2
                    )
                    best_match = None
                    min_distance = float('inf')

                    for icon, search_line in icon_search_lines:
                        distance = distance_to_segment(usage_center, 
                                                      (search_line[1], search_line[2]))
                        if distance < 200 and distance < min_distance:
                            min_distance = distance
                            best_match = icon

                    if best_match and best_match not in [m.get("app_icon") for m in matched_data]:
                        matched_data.append({
                            "app_usage": app_usage,
                            "app_icon": best_match,
                            "app_name": None
                        })
                        unmatched_app_icons.remove(best_match)
                        unmatched_app_usages.remove(app_usage)

                # Match names to icons 
                for match in matched_data:
                    if not match["app_icon"]:
                        continue
                        
                    closest_name = None
                    min_name_distance = float('inf')
                    
                    for app_name in app_names:
                        name_center = (
                            app_name["x"] + app_name["w"] // 2,
                            app_name["y"] + app_name["h"] // 2
                        )
                        icon_center = (
                            match["app_icon"]["x"] + match["app_icon"]["w"] // 2,
                            match["app_icon"]["y"] + match["app_icon"]["h"] // 2
                        )
                        distance = ((name_center[0]-icon_center[0])**2 + 
                                   (name_center[1]-icon_center[1])**2)**0.5
                        
                        if distance < 120 and distance < min_name_distance:
                            min_name_distance = distance
                            closest_name = app_name

                    if closest_name:
                        match["app_name"] = closest_name
                        unmatched_app_names.remove(closest_name)

        # Fallback matching
        if not icon_search_lines:
            for app_usage in app_usages:
                usage_center = (
                    app_usage["x"] + app_usage["w"] // 2,
                    app_usage["y"] + app_usage["h"] // 2
                )
                closest_name = None
                min_distance = float('inf')
                
                for app_name in app_names:
                    name_center = (
                        app_name["x"] + app_name["w"] // 2,
                        app_name["y"] + app_name["h"] // 2
                    )
                    distance = ((usage_center[0]-name_center[0])**2 + 
                               (usage_center[1]-name_center[1])**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_name = app_name
                
                if closest_name:
                    matched_data.append({
                        "app_usage": app_usage,
                        "app_name": closest_name,
                        "app_icon": None
                    })
                    unmatched_app_usages.remove(app_usage)
                    unmatched_app_names.remove(closest_name)

        return {
            "matched_data": matched_data,
            "unmatched": {
                "names": unmatched_app_names,
                "usages": unmatched_app_usages,
                "icons": unmatched_app_icons
            }
        }