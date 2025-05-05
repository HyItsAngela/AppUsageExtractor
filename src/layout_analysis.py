import logging
import numpy as np
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

def get_centroid(box_dict):
    """Calculates the centroid from a detection dictionary with x, y, w, h keys."""
    if not all(k in box_dict for k in ['x', 'y', 'w', 'h']):
        logger.error(f"Cannot calculate centroid, missing keys in dict: {box_dict}")
        return (0,0)
    x, y, w, h = box_dict['x'], box_dict['y'], box_dict['w'], box_dict['h']
    return (x + w / 2.0, y + h / 2.0)

def detect_grid(icon_centers):
    """Detects the grid structure of icons and determines number of rows and columns."""
    if len(icon_centers) < 2:
        logger.info("Not enough icons to determine a grid.")
        return 0, 0 

    centers = np.array(icon_centers)
    MAX_ROW_GAP = 50  # Sorting by Y
    row_breaks = []

    # Sort by y-coordinates to detect row structure
    y_sorted = centers[np.argsort(centers[:, 1])]
    for i in range(1, len(y_sorted)):
        if y_sorted[i, 1] - y_sorted[i-1, 1] > MAX_ROW_GAP:
            row_breaks.append(i)

    rows = np.split(y_sorted, row_breaks)
    num_rows = len(rows)

    # Count max columns by checking each row
    num_cols = max(len(row) for row in rows) if rows else 0

    logger.info(f"Detected Grid: {num_rows} rows, {num_cols} columns")
    return num_rows, num_cols


def find_reference_line(icon_centers, image_shape, max_gap=60, max_angle=25, min_icons=3):
    """
    Detects reference line based on detected grid structure. (Copied from User Input)
    """
    logger.debug(">>> ENTERING find_reference_line function <<<")
    logger.debug(f"Received {len(icon_centers)} icon centers. Params: max_gap={max_gap}, max_angle={max_angle}, min_icons={min_icons}")
    if not icon_centers or len(icon_centers) < min_icons:
        logger.warning(f"EXITING find_reference_line early: Not enough icon centers ({len(icon_centers)}) provided (min: {min_icons}).")
        return None
    
    if not icon_centers or len(icon_centers) < min_icons: 
        logger.warning(f"Not enough icon centers ({len(icon_centers)}) provided to find reference line (min: {min_icons}).")
        return None

    num_rows, num_cols = detect_grid(icon_centers) 

    if num_rows == 0 or num_cols == 0:
        logger.warning("Grid detection failed (0 rows or 0 cols), cannot find reference line.")
        return None

    def line_consistency_score(points_list):
        """Calculates how consistently spaced and aligned the points are."""
        points = np.array(points_list) 
        if len(points) < 2: return 0

        spacings = [np.linalg.norm(points[i] - points[i-1]) for i in range(1, len(points))]
        directions = [points[i] - points[i-1] for i in range(1, len(points))]
        norm_directions = [d / np.linalg.norm(d) for d in directions if np.linalg.norm(d) > 1e-6] 

        if len(norm_directions) < 2: 
            angle_variance = 0
        else:
            angle_variance = sum(np.arccos(np.clip(np.dot(norm_directions[i], norm_directions[i-1]), -1.0, 1.0))
                              for i in range(1, len(norm_directions)))

        spacing_std = np.std(spacings) if len(spacings) > 1 else 0
        angle_std = angle_variance / max(1, len(norm_directions)-1) if len(norm_directions) > 1 else 0

        return 1 / (1 + spacing_std + angle_std + 1e-6)

    def compute_avg_spacing(points_list, axis=0):
        """Computes average spacing between icons along a given axis (0=x, 1=y)."""
        if len(points_list) < 2: return float('inf')
        points = np.array(points_list) 
        # Sort points based on the specified axis
        sorted_indices = np.argsort(points[:, axis])
        sorted_points = points[sorted_indices]
        # Calculate gaps between consecutive points along the axis
        gaps = np.abs(np.diff(sorted_points[:, axis]))
        return np.mean(gaps) if gaps.size > 0 else float('inf')

    avg_col_spacing = compute_avg_spacing(icon_centers, axis=0)  # Spacing along x
    avg_row_spacing = compute_avg_spacing(icon_centers, axis=1)  # Spacing along y

    logger.debug(f"Avg Column Spacing (X): {avg_col_spacing:.2f}, Avg Row Spacing (Y): {avg_row_spacing:.2f}")

    # Prioritize the orientation with smaller average spacing
    prioritize_columns = avg_col_spacing < avg_row_spacing 

    logger.info(f"Prioritizing {'Columns (Vertical Lines)' if prioritize_columns else 'Rows (Horizontal Lines)'}")

    def try_orientation(centers_list, orientation='vertical'):
        """Attempts to find the best reference line in the specified orientation."""
        logger.debug(f"Attempting orientation: {orientation}")
        if len(centers_list) < min_icons:
            logger.debug(f"Too few centers ({len(centers_list)}) for min_icons ({min_icons}) in try_orientation.")
            return None
        centers_np = np.array(centers_list)
        sort_axis = 1 if orientation == 'vertical' else 0
        sorted_indices = np.argsort(centers_np[:, sort_axis])
        centers_sorted = centers_np[sorted_indices]
        logger.debug(f"Sorted centers for {orientation}: {centers_sorted.tolist()}")

        best_line_points = None; best_score = -1
        for start_idx in range(len(centers_sorted)):
            current_line = [centers_sorted[start_idx]]; used_indices = {start_idx}
            logger.debug(f"  Starting new line attempt from index {start_idx}: {current_line[0].tolist()}")
            while True:
                last_point = current_line[-1]; best_next_point = None; best_next_idx = -1
                min_dist_sq = (max_gap**2); best_angle_rad = np.radians(max_angle)

                normalized_direction_vec = None 
                if len(current_line) >= 2:
                    direction_vec = last_point - current_line[-2]
                    norm = np.linalg.norm(direction_vec)
                    if norm > 1e-6:
                        normalized_direction_vec = direction_vec.astype(float) / norm

                found_next_in_iter = False
                for idx in range(len(centers_sorted)):
                    if idx in used_indices: continue
                    candidate_point = centers_sorted[idx]
                    if orientation == 'vertical' and candidate_point[1] <= last_point[1]: continue
                    if orientation == 'horizontal' and candidate_point[0] <= last_point[0]: continue
                    dist_sq = np.sum((candidate_point - last_point)**2)
                    if dist_sq >= min_dist_sq: continue

                    angle_ok = False
                    if normalized_direction_vec is not None:
                        candidate_vec = candidate_point - last_point
                        norm_cand = np.linalg.norm(candidate_vec)
                        if norm_cand > 1e-6:
                            candidate_vec_normalized = candidate_vec.astype(float) / norm_cand 
                            dot_product = np.clip(np.dot(normalized_direction_vec, candidate_vec_normalized), -1.0, 1.0)
                            angle_rad = np.arccos(dot_product)
                            if angle_rad <= best_angle_rad: angle_ok = True
                        else: angle_ok = True
                    else: angle_ok = True

                    if angle_ok:
                        logger.debug(f"    Found potential next point {idx}: {candidate_point.tolist()} (Dist: {np.sqrt(dist_sq):.1f}, Angle OK: {angle_ok})")
                        best_next_point = candidate_point; best_next_idx = idx; found_next_in_iter = True; break

                if found_next_in_iter:
                    logger.debug(f"    Adding point {best_next_idx} {best_next_point.tolist()} to line.")
                    current_line.append(best_next_point); used_indices.add(best_next_idx)
                else:
                    logger.debug(f"    No suitable next point found for line starting {current_line[0].tolist()} (length {len(current_line)}).")
                    break
            logger.debug(f"  Finished line attempt. Length: {len(current_line)}")
            if len(current_line) >= min_icons:
                score = len(current_line) * line_consistency_score(current_line)
                logger.debug(f"  Line meets min_icons ({min_icons}). Score: {score:.4f}")
                if score > best_score:
                    logger.debug(f"    New best line found! Score: {score:.4f} > {best_score:.4f}")
                    best_score = score; best_line_points = current_line
        if best_line_points: logger.debug(f"Best line found for {orientation}: {len(best_line_points)} points, Score: {best_score:.4f}")
        else: logger.debug(f"No line meeting criteria found for {orientation}.")
        return best_line_points


    # Try finding the line in the prioritized orientation first
    num_rows, num_cols = detect_grid(icon_centers)
    logger.debug(f"Grid Check: num_rows={num_rows}, num_cols={num_cols}")
    if num_rows == 0 or num_cols == 0: logger.warning("Grid detection failed, cannot find reference line."); return None
    avg_col_spacing = compute_avg_spacing(icon_centers, axis=0); avg_row_spacing = compute_avg_spacing(icon_centers, axis=1)
    logger.debug(f"Avg Spacing X: {avg_col_spacing:.2f}, Y: {avg_row_spacing:.2f}")
    prioritize_columns = avg_col_spacing < avg_row_spacing
    logger.info(f"Prioritizing {'Columns (Vertical Lines)' if prioritize_columns else 'Rows (Horizontal Lines)'}")
    primary_orientation = 'vertical' if prioritize_columns else 'horizontal'; secondary_orientation = 'horizontal' if prioritize_columns else 'vertical'
    logger.debug(f"Trying primary orientation: {primary_orientation}"); best_line = try_orientation(icon_centers, primary_orientation)
    if best_line is None: logger.debug(f"Primary failed, trying secondary: {secondary_orientation}"); best_line = try_orientation(icon_centers, secondary_orientation)

    if best_line is None:
        logger.warning("Could not find a suitable reference line in either orientation.")
        logger.debug("<<< EXITING find_reference_line function (No line found) >>>") 
        return None

    logger.info(f"Found best reference line candidate with {len(best_line)} points.")
    best_line_np = np.array(best_line); x = best_line_np[:, 0]; y = best_line_np[:, 1]
    is_vertical = np.std(y) > np.std(x); final_orientation = "vertical" if is_vertical else "horizontal"
    try:
        if is_vertical: A = np.vstack([y, np.ones(len(y))]).T; m, c = np.linalg.lstsq(A, x, rcond=None)[0]; start_y = 0; start_x = m * start_y + c; end_y = image_shape[0]; end_x = m * end_y + c
        else: A = np.vstack([x, np.ones(len(x))]).T; m, c = np.linalg.lstsq(A, y, rcond=None)[0]; start_x = 0; start_y = m * start_x + c; end_x = image_shape[1]; end_y = m * end_x + c
    except np.linalg.LinAlgError as e: logger.error(f"Linear regression failed: {e}"); return None
    img_h, img_w = image_shape[0], image_shape[1]
    start_x_c, start_y_c = np.clip(start_x, 0, img_w), np.clip(start_y, 0, img_h)
    end_x_c, end_y_c = np.clip(end_x, 0, img_w), np.clip(end_y, 0, img_h)
    final_line_tuple = (final_orientation, (int(start_x_c), int(start_y_c)), (int(end_x_c), int(end_y_c)))
    logger.info(f"Final Reference Line: Orientation={final_orientation}, Start={final_line_tuple[1]}, End={final_line_tuple[2]}")
    logger.debug("<<< EXITING find_reference_line function (Line found) >>>") 
    return final_line_tuple


def create_search_line(center, reference_line, all_icon_bboxes, image_width, image_height, tolerance=5):
    """
    Create a search line perpendicular to the reference line, starting from 'center'.
    Stops if it hits another icon's bounding box (within tolerance). (Copied from User Input)
    """
    if reference_line is None:
        logger.warning("Cannot create search line without a reference line.")
        return None

    ref_orientation, ref_start, ref_end = reference_line
    ref_dx = ref_end[0] - ref_start[0]
    ref_dy = ref_end[1] - ref_start[1]

    # Calculate perpendicular direction vector (normal vector)
    # Determine search direction based on orientation (search right/down first)
    if ref_orientation == "vertical":
        # Search horizontally (typically right)
        search_dx, search_dy = 1.0, 0.0
    else: # Horizontal reference line
        # Search vertically (typically down)
        search_dx, search_dy = 0.0, 1.0

    # Normalize search direction
    magnitude = np.sqrt(search_dx**2 + search_dy**2)
    if magnitude > 1e-6:
        search_dx /= magnitude
        search_dy /= magnitude
    else: 
        logger.error("Zero magnitude search direction vector.")
        return None 

    start_x, start_y = center 

    # Determine the furthest possible endpoint in the search direction (image edge)
    t_max = float('inf')
    if search_dx > 1e-6: # Moving right
        t_max = min(t_max, (image_width - start_x) / search_dx)
    elif search_dx < -1e-6: # Moving left
        t_max = min(t_max, (0 - start_x) / search_dx)

    if search_dy > 1e-6: # Moving down
        t_max = min(t_max, (image_height - start_y) / search_dy)
    elif search_dy < -1e-6: # Moving up
        t_max = min(t_max, (0 - start_y) / search_dy)

    if t_max == float('inf'):
        logger.warning("Could not determine boundary intersection for search line.")
        t_max = max(image_width, image_height)

    # Now, check intersections with other icon bounding boxes
    min_intersect_t = t_max # Shorten the line if it hits another icon
    current_icon_bbox_dict = None
    # Find the bbox corresponding to the start 'center' to avoid self-intersection check
    for bbox_dict in all_icon_bboxes:
        bb_center_x = bbox_dict['x'] + bbox_dict['w'] // 2
        bb_center_y = bbox_dict['y'] + bbox_dict['h'] // 2
        if abs(start_x - bb_center_x) < 1 and abs(start_y - bb_center_y) < 1:
            current_icon_bbox_dict = bbox_dict
            break

    for other_bbox_dict in all_icon_bboxes:
        if current_icon_bbox_dict and other_bbox_dict == current_icon_bbox_dict:
            continue

        x1 = other_bbox_dict['x'] - tolerance
        y1 = other_bbox_dict['y'] - tolerance
        x2 = other_bbox_dict['x'] + other_bbox_dict['w'] + tolerance
        y2 = other_bbox_dict['y'] + other_bbox_dict['h'] + tolerance

        t_near = -float('inf')
        t_far = float('inf')

        for axis in range(2): 
            start_coord = center[axis]
            direction_coord = [search_dx, search_dy][axis]
            b_min = [x1, y1][axis]
            b_max = [x2, y2][axis]

            if abs(direction_coord) < 1e-6: 
                if start_coord < b_min or start_coord > b_max:
                    t_near = float('inf') 
                    break
            else:
                t1 = (b_min - start_coord) / direction_coord
                t2 = (b_max - start_coord) / direction_coord
                if t1 > t2: t1, t2 = t2, t1 

                t_near = max(t_near, t1)
                t_far = min(t_far, t2)

                if t_near > t_far or t_far < 0: 
                    t_near = float('inf')
                    break

        if t_near < min_intersect_t and t_near >= 0: 
            min_intersect_t = t_near

    # Final endpoint based on the minimum intersection distance found
    final_t = min_intersect_t
    final_end_x = start_x + search_dx * final_t
    final_end_y = start_y + search_dy * final_t

    # Clip final endpoint to image boundaries
    final_end_x = np.clip(final_end_x, 0, image_width)
    final_end_y = np.clip(final_end_y, 0, image_height)

    search_line_tuple = (ref_orientation, (int(start_x), int(start_y)), (int(final_end_x), int(final_end_y)))
    return search_line_tuple

def distance_to_segment(point, segment_points):
    """Calculate minimum distance from point to line segment defined by two points."""
    p = np.array(point)
    a = np.array(segment_points[0])
    b = np.array(segment_points[1])

    ab = b - a
    ap = p - a

    ab_dot_ab = np.dot(ab, ab)

    if ab_dot_ab < 1e-9:
        return np.linalg.norm(p - a)

    t = np.dot(ap, ab) / ab_dot_ab
    t_clipped = np.clip(t, 0, 1)

    # Find the nearest point on the segment itself
    nearest_point_on_segment = a + t_clipped * ab

    distance = np.linalg.norm(p - nearest_point_on_segment)
    return distance

def match_app_name_and_usage(detections_for_layout, image_shape, config):
    """
    Matches app names, usages, and icons based on search lines and layout.
    Falls back to Euclidean distance matching if no search lines are available.
    (Adapted from User Input to use dict format and config)
    Returns:
        matched_data: list of dicts {app_name, app_usage, app_icon}
        debug_info: dict containing intermediate results for visualization/logging
    """
    debug_info = {}
    app_name_class = config.get('app_name_class_name', 'app_name')
    app_usage_class = config.get('app_usage_class_name', 'app_usage')
    app_icon_class = config.get('app_icon_class_name', 'app_icon')
    id_class = config.get('id_class_name', 'id')

    app_names = [d for d in detections_for_layout if d['label'] == app_name_class]
    app_usages = [d for d in detections_for_layout if d['label'] == app_usage_class]
    app_icons = [d for d in detections_for_layout if d['label'] == app_icon_class]
    ids = [d for d in detections_for_layout if d['label'] == id_class]

    matched_data = []
    unmatched_app_names_keys = { (d['x'],d['y'],d['w'],d['h']) for d in app_names }
    unmatched_app_usages_keys = { (d['x'],d['y'],d['w'],d['h']) for d in app_usages }
    unmatched_app_icons_keys = { (d['x'],d['y'],d['w'],d['h']) for d in app_icons }

    icon_centers = [(i["x"] + i["w"] // 2, i["y"] + i["h"] // 2) for i in app_icons]
    image_height, image_width = image_shape[0], image_shape[1]

    icon_search_lines = []
    reference_line = None

    min_ref_line_icons = config.get('ref_line_min_icons', 3) # Default 3 if not in config
    ref_line_max_gap = config.get('ref_line_max_gap', 60)   # Default 60 if not in config (but should be increased)
    ref_line_max_angle = config.get('ref_line_max_angle', 25) # Default 25 if not in config

    logger.debug(f"Checking icon count for reference line: Found {len(app_icons)}, Minimum required: {min_ref_line_icons}")

    if len(app_icons) >= min_ref_line_icons:
        reference_line = find_reference_line(
            icon_centers, image_shape,
            max_gap=ref_line_max_gap,
            max_angle=ref_line_max_angle,
            min_icons=min_ref_line_icons
        )

    debug_info["reference_line"] = reference_line

    if reference_line:
        search_line_tolerance = config.get('search_line_tolerance', 5)
        USAGE_SEARCH_DISTANCE_THRESHOLD = config.get('usage_search_distance', 200)
        NAME_SEARCH_DISTANCE_THRESHOLD = config.get('name_search_distance', 120)

        ref_orientation = reference_line[0]
        debug_info["reference_orientation"] = ref_orientation
        all_icon_bboxes_for_search = [{"x": i["x"], "y": i["y"], "w": i["w"], "h": i["h"]} for i in app_icons]

        for icon in app_icons:
            center = (icon["x"] + icon["w"] // 2, icon["y"] + icon["h"] // 2)
            search_line = create_search_line(
                center, reference_line, all_icon_bboxes_for_search,
                image_width, image_height, tolerance=search_line_tolerance
            )
            if search_line:
                icon_search_lines.append({'icon': icon, 'search_line': search_line})
        debug_info["icon_search_lines"] = icon_search_lines

        # --- Matching using Search Lines ---
        # 1: Match app usages to closest icon's search line
        usage_matches = {} ; min_distances_usage = {}
        for usage_det in app_usages:
            usage_key = (usage_det['x'], usage_det['y'], usage_det['w'], usage_det['h'])
            usage_center = get_centroid(usage_det); min_dist = USAGE_SEARCH_DISTANCE_THRESHOLD; best_icon_key = None
            for item in icon_search_lines:
                icon = item['icon']; search_line_tuple = item['search_line']
                line_type, start, end = search_line_tuple
                distance = distance_to_segment(usage_center, (start, end))
                if distance < min_dist:
                    icon_key = (icon['x'], icon['y'], icon['w'], icon['h']); min_dist = distance; best_icon_key = icon_key
            if best_icon_key:
                existing_match = False
                for u_key, i_key in usage_matches.items():
                    if i_key == best_icon_key and min_distances_usage.get(u_key, float('inf')) < min_dist: existing_match = True; break
                if not existing_match:
                    keys_to_remove = [u_key for u_key, i_key in usage_matches.items() if i_key == best_icon_key]
                    for k in keys_to_remove: usage_matches.pop(k, None); min_distances_usage.pop(k, None)
                    usage_matches[usage_key] = best_icon_key; min_distances_usage[usage_key] = min_dist

        # Create initial matched_data based on usage-icon pairs
        icons_matched_to_usage = set()
        for usage_key, icon_key in usage_matches.items():
            usage_det = next((u for u in app_usages if (u['x'],u['y'],u['w'],u['h']) == usage_key), None)
            icon_det = next((i for i in app_icons if (i['x'],i['y'],i['w'],i['h']) == icon_key), None)
            if usage_det and icon_det:
                matched_data.append({"app_usage": usage_det, "app_icon": icon_det, "app_name": None})
                unmatched_app_usages_keys.discard(usage_key); unmatched_app_icons_keys.discard(icon_key)
                icons_matched_to_usage.add(icon_key)

        # 2: Match app names to icons that already have a usage matched
        name_matches = {} ; min_distances_name = {}
        for name_det in app_names:
            name_key = (name_det['x'], name_det['y'], name_det['w'], name_det['h'])
            name_center = get_centroid(name_det); min_dist = NAME_SEARCH_DISTANCE_THRESHOLD; best_icon_key = None
            for item in icon_search_lines:
                icon = item['icon']; icon_key = (icon['x'], icon['y'], icon['w'], icon['h'])
                if icon_key not in icons_matched_to_usage: continue
                search_line_tuple = item['search_line']; line_type, start, end = search_line_tuple
                distance = distance_to_segment(name_center, (start, end))
                if distance < min_dist: min_dist = distance; best_icon_key = icon_key
            if best_icon_key:
                existing_match = False
                for n_key, i_key in name_matches.items():
                    if i_key == best_icon_key and min_distances_name.get(n_key, float('inf')) < min_dist: existing_match = True; break
                if not existing_match:
                    keys_to_remove = [n_key for n_key, i_key in name_matches.items() if i_key == best_icon_key]
                    for k in keys_to_remove: name_matches.pop(k, None); min_distances_name.pop(k, None)
                    name_matches[name_key] = best_icon_key; min_distances_name[name_key] = min_dist

        # Add matched names to the matched_data structure
        for name_key, icon_key in name_matches.items():
            name_det = next((n for n in app_names if (n['x'],n['y'],n['w'],n['h']) == name_key), None)
            if not name_det: continue
            for match_entry in matched_data:
                if match_entry.get('app_icon'):
                    entry_icon_key = (match_entry['app_icon']['x'], match_entry['app_icon']['y'], match_entry['app_icon']['w'], match_entry['app_icon']['h'])
                    if entry_icon_key == icon_key:
                        if match_entry['app_name'] is None:
                            match_entry['app_name'] = name_det; unmatched_app_names_keys.discard(name_key); break

    # Fallback: If no search lines
    if not icon_search_lines and app_names and app_usages:
        logger.info("No search lines available. Falling back to Euclidean distance matching between names and usages.")
        usage_to_name_map = {} ; min_usage_name_dist = {}
        for usage_det in app_usages:
            usage_key = (usage_det['x'], usage_det['y'], usage_det['w'], usage_det['h'])
            usage_center = get_centroid(usage_det); best_name_key = None; min_dist_sq = float('inf')
            for name_det in app_names:
                name_key = (name_det['x'], name_det['y'], name_det['w'], name_det['h'])
                name_center = get_centroid(name_det)
                dist_sq = (usage_center[0] - name_center[0])**2 + (usage_center[1] - name_center[1])**2
                if dist_sq < min_dist_sq:
                    is_name_closer_elsewhere = False
                    for other_u_key, assigned_n_key in usage_to_name_map.items():
                         if assigned_n_key == name_key and min_usage_name_dist.get(other_u_key, float('inf')) < dist_sq: is_name_closer_elsewhere = True; break
                    if not is_name_closer_elsewhere: min_dist_sq = dist_sq; best_name_key = name_key
            if best_name_key:
                keys_to_remove = [u_key for u_key, n_key in usage_to_name_map.items() if n_key == best_name_key]
                for k in keys_to_remove: usage_to_name_map.pop(k, None); min_usage_name_dist.pop(k, None)
                usage_to_name_map[usage_key] = best_name_key; min_usage_name_dist[usage_key] = min_dist_sq

        # Create matched_data entries for fallback matches
        for usage_key, name_key in usage_to_name_map.items():
            usage_det = next((u for u in app_usages if (u['x'],u['y'],u['w'],u['h']) == usage_key), None)
            name_det = next((n for n in app_names if (n['x'],n['y'],n['w'],n['h']) == name_key), None)
            if usage_det and name_det:
                matched_data.append({"app_usage": usage_det, "app_name": name_det, "app_icon": None})
                unmatched_app_usages_keys.discard(usage_key); unmatched_app_names_keys.discard(name_key)

    # Final population of debug_info with actual unmatched items
    debug_info["unmatched_app_names"] = [n for n in app_names if (n['x'], n['y'], n['w'], n['h']) in unmatched_app_names_keys]
    debug_info["unmatched_app_usages"] = [u for u in app_usages if (u['x'], u['y'], u['w'], u['h']) in unmatched_app_usages_keys]
    debug_info["unmatched_app_icons"] = [i for i in app_icons if (i['x'], i['y'], i['w'], i['h']) in unmatched_app_icons_keys]
    debug_info["ids"] = ids

    logger.info(f"Layout analysis complete. Matched entries: {len(matched_data)}")
    return matched_data, debug_info
