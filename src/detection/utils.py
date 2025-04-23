import numpy as np
from typing import Tuple, List, Dict, Optional

def create_search_line(center, ref_orientation, reference_line, all_icon_bboxes, image_width, image_height, tolerance=5):
    """EXACT original working implementation"""
    try:
        _, ref_start, ref_end = reference_line
        ref_dx = ref_end[0] - ref_start[0]
        ref_dy = ref_end[1] - ref_start[1]
        ref_slope = ref_dy / ref_dx if ref_dx != 0 else float('inf')

        # Slope calculations
        if ref_slope == 0:
            perp_slope = float('inf')
        elif ref_slope == float('inf'):
            perp_slope = 0
        else:
            perp_slope = -1 / ref_slope

        x, y = center
        dx, dy = (1, perp_slope) if not np.isinf(perp_slope) else (0, 1)
        
        # Direction normalization
        magnitude = np.hypot(dx, dy)
        dx /= magnitude
        dy /= magnitude

        end_x = image_width if dx > 0 else 0
        end_y = image_height if dy > 0 else 0

        # Collision detection logic
        closest_t = float('inf')
        for icon in all_icon_bboxes:
            ix, iy, iw, ih = icon['x'], icon['y'], icon['w'], icon['h']
            
            # Vertical checks
            if dx != 0:
                # Left edge
                t_left = (ix - x) / dx
                if t_left > 0:
                    y_left = y + dy * t_left
                    if iy-tolerance <= y_left <= iy+ih+tolerance and t_left < closest_t:
                        closest_t = t_left
                        end_x = ix
                        end_y = y_left
                
                # Right edge
                t_right = (ix + iw - x) / dx
                if t_right > 0:
                    y_right = y + dy * t_right
                    if iy-tolerance <= y_right <= iy+ih+tolerance and t_right < closest_t:
                        closest_t = t_right
                        end_x = ix + iw
                        end_y = y_right

            # Horizontal checks
            if dy != 0:
                # Top edge
                t_top = (iy - y) / dy
                if t_top > 0:
                    x_top = x + dx * t_top
                    if ix-tolerance <= x_top <= ix+iw+tolerance and t_top < closest_t:
                        closest_t = t_top
                        end_x = x_top
                        end_y = iy
                
                # Bottom edge
                t_bottom = (iy + ih - y) / dy
                if t_bottom > 0:
                    x_bottom = x + dx * t_bottom
                    if ix-tolerance <= x_bottom <= ix+iw+tolerance and t_bottom < closest_t:
                        closest_t = t_bottom
                        end_x = x_bottom
                        end_y = iy + ih

        return (ref_orientation, (int(x), int(y)), (int(end_x), int(end_y)))

    except Exception as e:
        # Fallback behavior
        return (ref_orientation, (int(x), int(y)), (int(x)+10, int(y)+10))

def distance_to_segment(point, segment):
    """EXACT original distance calculation"""
    px, py = point
    (x1, y1), (x2, y2) = segment
    
    dx = x2 - x1
    dy = y2 - y1
    t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2 + 1e-6)
    t = max(0, min(1, t))
    
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return ((px - nearest_x)**2 + (py - nearest_y)**2)**0.5

def line_intersection(line1, line2):
    """EXACT original intersection code"""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
    if denom == 0:
        return None
        
    ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom
    if ua < 0 or ua > 1:
        return None
        
    ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom
    if ub < 0 or ub > 1:
        return None
        
    x = x1 + ua*(x2 - x1)
    y = y1 + ua*(y2 - y1)
    return (int(x), int(y))