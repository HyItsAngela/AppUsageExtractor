import numpy as np
from typing import List, Tuple, Optional
from .schemas import Detection

class GridProcessor:
  
    MAX_COLUMN_WIDTH = 50  
    MAX_GAP = 60           
    MAX_ANGLE = 25        
    
    @classmethod
    def detect_grid(cls, icon_centers: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Original grid detection logic"""
        if len(icon_centers) < 2:
            print("Not enough icons to determine grid")
            return (0, 0)

        centers = np.array(icon_centers)
        row_breaks = []
        
        y_sorted = centers[np.argsort(centers[:, 1])]
        for i in range(1, len(y_sorted)):
            if y_sorted[i, 1] - y_sorted[i-1, 1] > cls.MAX_COLUMN_WIDTH:
                row_breaks.append(i)
        
        rows = np.split(y_sorted, row_breaks)
        num_rows = len(rows)
        num_cols = max(len(row) for row in rows) if rows else 0
        
        print(f"Detected Grid: {num_rows} rows, {num_cols} columns")
        return num_rows, num_cols

    @classmethod
    def find_reference_line(
        cls,
        icon_centers: List[Tuple[int, int]],
        image_shape: Tuple[int, int],
        min_icons: int = 3
    ) -> Optional[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
        """Original reference line detection logic"""
        if len(icon_centers) < min_icons:
            return None

        num_rows, num_cols = cls.detect_grid(icon_centers)
        if num_rows == 0 or num_cols == 0:
            return None

        # Consistency scoring
        def line_consistency_score(points):
            if len(points) < 2:
                return 0
            
            spacings = [np.linalg.norm(points[i]-points[i-1]) for i in range(1,len(points))]
            directions = [points[i]-points[i-1] for i in range(1,len(points))]
            
            norm_directions = [d/np.linalg.norm(d) for d in directions if np.linalg.norm(d)!=0]
            angle_variance = sum(np.arccos(np.clip(np.dot(norm_directions[i],norm_directions[i-1]),-1,1)) 
                               for i in range(1,len(norm_directions)))
            
            spacing_std = np.std(spacings) if len(spacings)>1 else 0
            angle_std = angle_variance/max(1,len(norm_directions)-1)
            return 1/(1 + spacing_std + angle_std)

        # Spacing calculation
        def compute_avg_spacing(points, axis=0):
            sorted_points = sorted(points, key=lambda p: p[axis])
            gaps = [abs(sorted_points[i+1][axis]-sorted_points[i][axis]) 
                  for i in range(len(sorted_points)-1)]
            return np.mean(gaps) if gaps else float('inf')

        avg_col_spacing = compute_avg_spacing(icon_centers, 0)
        avg_row_spacing = compute_avg_spacing(icon_centers, 1)
        prioritize_columns = avg_col_spacing < avg_row_spacing

        # Line search logic
        def try_orientation(centers, orientation='vertical'):
            """Attempts to find a reference line in the specified orientation."""
            if orientation == 'vertical':
                centers = sorted(centers, key=lambda p: p[1])  # Sort by y-coordinate
                coord_idx = 0  # Use x-coordinates for vertical alignment
            else:
                centers = sorted(centers, key=lambda p: p[0])  # Sort by x-coordinate
                coord_idx = 1  # Use y-coordinates for horizontal alignment

            centers = np.array(centers)
            best_line = None
            best_score = 0

            for start_idx in range(len(centers) - min_icons + 1):
                current_line = [centers[start_idx]]
                remaining_centers = np.delete(centers, start_idx, axis=0)

                for candidate in remaining_centers:
                    if orientation == 'vertical':
                        if (abs(candidate[0] - current_line[-1][0]) < cls.MAX_GAP and 
                            candidate[1] > current_line[-1][1]):
                            current_line.append(candidate)
                            break
                    else:
                        if (abs(candidate[1] - current_line[-1][1]) < cls.MAX_GAP and 
                            candidate[0] > current_line[-1][0]):
                            current_line.append(candidate)
                            break
                else:  
                    continue

                mask = ~np.all(remaining_centers == current_line[-1], axis=1)
                remaining_centers = remaining_centers[mask]

                while True:
                    last_two = current_line[-2:]
                    direction = last_two[1] - last_two[0]
                    norm_direction = direction / np.linalg.norm(direction)

                    best_next = None
                    best_angle = float('inf')

                    candidates = remaining_centers.copy()
                    if orientation == 'vertical':
                        valid = candidates[:, 1] > current_line[-1][1]
                    else:
                        valid = candidates[:, 0] > current_line[-1][0]
                    
                    candidates = candidates[valid]
                    
                    if len(candidates) == 0:
                        break

                    vectors = candidates - current_line[-1]
                    norms = np.linalg.norm(vectors, axis=1)
                    norm_vectors = vectors / norms[:, None]
                    angles = np.arccos(np.clip(norm_vectors @ norm_direction, -1.0, 1.0))
                    
                    min_idx = np.argmin(angles)
                    if angles[min_idx] < np.radians(cls.MAX_ANGLE):
                        best_next = candidates[min_idx]
                        current_line.append(best_next)
                        mask = ~np.all(remaining_centers == best_next, axis=1)
                        remaining_centers = remaining_centers[mask]
                    else:
                        break

                if len(current_line) >= min_icons:
                    line_score = len(current_line) * line_consistency_score(current_line)
                    if line_score > best_score:
                        best_score = line_score
                        best_line = current_line

            return best_line

        # Priority handling
        if prioritize_columns:
            best_line = try_orientation(icon_centers, 'vertical') or try_orientation(icon_centers, 'horizontal')
        else:
            best_line = try_orientation(icon_centers, 'horizontal') or try_orientation(icon_centers, 'vertical')

        if not best_line or len(best_line) < min_icons:
            return None

        # Line fitting
        x = [p[0] for p in best_line]
        y = [p[1] for p in best_line]
        if np.std(x) < np.std(y):  # Vertical
            A = np.vstack([y, np.ones(len(y))]).T
            m, c = np.linalg.lstsq(A, x, rcond=None)[0]
            start = (int(m*0 + c), 0)
            end = (int(m*image_shape[0] + c), image_shape[0])
            orientation = "vertical"
        else:  # Horizontal
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            start = (0, int(m*0 + c))
            end = (image_shape[1], int(m*image_shape[1] + c))
            orientation = "horizontal"

        # Clipping to image bounds
        start = (
            np.clip(start[0], 0, image_shape[1]),
            np.clip(start[1], 0, image_shape[0])
        )
        end = (
            np.clip(end[0], 0, image_shape[1]),
            np.clip(end[1], 0, image_shape[0])
        )

        return (orientation, start, end)