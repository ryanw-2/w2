import cv2
import numpy as np
import math
import csv
import json
from skimage.morphology import skeletonize
from simplification.cutil import simplify_coords_vw

def snap_to_grid(pt, grid_size=5):
    return (round(pt[0] / grid_size) * grid_size, round(pt[1] / grid_size) * grid_size)

def angle_between(p1, p2, p3):
    """
    Returns angle (in degrees) between vectors p1→p2 and p2→p3.
    Measures how much the path turns at point p2.
    """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    angle_rad = math.acos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    return math.degrees(angle_rad)

def connect_nearby_paths(paths, max_distance=15):
    """
    Connect paths that have endpoints close to each other.
    Returns connected paths.
    """
    if len(paths) <= 1:
        return paths
    
    connected_paths = []
    used_indices = set()
    
    for i, path1 in enumerate(paths):
        if i in used_indices:
            continue
            
        current_path = path1.copy()
        used_indices.add(i)
        
        # Keep trying to extend the current path
        extended = True
        while extended:
            extended = False
            current_start = current_path[0]
            current_end = current_path[-1]
            
            for j, path2 in enumerate(paths):
                if j in used_indices:
                    continue
                    
                path2_start = path2[0]
                path2_end = path2[-1]
                
                # Check all possible connections
                distances = [
                    np.linalg.norm(np.array(current_end) - np.array(path2_start)),
                    np.linalg.norm(np.array(current_end) - np.array(path2_end)),
                    np.linalg.norm(np.array(current_start) - np.array(path2_start)),
                    np.linalg.norm(np.array(current_start) - np.array(path2_end))
                ]
                
                connection_types = ['end_to_start', 'end_to_end', 'start_to_start', 'start_to_end']
                
                # Find the best connection
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]
                best_type = connection_types[best_idx]
                
                # Get the appropriate path based on connection type
                if best_type in ['end_to_end', 'start_to_start']:
                    best_path = path2[::-1]
                else:
                    best_path = path2
                
                if best_dist <= max_distance:
                    if best_type.startswith('end_'):
                        current_path.extend(best_path[1:])  # Skip first point to avoid duplication
                    else:  # start_to_*
                        current_path = best_path[:-1] + current_path  # Skip last point to avoid duplication
                    
                    used_indices.add(j)
                    extended = True
                    break
        
        if len(current_path) >= 2:
            connected_paths.append(current_path)
    
    return connected_paths

def segment_path(points, angle_threshold=10):
    """
    Splits a path into 'straight' or 'curved' segments based on local angular deviation.
    Fixed logic to properly handle segment transitions.
    """
    if len(points) < 3:
        return [('straight', points)]

    segments = []
    current_segment = [points[0]]
    current_type = 'straight'

    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        angle = angle_between(a, b, c)
        angle_deviation = abs(angle - 180)
        
        # Determine if this point indicates a turn
        is_turn = angle_deviation > angle_threshold
        new_type = 'curved' if is_turn else 'straight'
        
        current_segment.append(b)
        
        # If segment type changes, start a new segment
        if new_type != current_type and len(current_segment) > 1:
            segments.append((current_type, current_segment))
            current_segment = [b]  # Start new segment with the turning point
            current_type = new_type
    
    # Add the last point and final segment
    current_segment.append(points[-1])
    if len(current_segment) >= 2:
        segments.append((current_type, current_segment))
    
    return segments

def remove_duplicate_points(points, tolerance=2):
    """Remove consecutive duplicate points that are very close."""
    if len(points) <= 1:
        return points
    
    filtered = [points[0]]
    for i in range(1, len(points)):
        dist = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
        if dist > tolerance:
            filtered.append(points[i])
    
    return filtered

def extract_geometry_from_sketch(image_path: str, epsilon=3.0, visualize_steps: bool = True):
    input_img = cv2.imread(image_path)
    if input_img is None:
        print("Error: Could not read image at file path")
        return []

    input_img = cv2.flip(input_img, 0)
    
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    binary_img[binary_img == 255] = 1
    skeleton_img = skeletonize(binary_img)
    skeleton_img = (skeleton_img.astype(np.uint8)) * 255

    contours, _ = cv2.findContours(
        skeleton_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    print(f"Found {len(contours)} initial contours.")

    if visualize_steps:
        cv2.imshow("Step 1. Contours", skeleton_img)
        cv2.waitKey(0)

    # First pass: extract and simplify all contours
    initial_paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        raw = contour.squeeze().tolist()
        simplified = simplify_coords_vw(raw, epsilon)
        
        # Remove duplicate points and snap to grid
        cleaned = remove_duplicate_points(simplified)
        snapped = [snap_to_grid(pt, grid_size=5) for pt in cleaned]
        
        if len(snapped) >= 2:
            initial_paths.append(snapped)

    print(f"Generated {len(initial_paths)} initial paths.")

    # Second pass: connect nearby paths
    connected_paths = connect_nearby_paths(initial_paths, max_distance=15)
    print(f"Connected into {len(connected_paths)} merged paths.")

    # Third pass: segment the connected paths
    refined_paths = []
    path_types = []

    for path in connected_paths:
        if len(path) < 2:
            continue
            
        segmented = segment_path(path, angle_threshold=10)
        
        for seg_type, seg_points in segmented:
            if len(seg_points) >= 2:
                refined_paths.append(seg_points)
                path_types.append(seg_type)

    print(f"Produced {len(refined_paths)} final segments.")

    if visualize_steps:
        black_mask = np.zeros(input_img.shape[:2], dtype="uint8")
        vis_img = input_img.copy()
        
        for i, path in enumerate(refined_paths):
            color = (0, 255, 0) if path_types[i] == 'straight' else (0, 0, 255)
            np_path = np.array(path, dtype=np.int32)
            cv2.polylines(vis_img, [np_path], isClosed=False, color=color, thickness=2)
            cv2.polylines(black_mask, [np_path], isClosed=False, color=(255,255,255), thickness=2)
            
            # Mark endpoints for debugging
            cv2.circle(vis_img, tuple(path[0]), 3, (255, 255, 255), -1)
            cv2.circle(vis_img, tuple(path[-1]), 3, (0, 0, 0), -1)
        
        cv2.imshow("Refined Geometry", vis_img)
        cv2.imshow("Path Cleaned", black_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return refined_paths, path_types


def save_paths_to_csv(paths, output_path):
    if not paths:
        print("No lines were able to be saved.")
        return    
    
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        path_splits: list[int] = [0]
        i = 0
        for path_id, path in enumerate(paths):
            for point in path:
                x = int(point[0])
                y = int(point[1])
                w.writerow([path_id, x, y])
                i += 1
            path_splits.append(i)
        w.writerow(path_splits)
    print(f"Successfully saved path polyline data to {output_path}")

if __name__ == "__main__":
    file_path = "./testPlanSketch2.jpg"
    cleaned_paths, path_kinds = extract_geometry_from_sketch(file_path, epsilon=2.0, visualize_steps=True)

    if cleaned_paths:
        save_paths_to_csv(cleaned_paths, 'paths_cleaned.csv')