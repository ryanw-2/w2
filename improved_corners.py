import cv2
import numpy as np
import math
import csv
import json
from skimage.morphology import skeletonize
from simplification.cutil import simplify_coords_vw
from collections import defaultdict
from scipy.spatial.distance import cdist

def snap_to_grid(pt, grid_size=5):
    return (round(pt[0] / grid_size) * grid_size, round(pt[1] / grid_size) * grid_size)

def angle_between(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    angle = math.acos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    return math.degrees(angle)

def find_path_endpoints(path):
    """Find the endpoints of a path"""
    if len(path) < 2:
        return []
    return [path[0], path[-1]]

def paths_can_connect(path1, path2, max_distance=15):
    """Check if two paths can be connected based on endpoint proximity"""
    endpoints1 = find_path_endpoints(path1)
    endpoints2 = find_path_endpoints(path2)
    
    min_dist = float('inf')
    best_connection = None
    
    for i, ep1 in enumerate(endpoints1):
        for j, ep2 in enumerate(endpoints2):
            dist = np.linalg.norm(np.array(ep1) - np.array(ep2))
            if dist < min_dist:
                min_dist = dist
                best_connection = (i, j, dist)
    
    return min_dist <= max_distance, best_connection

def connect_paths(path1, path2, connection_info):
    """Connect two paths based on connection information"""
    i, j, dist = connection_info
    
    # Determine how to connect the paths
    if i == 0 and j == 0:  # start to start
        return list(reversed(path1)) + path2
    elif i == 0 and j == 1:  # start to end
        return list(reversed(path1)) + list(reversed(path2))
    elif i == 1 and j == 0:  # end to start
        return path1 + path2
    else:  # end to end
        return path1 + list(reversed(path2))

def merge_connected_paths(paths, max_distance=15):
    """Merge paths that can be connected"""
    if not paths:
        return []
    
    # Create a graph of path connections
    path_graph = defaultdict(list)
    used_paths = set()
    
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            can_connect, connection_info = paths_can_connect(paths[i], paths[j], max_distance)
            if can_connect:
                path_graph[i].append((j, connection_info))
                path_graph[j].append((i, connection_info))
    
    merged_paths = []
    
    # Merge connected components
    for start_idx in range(len(paths)):
        if start_idx in used_paths:
            continue
            
        # Build connected component starting from this path
        current_path = paths[start_idx].copy()
        used_paths.add(start_idx)
        
        # Keep extending the path
        changed = True
        while changed:
            changed = False
            current_endpoints = find_path_endpoints(current_path)
            
            # Look for paths that can extend current path
            for path_idx in range(len(paths)):
                if path_idx in used_paths:
                    continue
                    
                candidate_path = paths[path_idx]
                can_connect, connection_info = paths_can_connect(current_path, candidate_path, max_distance)
                
                if can_connect:
                    current_path = connect_paths(current_path, candidate_path, connection_info)
                    used_paths.add(path_idx)
                    changed = True
                    break
        
        if len(current_path) >= 2:
            merged_paths.append(current_path)
    
    return merged_paths

def improve_corner_snapping(points, corners, snap_distance=10):
    """Improved corner snapping that avoids multiple points snapping to same corner"""
    if not corners:
        return points
    
    corners_array = np.array(corners)
    points_array = np.array(points)
    
    # Calculate distances from all points to all corners
    distances = cdist(points_array, corners_array)
    
    # For each corner, find the closest point within snap distance
    snapped_points = points.copy()
    used_corners = set()
    
    for corner_idx in range(len(corners)):
        if corner_idx in used_corners:
            continue
            
        # Find closest point to this corner
        point_distances = distances[:, corner_idx]
        closest_point_idx = np.argmin(point_distances)
        
        if point_distances[closest_point_idx] <= snap_distance:
            snapped_points[closest_point_idx] = corners[corner_idx]
            used_corners.add(corner_idx)
    
    return snapped_points

def segment_path_by_angle(points, angle_threshold=10):
    """Improved path segmentation that maintains connectivity"""
    if len(points) < 3:
        return [('straight', points)]

    segments = []
    current_segment = [points[0], points[1]]
    
    for i in range(2, len(points)):
        a, b, c = points[i - 2], points[i - 1], points[i]
        angle = angle_between(a, b, c)
        
        # Check if this is a corner (significant deviation from straight)
        if abs(angle - 180) > angle_threshold:
            # End current segment and start new one
            if len(current_segment) >= 2:
                segments.append(('straight', current_segment))
            current_segment = [b, c]  # Start new segment with the corner point
        else:
            current_segment.append(c)
    
    # Add the last segment
    if len(current_segment) >= 2:
        segments.append(('straight', current_segment))
    
    return segments

def remove_duplicate_points(points, tolerance=1e-6):
    """Remove duplicate consecutive points"""
    if not points:
        return points
    
    filtered = [points[0]]
    for i in range(1, len(points)):
        if np.linalg.norm(np.array(points[i]) - np.array(points[i-1])) > tolerance:
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

    # Corner detection using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.1, minDistance=10)
    corner_points = []
    if corners is not None:
        corners = corners.squeeze().astype(int)
        corner_points = [tuple(c) for c in corners]

    # Process individual contours
    raw_paths = []
    for contour in contours:
        if len(contour) < 3:
            continue

        raw = contour.squeeze().tolist()
        simplified = simplify_coords_vw(raw, epsilon)
        snapped = [snap_to_grid(pt, grid_size=5) for pt in simplified]
        
        # Remove duplicates
        snapped = remove_duplicate_points(snapped)
        
        if len(snapped) >= 2:
            raw_paths.append(snapped)

    print(f"Generated {len(raw_paths)} raw paths before connection.")

    # Connect related paths
    connected_paths = merge_connected_paths(raw_paths, max_distance=15)
    print(f"Connected into {len(connected_paths)} merged paths.")

    # Apply corner snapping to connected paths
    refined_paths = []
    for path in connected_paths:
        # Improve corner snapping
        snapped_path = improve_corner_snapping(path, corner_points, snap_distance=10)
        
        # Segment by angle after connection
        segments = segment_path_by_angle(snapped_path, angle_threshold=10)
        
        for _, segment in segments:
            if len(segment) >= 2:
                refined_paths.append(segment)

    print(f"Produced {len(refined_paths)} final line segments.")

    if visualize_steps:
        vis_img = input_img.copy()
        
        # Draw different path types in different colors
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, path in enumerate(refined_paths):
            color = colors[i % len(colors)]
            np_path = np.array(path, dtype=np.int32)
            cv2.polylines(vis_img, [np_path], isClosed=False, color=color, thickness=2)
            
            # Mark endpoints
            cv2.circle(vis_img, tuple(path[0]), radius=4, color=(255, 255, 255), thickness=2)
            cv2.circle(vis_img, tuple(path[-1]), radius=4, color=(0, 0, 0), thickness=2)

        # Draw detected corners
        for pt in corner_points:
            cv2.circle(vis_img, pt, radius=3, color=(0, 0, 255), thickness=-1)

        cv2.imshow("Connected and Refined Geometry", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return refined_paths

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
    file_path = "testPlanSketch2.jpg"
    cleaned_paths = extract_geometry_from_sketch(file_path, epsilon=3.0, visualize_steps=True)

    if cleaned_paths:
        save_paths_to_csv(cleaned_paths, 'connected_paths.csv')