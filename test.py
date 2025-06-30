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
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    angle = math.acos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    return math.degrees(angle)

def segment_path(points, angle_threshold=10):
    """
    Split a list of points into subpaths labeled as 'straight' or 'curved'
    """
    if len(points) < 3:
        return [('straight', points)]

    segments = []
    buffer = [points[0], points[1]]
    mode = 'straight'

    for i in range(2, len(points)):
        a, b, c = points[i - 2], points[i - 1], points[i]
        angle = angle_between(a, b, c)
        if abs(angle - 180) < angle_threshold or angle < angle_threshold:
            if mode != 'straight':
                segments.append((mode, buffer))
                buffer = [b]
                mode = 'straight'
        else:
            if mode != 'curved':
                segments.append((mode, buffer))
                buffer = [b]
                mode = 'curved'
        buffer.append(c)

    segments.append((mode, buffer))
    return segments

def extract_geometry_from_sketch(image_path: str, epsilon=2.0, visualize_steps: bool = True):
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

    refined_paths = []
    path_types = []

    for contour in contours:
        if len(contour) < 3:
            continue
        
        raw = contour.squeeze().tolist()

        simplified = simplify_coords_vw(raw, epsilon)
        segmented = segment_path(simplified, angle_threshold=10)

        for seg_type, seg_points in segmented:
            snapped = [snap_to_grid(pt, grid_size=5) for pt in seg_points]
            if len(snapped) >= 2:
                refined_paths.append(snapped)
                path_types.append(seg_type)

    print(f"Produced {len(refined_paths)} cleaned segments.")

    if visualize_steps:
        vis_img = input_img.copy()
        for i, path in enumerate(refined_paths):
            color = (0, 255, 0) if path_types[i] == 'straight' else (0, 0, 255)
            np_path = np.array(path, dtype=np.int32)
            cv2.polylines(vis_img, [np_path], isClosed=False, color=color, thickness=1)
        cv2.imshow("Refined Geometry", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return refined_paths, path_types

def save_paths_to_csv(paths, types, output_path):
    if not paths:
        print("No lines were able to be saved.")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for path_id, (path, kind) in enumerate(zip(paths, types)):
            for pt in path:
                writer.writerow([path_id, pt[0], pt[1], kind])
    print(f"Saved processed paths to {output_path}")

def save_paths_to_csv_prim(paths, output_path):
    if not paths:
        print("No lines were able to be saved.")
        return

    with open(output_path, 'w', newline='') as f:
        path_splits: list[int] = [0]
        writer = csv.writer(f)
        i = 0
        for path_id, path in enumerate(paths):
            for pt in path:
                writer.writerow([path_id, pt[0], pt[1]])
                i += 1
            path_splits.append(i)
        writer.writerow(path_splits)
    print(f"Saved processed paths to {output_path}")

if __name__ == "__main__":
    file_path = "./testPlanLine-Cleaned.jpg"
    cleaned_paths, path_kinds = extract_geometry_from_sketch(file_path, epsilon=2.0, visualize_steps=True)

    if cleaned_paths:
        save_paths_to_csv_prim(cleaned_paths, 'paths_cleaned.csv')