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

def segment_path_by_angle(points, angle_threshold=10):
    if len(points) < 3:
        return [('straight', points)]

    segments = []
    buffer = [points[0], points[1]]

    for i in range(2, len(points)):
        a, b, c = points[i - 2], points[i - 1], points[i]
        angle = angle_between(a, b, c)
        if abs(angle - 180) < angle_threshold:
            buffer.append(c)
        else:
            segments.append(('straight', buffer))
            buffer = [b, c]

    segments.append(('straight', buffer))
    return segments

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

    refined_paths = []

    # Corner detection using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.1, minDistance=10)
    corner_points = []
    if corners is not None:
        corners = corners.squeeze().astype(int)
        corner_points = [tuple(c) for c in corners]

    for contour in contours:
        if len(contour) < 3:
            continue

        raw = contour.squeeze().tolist()

        simplified = simplify_coords_vw(raw, epsilon)
        snapped = [snap_to_grid(pt, grid_size=5) for pt in simplified]

        # Snap to detected corner if nearby
        snapped_with_corners = []
        for pt in snapped:
            nearest = None
            min_dist = float('inf')
            for c in corner_points:
                dist = np.linalg.norm(np.array(pt) - np.array(c))
                if dist < min_dist:
                    min_dist = dist
                    nearest = c

            if nearest and min_dist < 10:
                snapped_with_corners.append(nearest)
            else:
                snapped_with_corners.append(pt)

        segments = segment_path_by_angle(snapped_with_corners, angle_threshold=10)

        for _, segment in segments:
            if len(segment) >= 2:
                refined_paths.append(segment)

    print(f"Produced {len(refined_paths)} cleaned line segments.")

    if visualize_steps:
        vis_img = input_img.copy()
        for path in refined_paths:
            np_path = np.array(path, dtype=np.int32)
            cv2.polylines(vis_img, [np_path], isClosed=False, color=(0, 255, 0), thickness=1)

        # Optionally display detected corners
        for pt in corner_points:
            cv2.circle(vis_img, pt, radius=3, color=(0, 0, 255), thickness=-1)

        cv2.imshow("Refined Geometry with Corners", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return refined_paths

def save_paths_to_csv(paths, output_path):
    if not paths:
        print("No lines were able to be saved.")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for path_id, path in enumerate(paths):
            for pt in path:
                writer.writerow([path_id, pt[0], pt[1]])
    print(f"Saved {len(paths)} paths to {output_path}")

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
    file_path = "sPoche.jpg"
    cleaned_paths = extract_geometry_from_sketch(file_path, epsilon=3.0, visualize_steps=True)

    if cleaned_paths:
        save_paths_to_csv_prim(cleaned_paths, 'corner_paths.csv')