import cv2
import numpy as np
import math
import csv
from skimage.morphology import skeletonize
import json
"""
mysql username: root
mysql passwd:    abcd
"""

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_red = np.array([0, 100, 100])
upper_red = np.array([20, 255, 255])
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])


def callback(input):
    pass

def extract_geometry_from_sketch(
    image_path: str, epsilon_multiplier: float = 0.001, visualize_steps: bool = True
):
    """
    Takes the path to a floor plan sketch and returns a list of polylines
    representing all walls (straight and curved).

    This pipeline can handle any shape by:
    1. Creating a clean, one-pixel-wide skeleton of the drawing.
    2. Finding all continuous contours in the skeleton.
    3. Approximating each contour with a series of connected line segments.

    Args:
        image_path (str): The path to the input image file.
        epsilon_multiplier (float): Controls the precision of the curve approximation.
                                  Smaller values give more detail but more points.
        visualize_steps (bool): If True, displays intermediate images for debugging.

    Returns:
        list: A list of "paths," where each path is a list of points (x, y).
    """
    input_img = cv2.imread(image_path)
    if input_img is None:
        print("Error: Could not read image at file path")
        return
    
    input_img = cv2.flip(input_img, 0)
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    binary_img[binary_img == 255] = 1
    skeleton_img = skeletonize(binary_img)
    skeleton_img = skeleton_img.astype(np.uint8) * 255

    # if visualize_steps:
    #     cv2.imshow("1. Skeletonized Image", skeleton_img)
    #     cv2.waitKey(0)

    ##
    contours, _ = cv2.findContours(
        skeleton_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    print(f"Found {len(contours)} initial contours.")

    all_paths = []
    min_contour_length = 3
    for contour in contours:
        if len(contour) < min_contour_length:
            continue
        epsilon = epsilon_multiplier * cv2.arcLength(contour, True)
        approximated_path = cv2.approxPolyDP(contour, epsilon, False)
        all_paths.append(approximated_path.squeeze().tolist())

    print(f"Processed contours into {len(all_paths)} paths.")

    if visualize_steps:
        vis_img = input_img.copy()
        for path in all_paths:
            np_path = np.array(path, dtype=np.int32)
            cv2.polylines(
                vis_img, [np_path], isClosed=False, color=(0, 255, 0), thickness=1
            )
        cv2.imshow("2. Approximated Geometry", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return all_paths

def save_lines_to_json(paths, output_path):
    if not paths:
        print("No lines were able to be saved.")
        return
    
    output_data = {"paths": []}
    for path in paths:
        # Ensure path is a list of lists, even for straight lines
        if isinstance(path[0], int): # A single point was found
            continue
        
        formatted_path = [{"x": int(point[0]), "y": int(point[1])} for point in path]
        output_data["paths"].append(formatted_path)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Successfully saved line data to {output_path}")

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
    wall_paths = extract_geometry_from_sketch(file_path, visualize_steps=True)

    if wall_paths:
        # save_lines_to_json(wall_paths, 'level.json')
        save_paths_to_csv(wall_paths, 'paths.csv')