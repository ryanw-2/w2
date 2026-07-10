import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import KDTree
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import defaultdict
import math

def preprocess_image(img):
    """Enhanced preprocessing for better corner detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive thresholding for better binary conversion
    binary = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return gray, cleaned

def detect_corners_enhanced(gray_img, binary_img, max_corners=100, quality_level=0.01, min_distance=10):
    """Enhanced corner detection using multiple methods"""
    corners = []
    
    # Method 1: Harris corner detection
    harris_corners = cv2.goodFeaturesToTrack(
        gray_img, max_corners, quality_level, min_distance
    )
    if harris_corners is not None:
        corners.extend([tuple(pt[0].astype(int)) for pt in harris_corners])
    
    # Method 2: Detect corners from contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Approximate contour to reduce points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        for point in approx:
            corners.append(tuple(point[0]))
    
    # Remove duplicate corners
    if corners:
        corners = list(set(corners))
        # Filter corners that are too close to each other
        filtered_corners = []
        for corner in corners:
            if not any(np.linalg.norm(np.array(corner) - np.array(fc)) < min_distance 
                      for fc in filtered_corners):
                filtered_corners.append(corner)
        corners = filtered_corners
    
    return corners

def create_skeleton(binary_img):
    """Create skeleton of the binary image"""
    # Convert to binary (0 and 1)
    binary_normalized = binary_img.copy()
    binary_normalized[binary_normalized == 255] = 1
    
    # Skeletonize
    skeleton = skeletonize(binary_normalized)
    skeleton = (skeleton * 255).astype(np.uint8)
    
    return skeleton

def sample_line_points(start, end, num_points=50):
    """Sample points along a line between two corners"""
    x_vals = np.linspace(start[0], end[0], num_points)
    y_vals = np.linspace(start[1], end[1], num_points)
    return list(zip(x_vals.astype(int), y_vals.astype(int)))

def check_line_support(corner_a, corner_b, skeleton_img, min_support_ratio=0.4):
    """Check if a line between corners has sufficient support from skeleton"""
    line_points = sample_line_points(corner_a, corner_b, 30)
    
    supported_points = 0
    total_points = len(line_points)
    
    # Check each point along the line
    for x, y in line_points:
        if 0 <= x < skeleton_img.shape[1] and 0 <= y < skeleton_img.shape[0]:
            # Check in a small neighborhood around the point
            neighborhood = skeleton_img[max(0, y-2):min(skeleton_img.shape[0], y+3),
                                      max(0, x-2):min(skeleton_img.shape[1], x+3)]
            if np.any(neighborhood > 0):
                supported_points += 1
    
    support_ratio = supported_points / total_points if total_points > 0 else 0
    return support_ratio >= min_support_ratio

def find_line_connections(corners, skeleton_img, max_distance=150):
    """Find valid line connections between corners"""
    connections = []
    
    if len(corners) < 2:
        return connections
    
    # Use KDTree for efficient nearest neighbor search
    tree = KDTree(corners)
    
    for i, corner in enumerate(corners):
        # Find nearby corners
        nearby_indices = tree.query_ball_point(corner, r=max_distance)
        
        for j in nearby_indices:
            if j <= i:  # Avoid duplicate pairs and self-connections
                continue
            
            corner_a, corner_b = corners[i], corners[j]
            
            # Check if line has sufficient support
            if check_line_support(corner_a, corner_b, skeleton_img):
                connections.append((corner_a, corner_b))
    
    return connections

def detect_curves(skeleton_img, min_curve_length=40):
    """Detect curved segments in the skeleton"""
    curves = []
    
    # Find contours in the skeleton
    contours, _ = cv2.findContours(skeleton_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        if len(contour) >= min_curve_length:
            # Simplify contour to get key points
            epsilon = 0.01 * cv2.arcLength(contour, False)
            simplified = cv2.approxPolyDP(contour, epsilon, False)
            
            if len(simplified) >= 3:  # Need at least 3 points for a curve
                curve_points = [tuple(pt[0]) for pt in simplified]
                curves.append(curve_points)
    
    return curves

def fit_bezier_curve(points):
    """Fit a quadratic Bezier curve to a set of points"""
    if len(points) < 3:
        return None
    
    # Use first, middle, and last points as control points
    p0 = np.array(points[0])
    p2 = np.array(points[-1])
    
    # Estimate middle control point
    mid_idx = len(points) // 2
    p1 = np.array(points[mid_idx])
    
    # Adjust p1 to better fit the curve
    # This is a simplified approach - more sophisticated curve fitting could be used
    direction = p2 - p0
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular) if np.linalg.norm(perpendicular) > 0 else perpendicular
    
    # Project middle point onto perpendicular
    mid_actual = np.array(points[mid_idx])
    line_mid = (p0 + p2) / 2
    offset = np.dot(mid_actual - line_mid, perpendicular)
    
    p1 = line_mid + offset * perpendicular
    
    return p0, p1, p2

def draw_bezier_curve(img, control_points, color=(255, 0, 0), thickness=2):
    """Draw a quadratic Bezier curve"""
    if len(control_points) != 3:
        return img
    
    p0, p1, p2 = control_points
    
    # Generate points along the curve
    t_values = np.linspace(0, 1, 100)
    curve_points = []
    
    for t in t_values:
        # Quadratic Bezier formula: (1-t)²P0 + 2(1-t)tP1 + t²P2
        point = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
        curve_points.append(tuple(point.astype(int)))
    
    # Draw the curve
    for i in range(len(curve_points) - 1):
        cv2.line(img, curve_points[i], curve_points[i+1], color, thickness)
    
    return img

def visualize_results(original_img, corners, line_connections, curves):
    """Visualize the detected corners, lines, and curves"""
    result_img = original_img.copy()
    
    # Draw line connections
    for corner_a, corner_b in line_connections:
        cv2.line(result_img, corner_a, corner_b, (0, 255, 0), 2)
    
    # Draw curves
    for curve_points in curves:
        if len(curve_points) >= 3:
            bezier_controls = fit_bezier_curve(curve_points)
            if bezier_controls:
                result_img = draw_bezier_curve(result_img, bezier_controls, (255, 0, 0), 2)
        else:
            # Draw as polyline if too few points for curve
            pts = np.array(curve_points, dtype=np.int32)
            cv2.polylines(result_img, [pts], False, (0, 0, 255), 2)
    
    # Draw corners
    for corner in corners:
        cv2.circle(result_img, corner, 4, (0, 0, 255), -1)
        cv2.circle(result_img, corner, 6, (255, 255, 255), 2)
    
    return result_img

def main():
    # Load image - replace with your image path
    img_path = "sPoche.jpg"  # Change this to your image path
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            print("Please make sure the image file exists and the path is correct.")
            return
        
        # Flip image if needed (uncomment if your image is upside down)
        # img = cv2.flip(img, 0)
        
        print("Processing image...")
        
        # Preprocess image
        gray, binary = preprocess_image(img)
        
        # Detect corners
        corners = detect_corners_enhanced(gray, binary)
        print(f"Detected {len(corners)} corners")
        
        # Create skeleton
        skeleton = create_skeleton(binary)
        
        # Find line connections
        line_connections = find_line_connections(corners, skeleton)
        print(f"Found {len(line_connections)} line connections")
        
        # Detect curves
        curves = detect_curves(skeleton)
        print(f"Detected {len(curves)} curves")
        
        # Visualize results
        result_img = visualize_results(img, corners, line_connections, curves)
        
        # Display results
        cv2.imshow("Original", img)
        cv2.imshow("Binary", binary)
        cv2.imshow("Skeleton", skeleton)
        cv2.imshow("Floor Plan Analysis", result_img)
        
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Optionally save result
        cv2.imwrite("floor_plan_analysis_result.jpg", result_img)
        print("Result saved as 'floor_plan_analysis_result.jpg'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check that all required libraries are installed:")
        print("pip install opencv-python numpy scikit-image scipy matplotlib")

if __name__ == "__main__":
    main()