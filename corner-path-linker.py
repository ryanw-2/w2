import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import KDTree


def detect_corners(gray_img, max_corners=100, quality_level=0.01, min_distance=10):
    corners = cv2.goodFeaturesToTrack(gray_img, max_corners, quality_level, min_distance)
    return [tuple(pt[0]) for pt in corners.astype(int)] if corners is not None else []


def skeletonize_image(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    binary[binary == 255] = 1
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return skeleton


def path_exists_between(corner_a, corner_b, skeleton_img, threshold_ratio=0.65):
    """
    Checks if a path exists between two corners by verifying the overlap
    of a straight line with the skeletonized image.
    """
    line_mask = np.zeros(skeleton_img.shape, dtype=np.uint8)
    cv2.line(line_mask, corner_a, corner_b, (0,255,0), 8) # Draw a line ONLY for the current pair

    # Find where the drawn line and the skeleton overlap
    overlap = cv2.bitwise_and(line_mask, skeleton_img)

    skeleton_pixels_in_line_area = np.count_nonzero(cv2.bitwise_and(skeleton_img, skeleton_img, mask=line_mask))
    match_pixels = np.count_nonzero(overlap)

    if skeleton_pixels_in_line_area == 0:
        return False

    # Check if the ratio of overlapping pixels to total line pixels is above the threshold
    return (match_pixels / skeleton_pixels_in_line_area) >= threshold_ratio


def find_connected_corner_pairs(corners, skeleton_img, max_dist=100):
    connected_pairs = []
    if not corners: # Guard against empty corners list
        return []
    tree = KDTree(corners)
    for i, corner in enumerate(corners):
        nearby_ids = tree.query_ball_point(corner, r=max_dist)
        for j in nearby_ids:
            if j <= i:
                continue
            a, b = corners[i], corners[j]
            if path_exists_between(a, b, skeleton_img):
                connected_pairs.append((a, b))
    return connected_pairs


def draw_connected_lines(img, pairs, color=(0, 255, 0)):
    for a, b in pairs:
        cv2.line(img, a, b, color, 1)
    return img


if __name__ == "__main__":
    # Load and preprocess image
    img = cv2.imread("sPoche.jpg")
    flipped = cv2.flip(img, 0)
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)

    # Step 1: Corner detection
    corners = detect_corners(gray)
    print(f"Detected {len(corners)} corners")

    # Step 2: Skeletonization
    skeleton_img = skeletonize_image(gray)

    cv2.imshow("Skeleton", skeleton_img)
    # Step 3: Connect corners if line overlaps skeleton path
    pairs = find_connected_corner_pairs(corners, skeleton_img)
    print(f"Found {len(pairs)} valid corner connections")

    # Step 4: Visualize result
    vis_img = draw_connected_lines(flipped.copy(), pairs)
    for pt in corners:
        cv2.circle(vis_img, pt, 2, (0, 0, 255), -1)

    cv2.imshow("Reconstructed Geometry", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()