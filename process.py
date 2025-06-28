import cv2
import numpy as np
import math
import csv
from skimage.morphology import skeletonize

"""
mysql username: root
mysql passwd:    abcd
"""

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_red = np.array([0,100,100])
upper_red = np.array([20,255,255])
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,30])

def callback(input):
    pass

'''
Takes in an image and returns an array of 
canny detected edges representing the different
layers of the image.

input type : Matlike
return type : Matlike list
'''
def image_to_edge(input_im):

    hsv = cv2.cvtColor(input_im, cv2.COLOR_BGR2HSV)  

    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    masks = [red_mask, blue_mask, black_mask]
    cannies = []

    for msk in masks:
        masked = cv2.bitwise_and(input_im, input_im, mask= msk)
        grayed = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayed, (5,5), 0)
        _, threshed = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY_INV)
        closed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel, iterations=1)
        edged = cv2.Canny(closed, 50, 150, apertureSize=3)
        cannies.append(edged)
    
    return cannies

'''
Run time helper function that takes in an array of 
canny edges and runs PHT, returning array of all lines

input type : Matlike list, int, float, float
output type : Matlike list
'''
def edge_to_lines(edges, thresh, minLine, maxLine):

    all_lines = []

    for edge in edges:
        # Detect lines using the probabilistic Hough transform
        segments = cv2.HoughLinesP(edge, 
                                rho=1, 
                                theta=np.pi/180, 
                                threshold=thresh,    # minimum number of votes
                                minLineLength=minLine,  # adjust depending on your scale
                                maxLineGap=maxLine)      # allowed gap between segments

        lines = []
        if segments is not None:
            lines += [tuple(sg[0]) for sg in segments]
        
        all_lines += lines

    print(all_lines)
    return all_lines

'''
Run time helper function to sort lines and add them to
detected if they are not null

input type: Matlike, Matlike list
return type: Matlike list
'''
def extract_geometry_from_sketch(image_path: str, epsilon_multiplier: float = 0.001, visualize_steps: bool = True):
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
    
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    binary_img[binary_img == 255] = 1
    skeleton_img = skeletonize(binary_img)
    skeleton_img = (skeleton_img.astype(np.uint8) * 255)

    # if visualize_steps:
    #     cv2.imshow("1. Skeletonized Image", skeleton_img)
    #     cv2.waitKey(0)
    
    ##
    contours, _ = cv2.findContours(skeleton_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"Found {len(contours)} initial contours.")
    ##
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
            np_path = np.array(path, dtype = np.int32)
            cv2.polylines(vis_img, [np_path], isClosed=False, color= (0,255,0), thickness=1)
        cv2.imshow("2. Approximated Geometry", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ##

    # orig_h, orig_w = input_img.shape[:2]
    # new_w = 600
    # new_h = int((new_w / orig_w) * orig_h)

    # cannies = image_to_edge(input_img)

    # # Setting up GUI
    # winname = 'canny'
    # cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE) 
    # cv2.resizeWindow(winname, new_w, new_h)              # force it to 800×600 pixels
    # cv2.createTrackbar('threshold', winname, 100, 255, callback)
    # cv2.createTrackbar('minLineLength', winname, 50, 100, callback)
    # cv2.createTrackbar('maxLineGap', winname, 10, 100, callback)

    # # Run-time Loop
    # while True:
       
    #     # Extracting User Input
    #     thresh = cv2.getTrackbarPos('threshold', winname)
    #     minLine = cv2.getTrackbarPos('minLineLength', winname)
    #     maxLine = cv2.getTrackbarPos('maxLineGap', winname)

    #     line_img = np.copy(input_img)
    #     lines_list = []
    #     detected_list = []
        
    #     for edges in cannies:
    #     # Detect lines using the probabilistic Hough transform
            
    #         lines = cv2.HoughLinesP(edges, 
    #                             rho=1, 
    #                             theta=np.pi/180, 
    #                             threshold=thresh,    # minimum number of votes
    #                             minLineLength=minLine,  # adjust depending on your scale
    #                             maxLineGap=maxLine)      # allowed gap between segments

    #         all_lines = []
    #         if lines is not None:
    #             all_lines += [tuple(sg[0]) for sg in lines]

    #         detected = []
    #         if lines is not None:
                
    #             for line in lines:
    #                 x1, y1, x2, y2 = line[0]
    #                 cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)    
    #                 detected.append((x1,y1,x2,y2)) 

    #             detected_list += detected

    #     # on exit, group the lines into a list
    #     if cv2.waitKey(1) == ord('q'):
            
    #         with open('lines.csv', 'w', newline='') as f:
    #             w = csv.writer(f)
    #             w.writerow(['x1','y1','x2','y2'])

    #             flag = 0
    #             # for detected in detected_list:
    #             #     w.writerow([flag])
    #             #     for x1,y1,x2,y2 in detected:
    #             #         w.writerow([x1,y1,x2,y2])

    #             #     flag += 1
    #             w.writerow([flag])
    #             for x1,y1,x2,y2 in detected_list[0]:
    #                 w.writerow([x1,y1,x2,y2])

    #         print(f"Exported segments to lines.csv")


    #         cv2.destroyAllWindows()
    #         break

    #     # # Display the image with detected lines.
    #     preview = cv2.resize(line_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #     cv2.imshow(winname, preview)
        

    


if __name__ == '__main__':
    extract_geometry_from_sketch("./testPlanSketch-Cleaned.jpg", visualize_steps=True)


