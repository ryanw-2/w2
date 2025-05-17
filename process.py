import cv2
import numpy as np
import math
import csv

def callback(input):
    pass

def process():
    # Load input image using cv2.imread then converts 
    # to grayscale
    input_img = cv2.imread('rgb_plan_test.jpg')
    
    ##
    orig_h, orig_w = input_img.shape[:2]
    new_w = 600
    new_h = int((new_w / orig_w) * orig_h)
    
    ##

    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_red = np.array([0,100,100])
    upper_red = np.array([20,255,255])
    lower_black = np.array([255,255,255])
    upper_black = np.array([255,255,255])

    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    red_res = cv2.bitwise_and(input_img, input_img, mask= red_mask)
    blue_res = cv2.bitwise_and(input_img, input_img, mask= blue_mask)
    black_res = cv2.bitwise_and(input_img, input_img, mask= black_mask)
    
    red_gray = cv2.cvtColor(red_res, cv2.COLOR_BGR2GRAY)
    blue_gray = cv2.cvtColor(blue_res, cv2.COLOR_BGR2GRAY)
    black_gray = cv2.cvtColor(black_res, cv2.COLOR_BGR2GRAY)
    
    # reduce noise with Gaussian Blur 
    red_blurred = cv2.GaussianBlur(red_gray, (5,5), 0)
    blue_blurred = cv2.GaussianBlur(blue_gray, (5,5), 0)
    black_blurred = cv2.GaussianBlur(black_gray, (5,5), 0)

    # cv2.imshow('red-res', red_res)
    # cv2.imshow('blue-res', blue_res)
    # cv2.imshow('black-res', black_res)

    ## intermediate DEBUG
    cv2.imshow("Edges", red_blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ##

    # Simple binary threshold, should be enough for well documented pictures
    _, red_thresh = cv2.threshold(red_blurred, 1, 255, cv2.THRESH_BINARY_INV)
    _, blue_thresh = cv2.threshold(blue_blurred, 1, 255, cv2.THRESH_BINARY_INV)
    _, black_thresh = cv2.threshold(black_blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    ## intermediate DEBUG
    cv2.imshow("Edges", black_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ##

    # Morphological closing to fill small gaps in strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Canny edge detection, good for object finding
    edges = cv2.Canny(closed, 50, 150, apertureSize=3)

    # Setting up GUI
    winname = 'canny'
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE) 
    cv2.resizeWindow(winname, new_w, new_h)              # force it to 800×600 pixels
    cv2.createTrackbar('threshold', winname, 100, 255, callback)
    cv2.createTrackbar('minLineLength', winname, 50, 100, callback)
    cv2.createTrackbar('maxLineGap', winname, 10, 100, callback)

    # Run-time Loop
    while True:
       
        # Extracting User Input
        thresh = cv2.getTrackbarPos('threshold', winname)
        minLine = cv2.getTrackbarPos('minLineLength', winname)
        maxLine = cv2.getTrackbarPos('maxLineGap', winname)

        # Detect lines using the probabilistic Hough transform
        lines = cv2.HoughLinesP(edges, 
                                rho=1, 
                                theta=np.pi/180, 
                                threshold=thresh,    # minimum number of votes
                                minLineLength=minLine,  # adjust depending on your scale
                                maxLineGap=maxLine)      # allowed gap between segments
                
        


        all_lines = []
        if lines is not None:
            all_lines += [tuple(line[0]) for line in lines]

        line_img = np.copy(input_img)
        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)    
                detected.append((x1,y1,x2,y2))    

        # on exit, group the lines into a list
        if cv2.waitKey(1) == ord('q'):
            
            with open('lines.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['x1','y1','x2','y2'])
                for x1,y1,x2,y2 in detected:
                    w.writerow([x1,y1,x2,y2])
        
            print(f"Exported {len(detected)} segments to lines.csv")


            cv2.destroyAllWindows()
            break

        # # Display the image with detected lines.
        preview = cv2.resize(line_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imshow(winname, preview)
        

    


if __name__ == '__main__':
    process()

