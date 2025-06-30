import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('blox.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
 
corners = cv.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=5)
corner_pts = [tuple(pt[0]) for pt in corners]
 
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)
 
plt.imshow(img),plt.show()