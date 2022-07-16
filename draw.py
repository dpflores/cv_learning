import cv2 as cv
import numpy as np


def rescaleFrame(frame, scale = 0.75):
    
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread('test_image.png')

overlay = img.copy() # Create a layer with perimeter
overlay2 = img.copy()   # Create a layer with filled area

# Vertices of the quadrilaterla
vertices = np.array([[13, 315],[10, 209],[134, 169],[225, 169]])

# Quadrilateral perimeter
cv.polylines(overlay, [vertices], isClosed=True, color=(0,0,255), thickness=3)

# Quadrilateral area
cv.fillPoly(overlay2, pts = [vertices], color =(0,0,255))

# Adding the perimeter layer with certain transparency
alpha = 0.8
img1 = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)

alpha = 0.4
img2 = cv.addWeighted(overlay2, alpha, img1, 1 - alpha, 0)

final_img = img2
#final_img = rescaleFrame(img, scale=1.5)

cv.imshow('test image', final_img)
cv.waitKey(0)