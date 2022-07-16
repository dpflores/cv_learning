import cv2 as cv
import numpy as np


def rescaleFrame(frame, scale = 0.75):
    
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread('test_image.png')

# Paint the image a certain color 
v1 = (13, 315)
v2 = (10, 209)
v3 = (134, 169)
v4 = (225, 169)
cv.line(img, v1, v2, (0,0,255), thickness=3)
cv.line(img, v2, v3, (0,0,255), thickness=3)
cv.line(img, v3, v4, (0,0,255), thickness=3)
cv.line(img, v4, v1, (0,0,255), thickness=3)

final_img = img
#final_img = rescaleFrame(img, scale=1.5)

cv.imshow('test image', final_img)
cv.waitKey(0)