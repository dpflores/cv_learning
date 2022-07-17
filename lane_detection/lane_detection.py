import cv2 as cv
import numpy as np

img = cv.imread('test_image.jpg')
lane_image = np.copy(img)
gray = cv.cvtColor(lane_image, cv.COLOR_RGB2GRAY)

# Applying gaussian blur of 5 size window
blur = cv.GaussianBlur(gray, (5,5), 0)


cv.imshow('test_image', blur)
cv.waitKey(0)