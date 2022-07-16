import cv2 as cv
import numpy as np
from polygon import *

def rescaleFrame(frame, scale = 0.75):
    
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread('test_image.png')


# Vertices of the quadrilateral
vertices1 = np.array([[13, 315],[10, 209],[134, 169],[225, 169]])
vertices2 = np.array([[13+100, 315],[10+100, 209],[134, 169],[225, 169]])

polygon1 = MyPolygon(vertices1,color=(0, 0, 255))
polygon2 = MyPolygon(vertices2,color=(0, 255, 0))

img2 = polygon1.draw_polygon(img)
img2 = polygon1.fill_polygon(img2)
img2 = polygon2.draw_polygon(img2)
img2 = polygon2.fill_polygon(img2)

final_img = img2
#final_img = rescaleFrame(final_img, scale=1.5)
print(polygon1.intersection_percentage(polygon2))
cv.imshow('test image', final_img)
cv.waitKey(0)