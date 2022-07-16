import cv2 as cv
import numpy as np

from shapely.geometry import Polygon

class MyPolygon:
    def __init__(self, vertices, color=(0, 0, 255)):
        self.vertices = vertices
        self.color = color
        self.geometry_polygon = Polygon(vertices)
    
    def draw_polygon(self, img, alpha=0.6):
        overlay = img.copy() # Create a layer with perimeter
    
        # Quadrilateral perimeter
        cv.polylines(overlay, [self.vertices], isClosed=True, color=self.color, thickness=3)
        new_img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return new_img

    def fill_polygon(self, img, alpha=0.3):
        overlay = img.copy()   # Create a layer with filled area
        # Quadrilateral area
        cv.fillPoly(overlay, pts = [self.vertices], color=self.color)
        new_img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return new_img
    
    def intersection_percentage(self, polygon):
        p1 = self.geometry_polygon
        p2 = polygon.geometry_polygon
        percentage = 100*p1.intersection(p2).area / p1.area
        return percentage
