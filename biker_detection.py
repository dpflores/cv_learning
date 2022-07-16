import cv2 as cv
import torch
from polygon import *

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.5    # Non-Maximum Suppression confidence threshold
model.classes = 0   # Just to detect persons according to coco.yalm

# Images
img = cv.imread('test_2.png') # OpenCV image 


# Inference
results = model(img.copy(), size=640)  # includes NMS

df = results.pandas().xyxy[0]  # predictions (pandas)
xmin = int(df.iloc[1].loc['xmin'])
xmax = int(df.iloc[1].loc['xmax'])
ymin = int(df.iloc[1].loc['ymin'])
ymax = int(df.iloc[1].loc['ymax'])


vertices_person = np.array([[xmin, ymin],[xmin, ymax],[xmax, ymax],[xmax, ymin]])
vertices_area = np.array([[2, 531],[513, 308],[659, 301],[580, 578]])

polygon1 = MyPolygon(vertices_person,color=(0, 255, 0))
polygon2 = MyPolygon(vertices_area,color=(0, 0, 255))

# Combine YOLO results
alpha = 0.8
img2 = cv.addWeighted(np.squeeze(results.render()), alpha, img, 1 - alpha, 0)
# Just draw the area since yolo will show the box of the person
img2 = polygon2.draw_polygon(img2)
img2 = polygon2.fill_polygon(img2)




final_img = img2
#final_img = rescaleFrame(final_img, scale=1.5)
print(polygon1.intersection_percentage(polygon2))
cv.imshow('test image', final_img)

cv.waitKey(0)
# # Results
# results.print()  
# results.show()  # or .show()



