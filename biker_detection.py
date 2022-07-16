import cv2 as cv
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
img = cv.imread('test_2.png')[..., ::-1] # OpenCV image (BGR to RGB)


# Inference
results = model(img, size=640)  # includes NMS

# Results
results.print()  
results.show()  # or .show()


print(results.pandas().xyxy[0])  # im1 predictions (pandas)