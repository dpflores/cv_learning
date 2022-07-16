import numpy as np
import cv2 as cv
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.5 # Non-Maximum Suppression confidence threshold
model.classes = 0   # Just to detect persons according to coco.yalm
capture = cv.VideoCapture('test_video.mp4')    # 0 for webcamera

while True:
    isTrue, frame = capture.read()

    # Yolo detections
    results = model(frame)

    cv.imshow('YOLO', np.squeeze(results.render()))


    if cv.waitKey(10) & 0xFF == ord('q'):   # If we press the q key, we will stop the video
        break

capture.release()
cv.destroyAllWindows()