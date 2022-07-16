import numpy as np
import cv2 as cv
import torch
from polygon import *

SAVE_VIDEO = True

def draw_text(img, text,
          font=cv.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=1,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.5    # Non-Maximum Suppression confidence threshold
model.classes = 0   # Just to detect persons according to coco.yalm
capture = cv.VideoCapture('test_video.mp4')    # 0 for webcamera

alert = False

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

size = (frame_width, frame_height)

if SAVE_VIDEO: out = cv.VideoWriter('/home/del/Del/Python/cv_learning/output.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, size)

while True:
    isTrue, frame = capture.read()

    if isTrue:
        # Yolo detections
        results = model(frame.copy())

        # Combine YOLO results
        alpha = 0.8
        frame_ed = cv.addWeighted(np.squeeze(results.render()), alpha, frame, 1 - alpha, 0)

        # Add polygon
        vertices_area = np.array([[2, 531],[513, 308],[659, 301],[580, 578]])
        polygon2 = MyPolygon(vertices_area,color=(0, 0, 255))
        frame_ed = polygon2.draw_polygon(frame_ed)
        frame_ed = polygon2.fill_polygon(frame_ed)

        df = results.pandas().xyxy[0]  # predictions (pandas)

        for index, row in df.iterrows():
            xmin = int(row.loc['xmin'])
            xmax = int(row.loc['xmax'])
            ymin = int(row.loc['ymin'])
            ymax = int(row.loc['ymax'])


            vertices_person = np.array([[xmin, ymin],[xmin, ymax],[xmax, ymax],[xmax, ymin]])
            

            polygon1 = MyPolygon(vertices_person,color=(0, 255, 0))
            

            percentage = polygon1.intersection_percentage(polygon2)

            if percentage > 0:
                alert = True
                text = f'Person {index + 1} intersects {np.round(percentage,2)}%'
                break


        if alert: draw_text(frame_ed, text, text_color=(0, 0, 255))
        alert = False
        final_frame = frame_ed

        if SAVE_VIDEO: out.write(final_frame)
        
        cv.imshow('test image', final_frame)


        if cv.waitKey(10) & 0xFF == ord('q'):   # If we press the q key, we will stop the video
            break

    else: break

capture.release()

if SAVE_VIDEO: out.release()

cv.destroyAllWindows()

if SAVE_VIDEO: print("The video was successfully saved")