import cv2 as cv
import torch
from polygon import *


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

# Images
img = cv.imread('test_2.png') # OpenCV image 


# Inference
results = model(img.copy(), size=640)  # includes NMS

df = results.pandas().xyxy[0]  # predictions (pandas)
for index, row in df.iterrows():
    xmin = int(row.loc['xmin'])
    xmax = int(row.loc['xmax'])
    ymin = int(row.loc['ymin'])
    ymax = int(row.loc['ymax'])


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



    
    #final_img = rescaleFrame(final_img, scale=1.5)
    percentage = polygon1.intersection_percentage(polygon2)

    if percentage > 0:
        text = f'Person {index + 1} intersects {np.round(percentage,2)}%'
        draw_text(img2, text, text_color=(0, 0, 255))
        #img2 = cv.putText(img2, text,(827, 37), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA, False)

    final_img = img2
    cv.imshow('test image'+ str(index), final_img)

cv.waitKey(0)
# # Results
# results.print()  
# results.show()  # or .show()



