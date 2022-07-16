from turtle import width
import cv2 as cv


def rescaleFrame(frame, scale = 0.75):
    
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Reading video
capture = cv.VideoCapture(0)    # 0 for webcamera

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, 0.5)
    
    cv.imshow('video', frame)
    cv.imshow('video resized', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('q'):   # If we press the q key, we will stop the video
        break

capture.release()
cv.destroyAllWindows()