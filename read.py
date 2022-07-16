import cv2 as cv

# # Reading image
# img = cv.imread('test_image.png')

# cv.imshow('test image', img)

# cv.waitKey(0)

# Reading video
capture = cv.VideoCapture('test_video.mp4')    # 0 for webcamera

while True:
    isTrue, frame = capture.read()
    cv.imshow('video', frame)

    if cv.waitKey(20) & 0xFF == ord('q'):   # If we press the q key, we will stop the video
        break

capture.release()
cv.destroyAllWindows()