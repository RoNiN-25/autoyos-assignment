# import the necessary packages
from imutils import face_utils
import time
import dlib
import cv2
import numpy as np


def pre_process(frame):
    kernel = np.ones((9, 9), np.uint8)  # kernel for erosion and dilation
    frame = cv2.erode(frame, kernel, iterations=1)
    frame = cv2.dilate(frame, kernel, iterations=1)
    frame = cv2.medianBlur(frame, 3)

    return frame


def get_eyes(frame, shape):
    # for right eye
    xr = shape[36][1]
    yr = min(shape[37][0], shape[38][0])
    wr = shape[39][1] - shape[36][1]
    hr = max(shape[40][0], shape[41][0]) - yr

    right_eye = frame[int(0.9 * xr): int(1.1 * (xr + wr)), int(0.9 * yr): int(1.1 * (yr + hr))]

    # for left eye
    xl = shape[42][1]
    yl = min(shape[43][0], shape[44][0])
    wl = shape[45][1] - shape[42][1]
    hl = max(shape[46][0], shape[47][0]) - yl

    left_eye = frame[int(0.9 * xl): int(1.1 * (xl + wl)), int(0.95 * yl): int(1.05 * (yl + hl))]

    return [[right_eye,int(0.9*yr),int(0.9*xr)], [left_eye,int(0.95*yl),int(0.9*xl)]]

def get_eye(frame, direction, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # loop over the face detections
    if len(rects)>0:
        rect = rects[0]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # landmarks 36 to 47 correspond to the eyes

        right_eye, left_eye = get_eyes(frame, shape)
        if direction == 'right':
            return [1,right_eye]
        elif direction =='left':
            return [1,left_eye]
    else:
        return [-1,None]
