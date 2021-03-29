#################### Module for the Landmark based eye detector using dlib ####################
# import the necessary packages
from imutils import face_utils
import time
import dlib
import cv2
import numpy as np

# function for pre-processing the frame
def pre_process(frame):
    '''
        Input:
            frame -> the frame to be pre-processed
        Output:
            The pre-processed frame
    '''
    frame = cv2.equalizeHist(frame)
    kernel = np.ones((3,3),np.uint8) # Kernel for erosion and dilation
    frame = cv2.dilate(frame, kernel, iterations=1)
    frame = cv2.erode(frame, kernel, iterations=1)
    return frame

# function to segement out the eyes
def get_eyes(frame, shape):
    '''
        Input:
            frame -> The frame from which to segment the eye from
            shape -> The output of the 81 points predictor giving the location of each of the 81 points
        Output:
            list containing the right eye, left eye and co-ordinates of the corner of the segmented region
    '''
    # for right eye
    # points 36,37,38,39,40,41 correspond to right eye

    # Create a bounding box around the eye based on the points
    xr = shape[36][1]
    yr = min(shape[37][0], shape[38][0])
    wr = shape[39][1] - shape[36][1]
    hr = max(shape[40][0], shape[41][0]) - yr

    # Segment out the section of the image lying in these bounds which is the right eye
    right_eye = frame[int(0.9 * xr): int(1.1 * (xr + wr)), int(0.9 * yr): int(1.1 * (yr + hr))]

    # for left eye
    # points 42,43,44,45,46,47 correspond to left eye

    # Create a bounding box around the eye based on the points
    xl = shape[42][1]
    yl = min(shape[43][0], shape[44][0])
    wl = shape[45][1] - shape[42][1]
    hl = max(shape[46][0], shape[47][0]) - yl

    # Segment out the section of the image lying in these bounds which is the left eye
    left_eye = frame[int(0.9 * xl): int(1.1 * (xl + wl)), int(0.95 * yl): int(1.05 * (yl + hl))]

    # return the data
    return [[right_eye,int(0.9*yr),int(0.9*xr)], [left_eye,int(0.95*yl),int(0.9*xl)]]

# function to detect the eye, segment it and return the segment along with status
def get_eye(frame, direction, detector, predictor):
    '''
        Input:
            frame -> Frame from which to segment the eye from
            direction -> Direction of the eye to detect and return
            detector -> dlib's face detector
            predictor -> the 81 points landmark detector
        Output:
            status and the segmented out eye data
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # pre-process the image
    gray = pre_process(gray)
    # get the faces from the image
    rects = detector(gray, 0)

    # loop over the face detections
    if len(rects)>0:
        # Choose the first face -> assuming only the person whose eye is to be detected is present int he frame of the camera
        rect = rects[0]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Get the segmented out left and right eyes
        right_eye, left_eye = get_eyes(frame, shape)
        # If the direction is right, then send the right eye with status 1 implying success
        if direction == 'right':
            return [1,right_eye]
        # If the direction is left, then send the left eye with status 1 implying success
        elif direction =='left':
            return [1,left_eye]
    # If no face is detected then send status -1 implying failure
    else:
        return [-1,None]
