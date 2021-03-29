#################### Module for getting the location of the eyeball/pupil ####################
# import required libraries
import Haar_eye_detector
import Landmark_eye_detector
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import dlib
import logger
import utils
import config
import inferno

# function for running the ML model on the image to obtain the pupil/eyeball
def predict(frame, model, sess):
    '''
    Input:
        frame -> Image of the eye segmented out from the camera captured Image
        model -> The tensorflow model
        sess -> Tensorflow session
    Output:
        List containing the the predicted x and y location of the eyeball's center and the radius
    '''
    # Convert frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get the shape
    f_shape = frame.shape
    # rescale if not properly sized
    if frame.shape[0] != 192:
        frame = inferno.rescale(frame)

    # pre-processing
    frame = utils.gray_normalizer(frame)
    image = utils.change_channel(frame, config.config["input_channel"])
    # Run the model to get the prediction
    [p] = model.predict(sess, [image])
    # upscale to input image co-ordinates
    x_pred, y_pred, w_pred = inferno.upscale_preds(p, f_shape)

    return [x_pred, y_pred, w_pred]

# function to get the eye location
def get_eye_location(frame, direction, detector, predictor, haar_classifier ,sess, model, logger):
    '''
    Input:
        frame -> Image captured from the camera
        direction -> left or right - which eye to detect
        detector -> dlib's face detector
        predictor -> dlib's 81 points Landmark predictor
        haar_classifier -> The haar cascade based classifier
        sess -> Tensorflow Session
        model -> Tensorflow model for the pupil detector
        logger -> Logger for logging data
    Output:
        List containing the x location, y location and the radius of the eyeball/pupil
        Also returns the segmented out eye
    '''
    locx = 0 # x Location
    locy = 0 # y loation
    radius = 0 # radius
    eye = None # the segmented out eye

    # Global variable to store the previous location information of the eye
    global prevLoc

    # run the landmark detector to obtaint he status and the eye data
    status,eye_data = Landmark_eye_detector.get_eye(frame, direction, detector, predictor)

    # if successful, status = 1
    if status == 1:
        # extract the segmented out eye, and the co-ordinates of the corner of the eye in the camera captured image
        eye, x, y = eye_data
        # cv2.imshow('EYE',eye)

        # run the prediction
        x_pred, y_pred, w_pred = predict(eye, model, sess)

        # upscale the predicted values to the co-ordinates in the camera captured image
        locx = x_pred + x
        locy = y_pred + y
        radius = w_pred
    # if face detection failed, run haar classifier for direct detection of eye
    elif status == -1:
        # obtain the status and the eye data
        status, eye_data = Haar_eye_detector.get_eye(frame, direction, haar_classifier)

        # if haar classifier successfully detected the eye, then status = 1
        if status == 1:
            # extract segmented out eye, and the co-ordinates of the corner of the eye in the camera captured image
            eye, x, y = eye_data
            # cv2.imshow('haar eye', eye)

            # run the prediction
            x_pred, y_pred, w_pred = predict(eye, model, sess)

            # upscale the predicted values to the co-ordinates in the camera captured image
            locx = x_pred + x
            locy = y_pred + y
            radius = w_pred
        # haar classifier fails in eye detection, then assume that the camera is very close to the eye
        # Initial assumption made is that at start of code, face is visible in the image
        elif status == -1:
            # get the eye data which is the image of the eye
            eye = eye_data
            # return error and the image of the eye
            return [-1,-1,-1, eye]

        # Preventing false positives:
        # Due to noise, the haar classifier has a lot of false positives.
        # since the platform moves slowly, the difference in the predicted location cannout jump by very large values

        try:
            # check if false positive
            if prevLoc[3] == direction and abs(prevLoc[0] - locx) > frame.shape[0]/4 or abs(prevLoc[1] - locy) > frame.shape[1]/4 or abs(prevLoc[3] - radius) > 100:
                # return previous locaiton in case of false positive
                return [prevLoc[0],prevLoc[1],prevLoc[2],eye]
        except :
            pass

    # update the previous location for the next iteration
    prevLoc = [locx, locy, radius, direction]
    # return the data
    return [locx, locy, radius, eye]
