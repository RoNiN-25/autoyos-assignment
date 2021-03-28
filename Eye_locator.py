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


def predict(frame, model, sess):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_shape = frame.shape
    if frame.shape[0] != 192:
        frame = inferno.rescale(frame)

    frame = utils.gray_normalizer(frame)
    image = utils.change_channel(frame, config.config["input_channel"])
    [p] = model.predict(sess, [image])
    x_pred, y_pred, w_pred = inferno.upscale_preds(p, f_shape)

    return [x_pred, y_pred, w_pred]


def get_eye_location(frame, direction, detector, predictor, haar_classifier ,sess, model, logger):
    locx = 0
    locy = 0
    radius = 0
    eye = None

    global prevLoc


    status,eye_data = Landmark_eye_detector.get_eye(frame, direction, detector, predictor)

    if status == 1:
        eye, x, y = eye_data
        cv2.imshow('EYE',eye)

        x_pred, y_pred, w_pred = predict(eye, model, sess)

        # print("Face Mode")

        locx = x_pred + x
        locy = y_pred + y
        radius = w_pred
    elif status == -1:
        status, eye_data = Haar_eye_detector.get_eye(frame, direction, haar_classifier)

        if status == 1:
            eye, x, y = eye_data
            cv2.imshow('haar eye', eye)

            x_pred, y_pred, w_pred = predict(eye, model, sess)

            # print("Haar mode")

            locx = x_pred + x
            locy = y_pred + y
            radius = w_pred
        elif status == -1:
            eye = eye_data
            return [-1,-1,-1, eye]

        try:
            if prevLoc[3] == direction and abs(prevLoc[0] - locx) > frame.shape[1]/4 or abs(prevLoc[1] - locy) > frame.shape[0]/4 or abs(prevLoc[3] - radius) > 100:
                return [prevLoc[0],prevLoc[1],prevLoc[2],eye]
        except :
            pass

    prevLoc = [locx, locy, radius, direction]
    return [locx, locy, radius, eye]
