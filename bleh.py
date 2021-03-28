import Eye_locator
import controller

import time

import cv2
import numpy as np
import dlib

import tensorflow.compat.v1 as tf

import logger
import config
import inferno
import utils



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
# haar_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
haar_classifier = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

direction = 'left'

ret = True
cap = cv2.VideoCapture(0);
_,k = cap.read()

des_pos = [k.shape[1]//2, k.shape[0]//2]

pid_x = [0.1,0,0.1]
pid_y = [0.1,0,0.1]
prev_error_data = [0, 0, 0, 0]

logger = logger.Logger('INC', "3A4Bh-Ref25","", config.config, dir="models/")
logger.log("Start inferring model...")
sess = tf.Session()
model = inferno.load_model(sess, 'INC', '3A4Bh-Ref25', logger)

while ret:
    ret, frame = cap.read()

    x, y, r, eye = Eye_locator.get_eye_location(frame, direction, detector, predictor, haar_classifier ,sess, model, logger)
    if x == -1 or y == -1 or r == -1:
        if direction == 'left':
            ## move forward in z by a small distance
            _,f = cap.read()
            cv2.imwrite('left.jpg',f)
            direction = 'right'
            controller.reset(cap,detector)
            prev_error_data = [0, 0, 0, 0]
            continue
        elif direction == 'right':
            ## move forward in z by a small distance
            _,f = cap.read()
            cv2.imwrite('right.jpg', f)
            break
        continue


    dist_x, dist_y, prev_error_data = controller.calc_distance([x,y], des_pos, pid_x, pid_y, prev_error_data)

    num_steps_x = controller.calc_steps(dist_x)
    num_steps_y = controller.calc_steps(dist_y)


    if abs(num_steps_x) < 100 and abs(num_steps_y) < 100:
        print('Move forward')
        # send a fixed forward motion number of steps
    else:
        if num_steps_x < 0:
            print('Move left')
            # send this value to the micro-controller
        else:
            print('Move right')
            # send this value to the micro-controller

        if num_steps_y < 0:
            print('Move down')
            # send this value to the micro-controller
        else:
            print('Move up')
            # send this value to the micro-controller


    frame = utils.annotator((0,255,0), frame, x, y, r)
    cv2.imshow("Image",frame)

    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
