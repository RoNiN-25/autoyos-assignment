#################### Main module which connects all the others together ####################
# import necessary packages
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


# Create a dlib face detector
detector = dlib.get_frontal_face_detector()
# Create a 81 points based landmark detector
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
# Create a Haar Classifer for the eye
haar_classifier = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Set the initial eye to be chosen as left
direction = 'left'

ret = True
# Create cv2 video capture for capturing the video
cap = cv2.VideoCapture(0);
# read the first frame
_,k = cap.read()

# get the size of the frame and set the desired position as center of the frame
des_pos = [k.shape[1]//2, k.shape[0]//2]

# Set the PID constants for ech axis
# Need to be tuned based on hardware
pid_x = [0.1,0,0.1]
pid_y = [0.1,0,0.1]
# Store the error data of previous interation for next one
# prev_error_data -> [error_x, error_y, sum_error_x, sum_error_y]
prev_error_data = [0, 0, 0, 0]

# Create a logger
logger = logger.Logger('INC', "3A4Bh-Ref25","", config.config, dir="models/")
logger.log("Start inferring model...")
# Create a tensorflow session
sess = tf.Session()
# Load the model
model = inferno.load_model(sess, 'INC', '3A4Bh-Ref25', logger)

# reset the position of the x-y-z platform
controller.reset(cap, detector)

# Infinite loop
while ret:
    ret, frame = cap.read() # Read the frame from the camera

    # Get the eye location by running the function in module Eye_locator
    '''
        x -> x location of the center of the eyeball/pupil
        y -> y location of the center of the eyeball/pupil
        r -> radius of the eyeball/pupil
        eye -> segmented out image of the eye
    '''
    x, y, r, eye = Eye_locator.get_eye_location(frame, direction, detector, predictor, haar_classifier ,sess, model, logger)
    # If eye detection failed, then it is assumed that the camera is very close to the eye
    if x == -1 or y == -1 or r == -1:
        # If direction being detected is left, then change it to right, move the camera a bit forward and take an Image
        # This is the image of the eye, store it
        if direction == 'left':
            ## move forward in z by a small distance
            _,f = cap.read()
            cv2.imwrite('left.jpg',f)
            direction = 'right'
            # reset the position of the x-y-z platform
            controller.reset(cap,detector)
            # Set the error data to 0
            prev_error_data = [0, 0, 0, 0]
            continue
        # if the direction being detected is right, then move the camera a bit forward and take an Image
        # this is the image of the right eye, store it and exit the program
        elif direction == 'right':
            ## move forward in z by a small distance
            _,f = cap.read()
            cv2.imwrite('right.jpg', f)
            break
        continue

    # Get the distance to be moved from the PID controller, given the current and desired location of the eye
    dist_x, dist_y, prev_error_data = controller.calc_distance([x,y], des_pos, pid_x, pid_y, prev_error_data)

    # Calculate the number of steps to rotate the stepper motor by to move by the given distance
    num_steps_x = controller.calc_steps(dist_x)
    num_steps_y = controller.calc_steps(dist_y)

    # If the distance to be moved is less than some threshold which needs to be tuned based on the hardware, move forward
    if abs(num_steps_x) < 100 and abs(num_steps_y) < 100:
        print('Move forward')
        # send a fixed forward motion number of steps
    else:
        # move the camera in appropriate direction to attain the desired position
        if num_steps_x < 0:
            print('Move right')
            # send this value to the micro-controller
        else:
            print('Move left')
            # send this value to the micro-controller

        if num_steps_y < 0:
            print('Move down')
            # send this value to the micro-controller
        else:
            print('Move up')
            # send this value to the micro-controller


    # Show the eyeball/pupil on the frame captured
    frame = utils.annotator((0,255,0), frame, x, y, r)
    # Display the frame
    cv2.imshow("Image",frame)

    # if q is pressed, exit the program
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
# perform cleanup
cv2.destroyAllWindows()
cap.release()
