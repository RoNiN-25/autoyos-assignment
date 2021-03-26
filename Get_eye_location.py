import haar_eye_detector
import Get_Eyes
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import dlib
import logger
import utils
import config
import inferno


def predict(frame):
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


    status,eye_data = Get_Eyes.get_eye(frame, direction, detector, predictor)

    if status == 1:
        eye, x, y = eye_data
        cv2.imshow('EYE',eye)

        x_pred, y_pred, w_pred = predict(eye)

        print("Face Mode")

        locx = x_pred + x
        locy = y_pred + y
        radius = w_pred
    elif status == -1:
        status, eye_data = haar_eye_detector.get_eye(frame, direction, haar_classifier)

        if status == 1:
            eye, x, y = eye_data
            cv2.imshow('haar eye', eye)

            x_pred, y_pred, w_pred = predict(eye)

            print("Haar mode")

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


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
# haar_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
haar_classifier = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

direction = 'left'

ret = True
cap = cv2.VideoCapture(0);

logger = logger.Logger('INC', "3A4Bh-Ref25","", config.config, dir="models/")
logger.log("Start inferring model...")
sess = tf.Session()
model = inferno.load_model(sess, 'INC', '3A4Bh-Ref25', logger)

while ret:
    ret, frame = cap.read()

    x, y, r, eye = get_eye_location(frame, direction, detector, predictor, haar_classifier ,sess, model, logger)
    if x == -1 or y == -1 or r == -1:
        if direction == 'left':
            cv2.imwrite('left.jpg',eye)
            direction = 'right'
            continue
        # elif direction == 'right':
        #     cv2.imwrite('right.jpg', eye)
        #     break
        continue

    frame = utils.annotator((0,255,0), frame, x, y, r)
    cv2.imshow("Image",frame)

    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
