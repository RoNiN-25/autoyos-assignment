# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np


def pre_process(frame):
	kernel = np.ones((5,5),np.uint8) # kernel for erosion and dilation
	frame = cv2.erode(frame,kernel,iterations=1)
	frame = cv2.dilate(frame,kernel,iterations=1)
	return frame

def get_eyes(frame,shape):
	# for right eye
	x = shape[36][1]
	y = min(shape[37][0],shape[38][0])
	w = shape[39][1]-shape[36][1]
	h = max(shape[40][0],shape[41][0]) - y

	right_eye = frame[int(0.9*x) : int(1.1*(x+w)) , int(0.9*y) : int(1.1*(y+h))]

	# for left eye
	x = shape[42][1]
	y = min(shape[43][0],shape[44][0])
	w = shape[45][1]-shape[42][1]
	h = max(shape[46][0],shape[47][0]) - y

	left_eye = frame[int(0.9*x) : int(1.1*(x+w)) , int(0.95*y) : int(1.05*(y+h))]

	return [right_eye, left_eye]


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream,
	frame = vs.read()
	#pre-process to remove noise
	frame = pre_process(frame)
	# convert to gray scale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

    	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		# landmarks 36 to 47 correspond to the eyes
		for (x, y) in shape[36:48]:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		# get the eyes cropped
		right_eye, left_eye = get_eyes(frame,shape)
		# display the eyes
		cv2.imshow("Right eye", right_eye)
		cv2.imshow("Left eye", left_eye)




	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
