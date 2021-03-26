# import the necessary packages
import time
import cv2
import numpy as np
import utils


def get_eye(frame, direction, haar_classifier):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray scale
	gray = cv2.equalizeHist(gray)
	kernel = np.ones((3,3),np.uint8)
	gray = cv2.dilate(gray, kernel, iterations=1)
	gray = cv2.erode(gray, kernel, iterations=1)

	# detect the eyes
	eyes = haar_classifier.detectMultiScale(gray)

	if(len(eyes)>=2):
		sorted_eye = sorted(eyes,key = lambda x:x[2]*x[3])

		left_eye = None
		right_eye = None

		if((sorted_eye[-1][1] + sorted_eye[-1][3])//2 > (sorted_eye[-2][1] + sorted_eye[-2][3])//2):
			left_eye_c = sorted_eye[-1]
			right_eye_c = sorted_eye[-2]
		else:
			left_eye_c = sorted_eye[-2]
			right_eye_c = sorted_eye[-1]


		left_eye = frame[left_eye_c[1]:left_eye_c[1]+left_eye_c[3],left_eye_c[0]:left_eye_c[0]+left_eye_c[2]]
		right_eye = frame[right_eye_c[1]:right_eye_c[1]+right_eye_c[3],right_eye_c[0]:right_eye_c[0]+right_eye_c[2]]

		if direction == 'right':
			return [1, [right_eye, right_eye_c[0], right_eye_c[1]]]
		elif direction == 'left':
			return [1, [left_eye, left_eye_c[0], left_eye_c[1]]]

	elif(len(eyes) == 1):
		eye = frame[eyes[0][1]:eyes[0][1]+eyes[0][3],eyes[0][0]:eyes[0][0]+eyes[0][2]]
		return [1, [eye, eyes[0][0], eyes[0][1]]]

	else:
		return [-1, frame]
