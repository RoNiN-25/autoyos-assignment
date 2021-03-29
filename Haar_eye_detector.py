#################### Module for eye detection based on haar classifiers ####################
# import the necessary packages
import time
import cv2
import numpy as np
import utils

# function for pre-processing the frame
def pre_process(frame):
	'''
		Input:
			frame -> The frame to pre-process
		Output:
			The pre-processed frame
	'''

	frame = cv2.equalizeHist(frame)
	kernel = np.ones((3,3),np.uint8) # Kernel for erosion and dilation
	frame = cv2.dilate(frame, kernel, iterations=1)
	frame = cv2.erode(frame, kernel, iterations=1)
	return frame

# function to get the segmented out eye
def get_eye(frame, direction, haar_classifier):
	'''
		Input:
			frame -> The frame from which the eye is to be detected
			direction -> The direction of the eye to be detected
			haar_classifier -> The haar classifier to be used for eye detection
		Output:
			List containg the status and the segmented out eye data
			The eye data consists of the segmented out eye region and the co-ordinates of the corner of the
			bounding box used for segmentation
	'''
	# convert to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Pre process the frame
	gray = pre_process(gray)

	# Run the haar classifier to detec the eyes
	eyes = haar_classifier.detectMultiScale(gray)

	# More than two detections are obtained
	if(len(eyes)>=2):
		# Sort the the detection based on area and the detections with the greatest areas are taken as the eyes
		sorted_eye = sorted(eyes,key = lambda x:x[2]*x[3])

		# Variables to store the segmented out eyes
		left_eye = None
		right_eye = None

		# Compare the centers of the 2 largest detections(eyes)
		# And allot them as left and right depending on positon in the frame
		if((sorted_eye[-1][1] + sorted_eye[-1][3])//2 > (sorted_eye[-2][1] + sorted_eye[-2][3])//2):
			left_eye_c = sorted_eye[-1]
			right_eye_c = sorted_eye[-2]
		else:
			left_eye_c = sorted_eye[-2]
			right_eye_c = sorted_eye[-1]

		# Segment out the images based on the prediction
		left_eye = frame[left_eye_c[1]:left_eye_c[1]+left_eye_c[3],left_eye_c[0]:left_eye_c[0]+left_eye_c[2]]
		right_eye = frame[right_eye_c[1]:right_eye_c[1]+right_eye_c[3],right_eye_c[0]:right_eye_c[0]+right_eye_c[2]]

        # If the direction is right, then send the right eye with status 1 implying success
		if direction == 'right':
			return [1, [right_eye, right_eye_c[0], right_eye_c[1]]]
		# If the direction is left, then send the left eye with status 1 implying success
		elif direction == 'left':
			return [1, [left_eye, left_eye_c[0], left_eye_c[1]]]

	# If only one detection is found then that is the eye itself
	elif(len(eyes) == 1):
		# segment out the eye based on the prediction
		eye = frame[eyes[0][1]:eyes[0][1]+eyes[0][3],eyes[0][0]:eyes[0][0]+eyes[0][2]]
		return [1, [eye, eyes[0][0], eyes[0][1]]]

	# If  No eye is detected, then send status -1 implying failure
	else:
		return [-1, frame]
