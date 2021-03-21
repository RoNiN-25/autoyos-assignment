# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import cv2
import numpy as np

def pre_process(frame):
	kernel = np.ones((5,5),np.uint8) # kernel for erosion and dilation
	frame = cv2.erode(frame,kernel,iterations=3)
	frame = cv2.dilate(frame,kernel,iterations=3)
	return frame

# Create the classifier object
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	frame = pre_process(frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray scale

	# detect the eyes
	eyes = eye_cascade.detectMultiScale(gray)

	if(len(eyes)>=2):
		sorted_eye = sorted(eyes,key = lambda x:x[2]*x[3])

		left_eye = None
		right_eye = None
		if(sorted_eye[-1][1] < sorted_eye[-2][1]):
			left_eye_c = sorted_eye[-1]
			right_eye_c = sorted_eye[-2]
		else:
			left_eye_c = sorted_eye[-2]
			right_eye_c = sorted_eye[-1]


		left_eye = frame[left_eye_c[1]:left_eye_c[1]+left_eye_c[3],left_eye_c[0]:left_eye_c[0]+left_eye_c[2]]
		right_eye = frame[right_eye_c[1]:right_eye_c[1]+right_eye_c[3],right_eye_c[0]:right_eye_c[0]+right_eye_c[2]]

		#(ex,ey) -> first co-ordinate of the bounding box
		#(ew,eh) -> width and height of the bounding box
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


		# show the frame
		cv2.imshow("Frame", frame)
		cv2.imshow("Left Eye", left_eye)
		cv2.imshow("Right Eye", right_eye)

	elif(len(eyes) == 1):
		eye = frame[eyes[0][1]:eyes[0][1]+eyes[0][3],eyes[0][0]:eyes[0][0]+eyes[0][2]]
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


		# show the frame
		cv2.imshow("Frame", frame)
		cv2.imshow("Eye", eye)
	else:
		print("No eye detected")

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
