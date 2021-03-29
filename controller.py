#################### Module for the controller ####################
# inport the required packages
import dlib
import cv2

# Function to reset the position of the x-y-z platform to fit the whole face
def reset(cap, detector):
    '''
        Input:
            cap -> cv2 Video Capture object
            detector -> dlib's face detector
        Output:
            status as 1
    '''
    while True:
        # read the frame
        _, frame = cap.read()
        # convert frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect the faces
        rects = detector(gray, 0)
        # if a face is detected, then exit else, keep moving the camera backwards until face is detected
        if len(rects) > 0:
            break
        else:
            print('move back')
            # send a fixed backward steps value to micro-controller
    return 1

# Function to convert the linear distance into steps of the stepper motor
def calc_steps(dist):
    '''
        Input:
            dist -> Linear distance in cm
        Output:
            Number of steps to rotate
            negative implies opposite direction
    '''
    ## calculate the number of steps needed for the given motion in motor
    # pitch = 0.1cm, 1 360deg rotation -> 200steps for NEMA23 stepper
    step_dist = 2.5e-3 # cm -> Linear distance travelled in one step rotation
    num_steps = dist / step_dist
    return int(num_steps)

# Function to run the PID controller to get the distance to be moved
def calc_distance(pos, des_pos, pid_x, pid_y, prev_error_data):
    '''
        Input:
            pos -> the current position of the eye
            des_pos -> desired position of the eye
            pid_x -> PID gains for the x axis
            pid_y -> PID gains for the y axis
            prev_error_data -> Error data of the previous iteration - [error_x, error_x, sum_error_x, sum_error_y]
        Output:
            The distance to be travelled in the x and y axes and the error data for the next iteration
    '''

    # calculate the errors in the x and y axis
    error_x = des_pos[0] - pos[0]
    error_y = des_pos[1] - pos[1]

    # Get the sum of error for the integral component of the PID controller
    sum_error_x = error_x + prev_error_data[2]
    sum_error_y = error_y + prev_error_data[3]

    # Calculate the distance to be traveled using the PID controller
    #( Gains are the be tuned based on the hardware)
    dist_x = pid_x[0]*error_x + pid_x[1]*sum_error_x + pid_x[2]*(prev_error_data[0] - error_x)
    dist_y = pid_y[0]*error_y + pid_y[1]*sum_error_y + pid_y[2]*(prev_error_data[1] - error_y)

    # populate the data to be returned
    data = [dist_x, dist_y, [error_x, error_y, sum_error_x, sum_error_y]]

    return data
