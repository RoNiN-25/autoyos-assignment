import dlib
import cv2

def reset(cap, detector):
    ## reset the x-y-z platform to fit the whole face
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        cv2.imshow("MOVE BACK!!!", frame)

        if len(rects) > 0:
            break
        else:
            print('move back')
            # send a fixed backward steps value to micro-controller
    return 1
def calc_steps(dist):
    ## calculate the number of steps needed for the given motion in motor
    # pitch = 0.1cm, 1 360deg rotation -> 200steps for NEMA23 stepper
    step_dist = 2.5e-3 # cm -> Linear distance travelled in one step rotation
    num_steps = dist / step_dist
    return int(num_steps)

def calc_distance(pos, des_pos, pid_x, pid_y, prev_error_data):
    ## with the error, calculate the distance to be moved in the x-y-z axes
    '''
        des_pos -> [x_pos, y_pos]
        prev_error_data -> [error_x, error_y, sum_error_x, sum_error_y]
    '''
    error_x = des_pos[0] - pos[0]
    error_y = des_pos[1] - pos[1]

    sum_error_x = error_x + prev_error_data[2]
    sum_error_y = error_y + prev_error_data[3]

    dist_x = pid_x[0]*error_x + pid_x[1]*sum_error_x + pid_x[2]*(prev_error_data[0] - error_x)
    dist_y = pid_y[0]*error_y + pid_y[1]*sum_error_y + pid_y[2]*(prev_error_data[1] - error_y)

    data = [dist_x, dist_y, [error_x, error_y, sum_error_x, sum_error_y]]

    return data
