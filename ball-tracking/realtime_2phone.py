import cv2
import numpy as np
import math

cam1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cam1.isOpened() or not cam2.isOpened():
    print("Failed to open one or both cameras")
    exit()

fps = 30
width1, height1 = int(cam1.get(3)), int(cam1.get(4))
width2, height2 = int(cam2.get(3)), int(cam2.get(4))

initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1 = 0, 0, 100, 500
initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2 = 0, 0, 100, 500
roi1_x, roi1_y, roi1_width, roi1_height = initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1
roi2_x, roi2_y, roi2_width, roi2_height = initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2

def init_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32)*0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman

kalman1 = init_kalman()
kalman2 = init_kalman()

ret1, prev_frame1 = cam1.read()
ret2, prev_frame2 = cam2.read()
ret1, curr_frame1 = cam1.read()
ret2, curr_frame2 = cam2.read()

blurred_value = 9
kernel_value = 9
diff_value = 19
area_size = 50

if not ret1 or not ret2:
    print("Error: Unable to read initial frames.")
    exit()

# 두 프레임을 미리 읽어서 이전 프레임과 현재 프레임을 초기화합니다.


def track_ball(prev_frame, curr_frame, next_frame, roi_x, roi_y, roi_width, roi_height, kalman, width, height, initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height):
    prev_roi = prev_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    curr_roi = curr_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    next_roi = next_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    diff1 = cv2.absdiff(prev_roi, curr_roi)
    diff2 = cv2.absdiff(curr_roi, next_roi)
    combined_diff = cv2.bitwise_and(diff1, diff2)

    gray_diff = cv2.cvtColor(combined_diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_diff, (blurred_value, blurred_value), 0)
    _, roi_thresh = cv2.threshold(blurred, diff_value, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_value, kernel_value))
    roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel)
    roi_thresh = cv2.dilate(roi_thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measured = None

    prediction = kalman.predict()
    predicted_cx, predicted_cy = int(prediction[0]), int(prediction[1])
    
    non_contours = 0

    if len(contours) == 1 and cv2.contourArea(contours[0]) > area_size:
        contour = contours[0]
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) + roi_x
        cy = int(M['m01'] / M['m00']) + roi_y
        measured = np.array([[np.float32(cx)], [np.float32(cy)]])
        kalman.correct(measured)
        cv2.circle(curr_frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.drawContours(curr_frame, [contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)

    elif len(contours) > 1:
        min_distance = float('inf')
        closest_contour = None
        for contour in contours:
            if cv2.contourArea(contour) > area_size:
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00']) + roi_x
                cy = int(M['m01'] / M['m00']) + roi_y
                distance = math.sqrt((predicted_cx - cx)**2 + 0.8 * (predicted_cy - cy)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = (cx, cy, contour) 

        if closest_contour:
            cx, cy, contour = closest_contour
            measured = np.array([[np.float32(cx)], [np.float32(cy)]])
            kalman.correct(measured)
            cv2.circle(curr_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.drawContours(curr_frame, [contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)

    else:
        non_contours = 1

    cv2.circle(curr_frame, (predicted_cx, predicted_cy), 5, (255, 255, 0), -1)

    if non_contours == 0:
        roi_x = max(0, min(predicted_cx - roi_width // 2, width - roi_width))
        roi_y = max(0, min(predicted_cy - roi_height // 2, height - roi_height))
    else:
        roi_x, roi_y, roi_width, roi_height = initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height

    cv2.rectangle(curr_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    return curr_frame, roi_x, roi_y

while True:
    ret1, next_frame1 = cam1.read()
    ret2, next_frame2 = cam2.read()
    if not ret1 or not ret2:
        break

    frame1, roi1_x, roi1_y = track_ball(prev_frame1, curr_frame1, next_frame1, roi1_x, roi1_y, roi1_width, roi1_height, kalman1, width1, height1, initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1)
    frame2, roi2_x, roi2_y = track_ball(prev_frame2, curr_frame2, next_frame2, roi2_x, roi2_y, roi2_width, roi2_height, kalman2, width2, height2, initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2)

    cv2.imshow("Camera 1 Real-Time Tracking", frame1)
    cv2.imshow("Camera 2 Real-Time Tracking", frame2)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
        break
    
    change_flag = 0

    key = cv2.waitKey(int(1000 / fps)) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('d'):
        blurred_value -= 2
        kernel_value -= 2
        if blurred_value < 1:
            blurred_value = 1
        if kernel_value < 1:
            kernel_value = 1
        change_flag = 1
    elif key == ord('u'):
        blurred_value += 2
        kernel_value += 2
        if blurred_value > 9:
            blurred_value = 9
        if kernel_value > 9:
            kernel_value = 9
        change_flag = 1
    elif key == ord('s'):
        area_size -= 3
        if area_size < 20:
            area_size = 20
        change_flag = 1
    elif key == ord('w'):
        area_size += 3
        if area_size > 60:
            area_size = 60
        change_flag = 1
    
    
    if change_flag == 1:
        print(f"blurred_value: {blurred_value}, kernel_value: {kernel_value}, area_size: {area_size}")

    prev_frame1, curr_frame1 = curr_frame1, next_frame1
    prev_frame2, curr_frame2 = curr_frame2, next_frame2

cam1.release()
cam2.release()
cv2.destroyAllWindows()
