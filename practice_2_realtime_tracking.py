import cv2
import numpy as np

# 두 대의 IVCam 카메라 연결
cam1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 첫 번째 카메라
cam2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 두 번째 카메라

if not cam1.isOpened() or not cam2.isOpened():
    print("Failed to open one or both cameras")
    exit()

# FPS 및 해상도 설정
fps = 30
width1 = int(cam1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cam1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cam2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cam2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ROI 초기 설정 (카메라 1)
roi1_x, roi1_y, roi1_width, roi1_height = 200, 100, 80, 300
roi2_x, roi2_y, roi2_width, roi2_height = 200, 100, 80, 300

# 칼만 필터 초기화 (각 카메라별)
def init_kalman():
    kf = cv2.KalmanFilter(6, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
                                     [0, 1, 0, 1, 0, 0.5],
                                     [0, 0, 1, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 1],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.05
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf

kalman1 = init_kalman()
kalman2 = init_kalman()

# 좌표 저장 배열 생성
point_1_x = []
point_1_y = []
point_2_x = []
point_2_y = []


# 카메라1인지 카메라 2인지
flag = 0
# 이전 프레임 설정
ret1, prev_frame1 = cam1.read()
ret2, prev_frame2 = cam2.read()


def process_frame(prev_frame, curr_frame, next_frame, roi_x, roi_y, roi_width, roi_height, kalman, flag):
    roi_x = max(0, min(roi_x, width1 - roi_width))
    roi_y = max(0, min(roi_y, height1 - roi_height))

    prev_roi = prev_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    curr_roi = curr_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    next_roi = next_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    diff1 = cv2.absdiff(prev_roi, curr_roi)
    diff2 = cv2.absdiff(curr_roi, next_roi)
    combined_diff = cv2.bitwise_and(diff1, diff2)

    gray_diff = cv2.cvtColor(combined_diff, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

    cv2.rectangle(curr_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measured = None
    contours_found = False

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if w > h else h / w
        if area > 50 and 0.5 <= aspect_ratio <= 2.0:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            global_cx = cx + roi_x
            global_cy = cy + roi_y
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 32: # 스페이스바바
                if flag == 0:
                    point_1_x.append(global_cx)
                    point_1_y.append(global_cy)
                elif flag == 1:
                    point_2_x.append(global_cx)
                    point_2_y.append(global_cy)

            measured = np.array([[np.float32(global_cx)], [np.float32(global_cy)]])
            kalman.correct(measured)

            cv2.circle(curr_frame, (global_cx, global_cy), 5, (0, 0, 255), -1)
            cv2.drawContours(curr_frame, [contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)
            contours_found = True

    prediction = kalman.predict()
    predicted_cx, predicted_cy = int(prediction[0][0]), int(prediction[1][0])

    if contours_found:
        roi_x = max(0, min(predicted_cx - roi_width // 2, width1 - roi_width))
        roi_y = max(0, min(predicted_cy - roi_height // 2, height1 - roi_height))
    else:
        roi_x, roi_y = 200, 100

    return curr_frame, roi_x, roi_y


# 카메라1인지 카메라 2인지
flag = 0
# 이전 프레임 설정
ret1, prev_frame1 = cam1.read()
ret2, prev_frame2 = cam2.read()

if not ret1 or not ret2:
    print("Error: Unable to read initial frames.")
    cam1.release()
    cam2.release()
    exit()

ret1, curr_frame1 = cam1.read()
ret2, curr_frame2 = cam2.read() 

if not ret1 or not ret2:
    print("Error: Unable to read second frames.")
    cam1.release()
    cam2.release()
    exit()

# 메인 루프
while cam1.isOpened() and cam2.isOpened():
    ret1, next_frame1 = cam1.read()
    ret2, next_frame2 = cam2.read()

    if not ret1 or not ret2:
        break
    
    
    # 카메라 1, 2 각각 처리
    frame1, roi1_x, roi1_y = process_frame(prev_frame1, curr_frame1, next_frame1, roi1_x, roi1_y, roi1_width, roi1_height, kalman1, 0)
    frame2, roi2_x, roi2_y = process_frame(prev_frame2, curr_frame2, next_frame2, roi2_x, roi2_y, roi2_width, roi2_height, kalman2, 1)

    # 결과 출력
    cv2.imshow("Camera 1 Processed", frame1)
    cv2.imshow("Camera 2 Processed", frame2)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF

    if key == 97 or key == 65: # a 눌리면 배열 초기화화
        point_1_x.clear()
        point_1_y.clear()
        point_2_x.clear()
        point_2_y.clear()

    elif key == 98 or key == 66: # b 눌리면 배열 출력력
        for i in range(min(len(point_1_x), len(point_2_x))):
            print(f"x_1 = {point_1_x[i]},   y_1 = {point_1_y[i]}")
            print(f"x_2 = {point_2_x[i]},   y_2 = {point_2_y[i]}")
        
        # 정규화하기
        
        point_1_x.clear()
        point_1_y.clear()
        point_2_x.clear()
        point_2_y.clear()


    if key == 27:  # ESC 키
        break

    prev_frame1, curr_frame1 = curr_frame1, next_frame1
    prev_frame2, curr_frame2 = curr_frame2, next_frame2

# 종료
cam1.release()
cam2.release()
cv2.destroyAllWindows()
