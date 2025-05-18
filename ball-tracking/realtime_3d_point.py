import cv2
import numpy as np
import math


# 카메라 실시간으로 화면 가져옴.
cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam1.isOpened() or not cam2.isOpened():
    print("Failed to open one or both cameras")
    exit()


# 프레임 및 카메라 1,2 너비, 높이 정보 불러옴.
width1, height1 = int(cam1.get(3)), int(cam1.get(4))
width2, height2 = int(cam2.get(3)), int(cam2.get(4))


# 각 카메라의 관심영역 설정정
initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1 = 0, 60, 300, 350
initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2 = width2 - initial_roi_x_1 - initial_roi_width_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1
roi1_x, roi1_y, roi1_width, roi1_height = initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1
roi2_x, roi2_y, roi2_width, roi2_height = initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2
prev_roi1_x = roi1_x
prev_roi1_y = roi1_y

prev_roi2_x = roi2_x
prev_roi2_y = roi2_y
#관심영역 색색
roi_rect_color = (255, 0, 0)

# 화면에 처음 표시될 마크.
homeplate_x = 423
homeplate_y = 347
homeplate_width = 55
homeplate_height = 12
Dead_line = homeplate_width + homeplate_x + 5
Draw_line = True

# 칼만필터 생성하는 함수 - 초기 좌표 설정.
def init_kalman(initial_x, initial_y):
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    kalman.statePre = np.array([[initial_x], [initial_y], [0], [0]], np.float32)
    kalman.statePost = np.array([[initial_x], [initial_y], [0], [0]], np.float32)
    return kalman


# 카메라 1, 2에 사용될 칼만필터 생성
kalman1 = init_kalman(0,0)
kalman2 = init_kalman(width2,0)

#이전 야구공 좌표
prev_baseball_x1 = 0
prev_baseball_x2 = width2


ret1, prev_frame1 = cam1.read()
ret2, prev_frame2 = cam2.read()
ret1, curr_frame1 = cam1.read()
ret2, curr_frame2 = cam2.read()

fps = 60

blurred_value = 9
kernel_value = 5
diff_value = 2

ball_trace_1 = []
ball_trace_2 = []

update_roi_1 = False
update_roi_2 = False


def track_ball(prev_frame, curr_frame, next_frame, roi_x, roi_y, roi_width, roi_height, width, height, kalman, ball_trace, update_roi, value):
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

    # ROI 영역을 비디오에 표시
    cv2.rectangle(curr_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), roi_rect_color, 2)

    # Dead line 그리기
    if Draw_line:
      if value == 0:
        cv2.line(curr_frame, (Dead_line, 0), (Dead_line, height), (0, 255, 255), 2)  # 노란색 세로선

    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measured = None


    # 예측은 무조건 먼저 진행
    prediction = kalman.predict()
    predicted_cx, predicted_cy = int(prediction[0].item()), int(prediction[1].item())
    
    
    # 윤관석이 1개 이상 검출 되었는지 확인. 왜냐하면 공은 하나니깐 여러개 검출되면 처리할려고
    flag = 0
    if len(contours) > 1:
      flag = 1
    
    # 칼만 필터가 예측한 좌표와 제일 가까운 좌표 저장 변수.
    predict_x = None
    predict_y = None
    min_distance = width**2
    select_contour = None

    # 야구공 윤곽선 추출 되었는지 확인
    flag2 = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter + 1e-5)
        # 진짜 공처럼 둥근 물체만 추적
        if area > 200 and area < 360 and 0.7 < circularity < 1.2:
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue  # 해당 contour는 무시하고 다음 contour로 넘어감
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print(f"area = {area}")
            # ROI 내의 좌표를 전체 프레임의 좌표로 변환
            global_cx = cx + roi_x
            global_cy = cy + roi_y

            # 제일 가까운 좌표 구하기
            if flag == 1:
              distance = math.sqrt((global_cx - predicted_cx)**2 + (global_cy - predicted_cy)**2)
              if distance < min_distance:               
                #야구공 감지 유무 판단하는 flag
                flag2 = 1
                min_distance = distance
                predict_x = global_cx
                predict_y = global_cy
                select_contour = contour
            else:
              predict_x = global_cx
              predict_y = global_cy
              select_contour = contour
              flag2 = 1

    if flag2 == 1:
      measured = np.array([[np.float32(predict_x)], [np.float32(predict_y)]])

      # 칼만 필터에 측정값 설정
      kalman.correct(measured)
      update_roi = True

      # 공 중심점 표시
      cv2.circle(curr_frame, (predict_x, predict_y), 5, (0, 0, 255), -1)
      if select_contour is not None:
        cv2.drawContours(curr_frame, [select_contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)

    # 예측 위치에 원 그리기
    cv2.circle(curr_frame, (predicted_cx, predicted_cy), 5, (255, 255, 0), -1)  # 노란색 예측 점

    if update_roi  and 0 <= predicted_cx < width and 0 <= predicted_cy < height:
      roi_x = max(0, min(predicted_cx - roi_width // 2, width - roi_width))
      roi_y = max(0, min(predicted_cy - roi_height // 2, height - roi_height))
      roi_width, roi_height = 500, 300
    

    if ball_trace is not None:
        # 야구공 중심점 경로 그리기
        for trace_x, trace_y in ball_trace:
            cv2.circle(curr_frame, (trace_x, trace_y), 5, (255, 0, 0), -1)
    
    
    return curr_frame, roi_x, roi_y,update_roi, predict_x, predict_y


if not ret1 or not ret2:
    print("Error: Unable to read initial frames.")
    exit()

# 두 프레임을 미리 읽어서 이전 프레임과 현재 프레임을 초기화합니다.


while True:
    ret1, next_frame1 = cam1.read()
    ret2, next_frame2 = cam2.read()
    if not ret1 or not ret2:
        break
    

    frame1, roi1_x, roi1_y, update_roi_1,current_x1, current_y1 = track_ball(prev_frame1, curr_frame1, next_frame1, roi1_x, roi1_y, roi1_width, roi1_height, width1, height1, kalman1, ball_trace_1,  update_roi_1,0)
    frame2, roi2_x, roi2_y, update_roi_2, current_x2, current_y2 = track_ball(prev_frame2, curr_frame2, next_frame2, roi2_x, roi2_y, roi2_width, roi2_height, width2, height2, kalman2, ball_trace_2,  update_roi_2,1)

    #둘 다 인식되면...
    if current_x1 is not None and current_x2 is not None:
       # 만약 야구공의 x 좌표가 이상하다면?? 관심영역 초기화.
        if prev_baseball_x1 >= current_x1 or prev_baseball_x2 <= current_x2:
            roi1_x, roi1_y, roi1_width, roi1_height = initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1
            roi2_x, roi2_y, roi2_width, roi2_height = initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2
            kalman1 = init_kalman(0,0)
            kalman2 = init_kalman(width2,0)
            update_roi_1 = False
            update_roi_2 = False
            Draw_line = False
            ball_trace_1.clear()
            ball_trace_2.clear()
            prev_baseball_x1 = 0
            prev_baseball_x2 = width2
        #관심영역 업데이트
        else:
            prev_roi1_x = roi1_x
            prev_roi1_y = roi1_y
            prev_roi2_x = roi2_x
            prev_roi2_y = roi2_y
            ball_trace_1.append(current_x1, current_y1)
            ball_trace_2.append(current_x2, current_y2)

    # 둘 중 하나라도도 인식 안되면 그냥 관심영역 유지.   
    else:
       roi1_x = prev_roi1_x
       roi1_y = prev_roi1_y
       roi2_x = prev_roi2_x
       roi2_y = prev_roi2_y
       update_roi_1 = False
       update_roi_2 = False

    prev_baseball_x1 = current_x1
    prev_baseball_x2 = current_x2

    
    cv2.imshow("Camera 1 Real-Time Tracking", frame1)
    cv2.imshow("Camera 2 Real-Time Tracking", frame2)

    key = cv2.waitKey(int(1000/fps)) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('r'):
        roi1_x, roi1_y, roi1_width, roi1_height = initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1
        roi2_x, roi2_y, roi2_width, roi2_height = initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2
        kalman1 = init_kalman(0,0)
        kalman2 = init_kalman(width2,0)
        update_roi_1 = False
        update_roi_2 = False
        Draw_line = False
        ball_trace_1.clear()
        ball_trace_2.clear()
 
    
    prev_frame1, curr_frame1 = curr_frame1, next_frame1
    prev_frame2, curr_frame2 = curr_frame2, next_frame2

cam1.release()
cam2.release()
cv2.destroyAllWindows()
