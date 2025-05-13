import os
import cv2
import numpy as np
import math
import datetime

# 동영상 파일 불러오기
video_path_1 = "subin_640_480.mp4"  # 여기에 사용하고 싶은 동영상 경로 입력
video_path_2 = "hynseo_640_480.mp4"

cap1 = cv2.VideoCapture(video_path_1)
cap2 = cv2.VideoCapture(video_path_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Failed to open video")
    exit()

fps = 60
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 커서 및 포인트 설정
cursor_x_1, cursor_y_1 = 150, 350
homeplate_points_1 = []   # 홈플레이트 양 끝 좌표 저장
middle_point_1 = []       # 홈플레이트 양 끝 점의 중앙점
move_step = 2

cursor_x_2, cursor_y_2 = width2 - 150, 350
homeplate_points_2 = []   # 홈플레이트 양 끝 좌표 저장
middle_point_2 = []       # 홈플레이트 양 끝 점의 중앙점

initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1 = width1 - 300, 60, 300, 350
roi_x_1, roi_y_1, roi_width_1, roi_height_1 = initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1
roi_rect_color = (255, 0, 0)

initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2 = 0, 60, 300, 350
roi_x_2, roi_y_2, roi_width_2, roi_height_2 = initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2

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

kalman_1 = init_kalman(width1, 0)
ball_trace_1 = []

kalman_2 = init_kalman(0, 0)
ball_trace_2 = []

#관심영역 활성화
roi_1_enable = True
roi_2_enable = True

# 야구공 궤적 예측하기 위해 2차 다항식 회귀
def polyfit_3rd_degree(ball_trace):
    if len(ball_trace) < 3:
        print("적어도 3개의 좌표가 필요합니다.")
        return None

    # x, y 좌표 분리
    x_vals = np.array([point[0] for point in ball_trace])
    y_vals = np.array([point[1] for point in ball_trace])

    # 2차 다항식 회귀 (degree=3)
    coeffs = np.polyfit(x_vals, y_vals, 3)

    # 회귀 방정식 생성
    polynomial = np.poly1d(coeffs)

    return polynomial

# 현재 프레임에 궤적 그리기
def draw_predicted_trajectory(frame, polynomial, width, height):
    # 예측된 x 좌표로 y값 계산
    x_vals = np.linspace(0, width, 100)
    y_vals = polynomial(x_vals)

    for x, y in zip(x_vals, y_vals):
        # 좌표가 이미지 크기 내에 있으면 그리기
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)  # 초록색 점으로 궤적 그리기

    return frame


# 홈플레이트 점을 기록하는 함수
def handle_cursor_and_homeplate(key, cursor_x, cursor_y, move_step, width, height, homeplate_points, middle_point):
    # 커서 이동 처리
    if key == ord('a'):  # 왼쪽
        cursor_x = max(cursor_x - move_step, 0)
    elif key == ord('w'):  # 위쪽
        cursor_y = max(cursor_y - move_step, 0)
    elif key == ord('d'):  # 오른쪽
        cursor_x = min(cursor_x + move_step, width - 1)
    elif key == ord('s'):  # 아래쪽
        cursor_y = min(cursor_y + move_step, height - 1)

    # Enter 키를 눌러 홈플레이트 점 기록
    if key == 13:  # Enter
        if len(homeplate_points) < 2:
            homeplate_points.append((cursor_x, cursor_y))
            print("홈플레이트 점:", homeplate_points)
            if len(homeplate_points) == 2:
                middle_point.append((int((homeplate_points[0][0] + homeplate_points[1][0]) / 2),
                                    int((homeplate_points[0][1] + homeplate_points[1][1]) / 2)))
                print("스트라이크 존 고정됨")
                print("중앙점", middle_point)
        else:
            print("이미 점 2개 선택됨")

    return cursor_x, cursor_y, homeplate_points, middle_point


# 이전 프레임 설정
ret1, prev_frame_1 = cap1.read()
ret1, curr_frame_1 = cap1.read()

ret2, prev_frame_2 = cap2.read()
ret2, curr_frame_2 = cap2.read()

update_roi_1 = False
update_roi_2 = False

kernel_value = 5
diff_value = 2

def track_ball(prev_frame, curr_frame, next_frame, roi_x, roi_y, roi_width, roi_height, width, height, kalman, ball_trace, update_roi, value, Dead_line):

    prev_roi = prev_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    curr_roi = curr_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    next_roi = next_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    diff1 = cv2.absdiff(prev_roi, curr_roi)
    diff2 = cv2.absdiff(curr_roi, next_roi)
    combined_diff = cv2.bitwise_and(diff1, diff2)

    gray_diff = cv2.cvtColor(combined_diff, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray_diff, (blurred_value, blurred_value), 0)
    _, roi_thresh = cv2.threshold(gray_diff, diff_value, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_value, kernel_value))
    roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel)
    roi_thresh = cv2.dilate(roi_thresh, kernel, iterations=1)

    # ROI 영역을 비디오에 표시
    cv2.rectangle(curr_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), roi_rect_color, 2)

    # 칼만 필터 예측값 계산

    prediction = kalman.predict()
    predicted_cx, predicted_cy = int(prediction[0].item()), int(prediction[1].item())
    
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measured = None

    # 윤관석이 1개 이상 검출 되었는지 확인. 왜냐하면 공은 하나니깐 여러개 검출되면 처리할려고
    flag = 0
    if len(contours) > 1:
      flag = 1
    
    # 칼만 필터가 예측한 좌표와 제일 가까운 좌표 저장 변수.
    predict_x = 0
    predict_y = 0
    min_distance = 1000.0
    select_contour = None

    # 야구공 윤곽선 추출 되었는지 확인
    flag2 = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        # 야구공이 크면 area > 50으로
        if area > 70 and area < 150:
            flag2 = 1
            M = cv2.moments(contour)
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
                min_distance = distance
                predict_x = global_cx
                predict_y = global_cy
                select_contour = contour
            else:
              predict_x = global_cx
              predict_y = global_cy
              select_contour = contour

    if flag2 == 1 and value == 0 and Dead_line < predict_x:
      measured = np.array([[np.float32(predict_x)], [np.float32(predict_y)]])

      # 칼만 필터에 측정값 설정
      kalman.correct(measured)
      update_roi = True

      # 공 중심점 표시
      cv2.circle(curr_frame, (predict_x, predict_y), 5, (0, 0, 255), -1)
      ball_trace.append((predict_x, predict_y))
      if select_contour is not None:
        cv2.drawContours(curr_frame, [select_contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)

    elif flag2 == 1 and value == 1 and Dead_line > predict_x:
      measured = np.array([[np.float32(predict_x)], [np.float32(predict_y)]])

      # 칼만 필터에 측정값 설정
      kalman.correct(measured)
      update_roi = True

      # 공 중심점 표시
      cv2.circle(curr_frame, (predict_x, predict_y), 5, (0, 0, 255), -1)
      ball_trace.append((predict_x, predict_y))
      if select_contour is not None:
        cv2.drawContours(curr_frame, [select_contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)




    # 예측 위치에 원 그리기
    cv2.circle(curr_frame, (predicted_cx, predicted_cy), 5, (255, 255, 0), -1)  # 노란색 예측 점

    if update_roi  and 0 <= predicted_cx < width and 0 <= predicted_cy < height:
      roi_x = max(0, min(predicted_cx - roi_width // 2, width - roi_width))
      roi_y = max(0, min(predicted_cy - roi_height // 2, height - roi_height))
      roi_width, roi_height = 400, 300
    

    if ball_trace is not None:
        # 야구공 중심점 경로 그리기
        for trace_x, trace_y in ball_trace:
            cv2.circle(curr_frame, (trace_x, trace_y), 5, (0, 255, 0), -1)
    
    return curr_frame, roi_x, roi_y,update_roi



polynomial_1 = None
select_point_1 = False
frame_1_points = []
Dead_line_1 = None


polynomial_2 = None
select_point_2 = False
frame_2_points = []
Dead_line_2 = None



# 이 부분 이제 2개의 point가 찍힐때까지 대기하는 부분
while len(homeplate_points_1) < 2:
    # 커서와 홈플레이트 기록 대기
    frame1 = curr_frame_1.copy()
    key = cv2.waitKey(1) & 0xFF
    cursor_x_1, cursor_y_1, homeplate_points_1, middle_point_1 = handle_cursor_and_homeplate(key, cursor_x_1, cursor_y_1, move_step, width1, height1, homeplate_points_1, middle_point_1)
    # 커서 표시
    cv2.circle(frame1, (cursor_x_1, cursor_y_1), 3, (0, 0, 255), -1)  # 빨간색 원으로 커서
    cv2.imshow("Camera 1 Real-Time Tracking", frame1)
    

while len(homeplate_points_2) < 2:
    # 커서와 홈플레이트 기록 대기
    frame2 = curr_frame_2.copy()
    key = cv2.waitKey(1) & 0xFF
    cursor_x_2, cursor_y_2, homeplate_points_2, middle_point_2 = handle_cursor_and_homeplate(key, cursor_x_2, cursor_y_2, move_step, width2, height2, homeplate_points_2, middle_point_2)
    # 커서 표시
    cv2.circle(frame2, (cursor_x_2, cursor_y_2), 3, (0, 0, 255), -1)  # 빨간색 원으로 커서
    cv2.imshow("Camera 1 Real-Time Tracking", frame2)



while True:

    ret1, next_frame_1 = cap1.read()
    ret2, next_frame_2 = cap2.read()

    if not ret1 and not ret2:
        break
    
    Dead_line_1 = homeplate_points_1[0][0]
    Dead_line_2 = homeplate_points_2[1][0]

    #관심영역 활성화화
    if roi_1_enable:
        frame1, roi_x_1, roi_y_1, update_roi_1 = track_ball(prev_frame_1, curr_frame_1, next_frame_1, roi_x_1, roi_y_1, roi_width_1, roi_height_1, width1, height1, kalman_1, ball_trace_1, update_roi_1, 0, Dead_line_1) 
    
    #관심영역 활성화화
    if roi_2_enable:
        frame2, roi_x_2, roi_y_2, update_roi_2 = track_ball(prev_frame_2, curr_frame_2, next_frame_2, roi_x_2, roi_y_2, roi_width_2, roi_height_2, width2, height2, kalman_2, ball_trace_2, update_roi_2, 1, Dead_line_2) 
    
    
    
    key = cv2.waitKey(int(1000/fps)) & 0xFF

    # 3차 다항식 그리기기
    if polynomial_1:
            frame1 = draw_predicted_trajectory(curr_frame_1, polynomial_1,width1,height1)

    # 3차 다항식 그리기기
    if polynomial_2:
            frame2 = draw_predicted_trajectory(curr_frame_2, polynomial_2,width2,height2)

    #홈플레이트 끝 점을 지나는 야구공 좌표들.
    if homeplate_points_1 and polynomial_1 is not None and polynomial_1(int(homeplate_points_1[0][0])) is not None:
        x_value_1 = int(homeplate_points_1[0][0])  # 첫 번째 홈플레이트 점의 x좌표
        y_value_1 = polynomial_1(x_value_1)  # polynomial_1에서 y값 계산
        frame_1_points.append((x_value_1, y_value_1))  # x, y 좌표를 리스트에 추가
        cv2.circle(frame1, (int(x_value_1), int(y_value_1)), 3, (0, 0, 255), -1)  # 빨간색 원으로 커서
        x_value_2 = int(homeplate_points_1[1][0])  # 첫 번째 홈플레이트 점의 x좌표
        y_value_2 = polynomial_1(x_value_2)  # polynomial_1에서 y값 계산
        frame_1_points.append((x_value_2, y_value_2))  # x, y 좌표를 리스트에 추가
        cv2.circle(frame1, (int(x_value_2), int(y_value_2)), 3, (0, 0, 255), -1)  # 빨간색 원으로 커서
        # 홈플레이트의 두 점과 예측된 점을 직선으로 연결
        cv2.line(frame1, (int(x_value_1), int(y_value_1)), (int(homeplate_points_1[0][0]), int(homeplate_points_1[0][1])), (255, 255, 255), 2)  # 흰색 직선
        cv2.line(frame1, (int(x_value_2), int(y_value_2)), (int(homeplate_points_1[1][0]), int(homeplate_points_1[1][1])), (255, 255, 255), 2)  # 흰색 직선

 
    #홈플레이트 끝 점을 지나는 야구공 좌표들.
    if homeplate_points_2 and polynomial_2 is not None and polynomial_2(int(homeplate_points_2[0][0])) is not None:
        x_value_1 = int(homeplate_points_2[0][0])  # 첫 번째 홈플레이트 점의 x좌표
        y_value_1 = polynomial_2(x_value_1)  # polynomial_1에서 y값 계산
        frame_2_points.append((x_value_1, y_value_1))  # x, y 좌표를 리스트에 추가
        cv2.circle(frame2, (int(x_value_1), int(y_value_1)), 3, (0, 0, 255), -1)  # 빨간색 원으로 커서
        x_value_2 = int(homeplate_points_2[1][0])  # 첫 번째 홈플레이트 점의 x좌표
        y_value_2 = polynomial_2(x_value_2)  # polynomial_1에서 y값 계산
        frame_2_points.append((x_value_2, y_value_2))  # x, y 좌표를 리스트에 추가
        cv2.circle(frame2, (int(x_value_2), int(y_value_2)), 3, (0, 0, 255), -1)  # 빨간색 원으로 커서
        # 홈플레이트의 두 점과 예측된 점을 직선으로 연결
        cv2.line(frame2, (int(x_value_1), int(y_value_1)), (int(homeplate_points_2[0][0]), int(homeplate_points_2[0][1])), (255, 255, 255), 2)  # 흰색 직선
        cv2.line(frame2, (int(x_value_2), int(y_value_2)), (int(homeplate_points_2[1][0]), int(homeplate_points_2[1][1])), (255, 255, 255), 2)  # 흰색 직선


    if key == 27:  # ESC
        break
    elif key == ord('z'):
        roi_x_1, roi_y_1, roi_width_1, roi_height_1 = initial_roi_x_1, initial_roi_y_1, initial_roi_width_1, initial_roi_height_1
        kalman_1 = init_kalman(width1,0)
        update_roi_1 = False
        #관심영역 활성화화
        roi_1_enable = True
        #3차 다항식 제거거
        polynomial_1 = None
    elif key == ord('x'):
        # ball_trace에 저장된 좌표로 3차 다항식 회귀
        polynomial_1 = polyfit_3rd_degree(ball_trace_1)
        roi_1_enable = False
        ball_trace_1.clear()    
    elif key == ord('c'):         
        roi_x_2, roi_y_2, roi_width_2, roi_height_2 = initial_roi_x_2, initial_roi_y_2, initial_roi_width_2, initial_roi_height_2
        kalman_2 = init_kalman(0,0)
        update_roi_2 = False
        #관심영역 활성화화
        roi_2_enable = True
        #3차 다항식 제거거
        polynomial_2 = None
    elif key == ord('v'):
        # ball_trace에 저장된 좌표로 3차 다항식 회귀
        polynomial_2 = polyfit_3rd_degree(ball_trace_2)
        roi_2_enable = False
        ball_trace_2.clear()
    elif key == ord('k'):   
        #홈플레이트 좌표 초기화 -> 만약 카메라가 넘어져서 처음부터 시작할 때 누르기.
        homeplate_points_1 = None
    
    cv2.imshow("Camera 1 Real-Time Tracking", frame1)
    cv2.imshow("Camera 2 Real-Time Tracking", frame2)
    prev_frame_1, curr_frame_1 = curr_frame_1, next_frame_1
    prev_frame_2, curr_frame_2 = curr_frame_2, next_frame_2
cap1.release()
cap2.release()

cv2.destroyAllWindows()
