import cv2
import numpy as np
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


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

# Dead_line 설정
Dead_line = 480
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
      else:
         cv2.line(curr_frame, (width - Dead_line, 0), (width - Dead_line, height), (0, 255, 255), 2)  # 노란색 세로선
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

#홈플레이트 커서 및 좌표
cursor_x_1, cursor_y_1 = width1 - 150, 350
homeplate_points_1 = []
middle_point_1 = []
move_step = 2

# 내 폰폰
cursor_x_2, cursor_y_2 = 150, 350
homeplate_points_2 = []   # 홈플레이트 양 끝 좌표 저장
middle_point_2 = []       # 홈플레이트 양 끝 점의 중앙점

def capture_frame(frame, cursor_x, cursor_y, move_step, width, height, homeplate_points, middle_point ):
    while len(homeplate_points) < 2:
        display_frame = frame.copy()
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        cursor_x, cursor_y, homeplate_points, middle_point = handle_cursor_and_homeplate(key, cursor_x, cursor_y, move_step, width, height, homeplate_points, middle_point)
        cv2.circle(display_frame, (cursor_x, cursor_y), 3, (0, 0, 255), -1)  # 빨간색 원으로 커서
        cv2.imshow("Camera 1 Real-Time Tracking", display_frame)


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


# 1. 카메라 내부 파라미터 + 왜곡 계수
def get_K_dist(side):
    if side == 'left':
        K = np.array([[1777.3987, 0, 643.4449],
                      [0, 1775.0710, 370.0626],
                      [0, 0, 1]], dtype=np.float32)
        dist = np.array([0.02623, 0.1816, 0, 0, 0], dtype=np.float32)
    else:
        K = np.array([[1707.5227, 0, 640.9289],
                      [0, 1707.6282, 343.4517],
                      [0, 0, 1]], dtype=np.float32)
        dist = np.array([0.17445, -0.5164, 0, 0, 0], dtype=np.float32)
    return K, dist


# 2. 픽셀 좌표 → 방향 벡터(cam 기준)
def pixel_to_cam_dir(u, v, K, dist):
    pt = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.undistortPoints(pt, K, dist, P=None)
    x_norm, y_norm = und[0, 0]
    dir_cam = np.array([x_norm, -y_norm, 1.0], dtype=np.float32)  # y축 뒤집기 (y-up 보정)
    return dir_cam / np.linalg.norm(dir_cam)


# 3. 방향벡터 → 좌우/위아래 각도 계산
def compute_angle_from_center(cam_dir):
    azimuth_rad = np.arctan2(cam_dir[0], cam_dir[2])  # 좌우
    elevation_rad = np.arctan2(cam_dir[1], np.sqrt(cam_dir[0]**2 + cam_dir[2]**2))  # 위아래
    return np.degrees(azimuth_rad), np.degrees(elevation_rad)


# 4. 추출된 좌표를 통해 방향벡터를 구하고 좌우/위아래 각도 계산
def compute_angles(side, points):
    K, dist = get_K_dist(side)
    u, v = points[0]
    cam_dir = pixel_to_cam_dir(u, v, K, dist)
    az_deg, el_deg = compute_angle_from_center(cam_dir)
    return az_deg, el_deg


#3d 좌표 계산산
def calculate_3d_points(angle1_az_deg_homplate, angle1_az_deg_baseball, angle2_az_deg_homplate, angle2_az_deg_baseball,angle1_el_deg_homeplate,angle1_el_deg_baseball,angle2_el_deg_homeplate,angle2_el_deg_baseball):
    # print(f"\n [CAMERA]")
    # print(f"  angle1_az_deg_homeplate:  {angle1_az_deg_homplate:+.2f}°")
    # print(f"  angle1_az_deg_baseball:  {angle1_az_deg_baseball:+.2f}°")
    # print(f"  angle2_az_deg_homplate:  {angle2_az_deg_homplate:+.2f}°")
    # print(f"  angle2_az_deg_baseball:  {angle2_az_deg_baseball:+.2f}°")
    # print(f"  angle1_el_deg_homeplate:  {angle1_el_deg_homeplate:+.2f}°")
    # print(f"  angle1_el_deg_baseball:  {angle1_el_deg_baseball:+.2f}°")
    # print(f"  aangle2_el_deg_homeplate:  {angle2_el_deg_homeplate:+.2f}°")
    # print(f"  angle2_el_deg_baseball:  {angle2_el_deg_baseball:+.2f}°")
    
    
    angle1_deg = 60 + abs(angle1_az_deg_homplate) - abs(angle1_az_deg_baseball)
    angle2_deg = 60 + abs(angle2_az_deg_homplate) - abs(angle2_az_deg_baseball)

    # 기울기 계산 (90 - angle), 왼쪽 카메라는 음수
    slope1 = np.tan(np.radians(90 - angle1_deg))      # 오른쪽 카메라: 양수 기울기
    slope2 = -np.tan(np.radians(90 - angle2_deg))     # 왼쪽 카메라: 음수 기울기

    # 시작점
    x1_start, y1_start = 500 * np.sqrt(3), 500   # 수빈 카메라
    x2_start, y2_start = -500 * np.sqrt(3), 500  # 현서 카메라
    
    # x,y 교차점 계산
    numerator = (slope1 * x1_start - y1_start) - (slope2 * x2_start - y2_start)
    denominator = slope1 - slope2
    
    # z값 각도 (degree)
    angle1_between = abs(angle1_el_deg_homeplate) - 5.94 # 여기서 5.94는 이제 tan104/1000 계산값값  
    angle1_baseball = abs(angle1_el_deg_baseball) - abs(angle1_between)

    final_target_height = 0
    target_z = 0

    if abs(denominator) < 1e-6:
        intersection_point = None
    else:
        x_intersect = numerator / denominator
        y_intersect = slope1 * (x_intersect - x1_start) + y1_start
        intersection_point = (x_intersect, y_intersect)
        # 거리
        distance = math.sqrt((x1_start - intersection_point[0])**2 + (y1_start - intersection_point[1])**2)
        target_height = math.tan(math.radians(angle1_baseball)) * distance
        final_target_height = 104 - target_height
        target_z = final_target_height
    
    return intersection_point[0], intersection_point[1], target_z

def visualize_3d_point(point_3d):     
    
    # ===== 시각화 =====
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # 카메라 시작점
    x1_start, y1_start, z1 = 500 * np.sqrt(3), 500, 104
    x2_start, y2_start, z2 = -500 * np.sqrt(3), 500, 104

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 카메라 위치
    ax.scatter(x1_start, y1_start, z1, color='blue', label='Subin Camera')
    ax.scatter(x2_start, y2_start, z2, color='green', label='Hyunseo Camera')
    # 3D 점 시각화
    for a,b,c in point_3d:
        ax.scatter(a, b, c, color='red', s=50, label='Detected Ball')
    
    # 홈플레이트 좌표
    plate = [[21.6, 0, 0], [-21.6, 0, 0], [21.6, -30.48, 0], [-21.6, -30.48, 0], [0, -45.98, 0]]
    ax.plot([plate[0][0], plate[1][0]], [plate[0][1], plate[1][1]], [0, 0], color='orange', linewidth=2)
    ax.plot([plate[0][0], plate[2][0]], [plate[0][1], plate[2][1]], [0, 0], color='orange', linewidth=2)
    ax.plot([plate[1][0], plate[3][0]], [plate[1][1], plate[3][1]], [0, 0], color='orange', linewidth=2)
    ax.plot([plate[2][0], plate[4][0]], [plate[2][1], plate[4][1]], [0, 0], color='orange', linewidth=2)
    ax.plot([plate[3][0], plate[4][0]], [plate[3][1], plate[4][1]], [0, 0], color='orange', linewidth=2)
    plate_face = [plate[0], plate[1], plate[3], plate[4], plate[2]]
    homeplate_poly = Poly3DCollection([plate_face], color='orange', alpha=0.3)
    ax.add_collection3d(homeplate_poly)

    # 축 설정
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title('3D Ball Position Visualization')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


#3d 좌표 저장
point_3d = []

ret1, prev_frame1 = cam1.read()
ret2, prev_frame2 = cam2.read()
ret1, curr_frame1 = cam1.read()
ret2, curr_frame2 = cam2.read()

if not ret1 or not ret2:
    print("Error: Unable to read initial frames.")
    exit()

# 두 프레임을 미리 읽어서 이전 프레임과 현재 프레임을 초기화합니다.
angle1_az_deg_homplate, angle1_el_deg_homeplate = None, None
angle2_az_deg_homplate, angle2_el_deg_homeplate = None, None

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
            # 2 카메라가 Dead_line까지 공을 추적했다면...
            if current_x1 >= Dead_line or current_x2 <= width2 - Dead_line:
               #3각 측량법 시행해야지
               print("삼각측량법")

               if ball_trace_1 is not None:
                    for point1, point2 in zip(ball_trace_1, ball_trace_2):
                        angle1_az_deg_baseball, angle1_el_deg_baseball = compute_angles('left', [(point2[0], point2[1])])
                        angle2_az_deg_baseball, angle2_el_deg_baseball = compute_angles('right', [(point1[0], point1[1])])
                        points = calculate_3d_points(angle1_az_deg_homplate, angle1_az_deg_baseball, angle2_az_deg_homplate, angle2_az_deg_baseball,angle1_el_deg_homeplate,angle1_el_deg_baseball,angle2_el_deg_homeplate,angle2_el_deg_baseball)
                        point_3d.append(points)

                    # 저장된 3d 좌표 시각화.
                    visualize_3d_point(point_3d)
                

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
    #관심영역 수동으로 초기화 하는 키키
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
        point_3d.clear()

    #홈플레이트 좌표 찍는 키키
    elif key == ord('t'):
        homeplate_points_1.clear()
        homeplate_points_2.clear()
        middle_point_1.clear()
        middle_point_2.clear()
        capture_frame(frame1, cursor_x_1, cursor_y_1, move_step, width1, height1, homeplate_points_1, middle_point_1 )
        capture_frame(frame2, cursor_x_2, cursor_y_2, move_step, width2, height2, homeplate_points_2, middle_point_2 )
        angle1_az_deg_homplate, angle1_el_deg_homeplate = compute_angles('left', middle_point_2)
        angle2_az_deg_homplate, angle2_el_deg_homeplate = compute_angles('right', middle_point_1)
        


    prev_frame1, curr_frame1 = curr_frame1, next_frame1
    prev_frame2, curr_frame2 = curr_frame2, next_frame2

cam1.release()
cam2.release()
cv2.destroyAllWindows()
