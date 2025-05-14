import os
import cv2
import numpy as np
import math
import datetime
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from ultralytics import YOLO

# yolo on/off
yolo_enabled = False

#current_dir = os.path.dirname(os.path.abspath(__file__))
#pt_file_path = os.path.join(current_dir, 'YOLO11_batter_homeplate.pt')
model = YOLO('YOLO11_batter_homeplate.pt')

# 동영상 파일 불러오기
video_path_1 = "5-13/cam2_30fps.mp4"  # 여기에 사용하고 싶은 동영상 경로 입력
video_path_2 = "5-13/cam1_30fps.mp4"

#수빈 폰
cap1 = cv2.VideoCapture(video_path_1)
#현서 폰
cap2 = cv2.VideoCapture(video_path_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Failed to open video")
    exit()

fps = 30
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

# 이전 프레임 설정
ret1, prev_frame_1 = cap1.read()
ret1, curr_frame_1 = cap1.read()

ret2, prev_frame_2 = cap2.read()
ret2, curr_frame_2 = cap2.read()

update_roi_1 = False
update_roi_2 = False

kernel_value = 5
diff_value = 3
blurred_value = 9
#이전 좌표값
prev_x_1= width1
prev_x_2 = 0

def track_ball(prev_frame, curr_frame, next_frame, roi_x, roi_y, roi_width, roi_height, width, height, kalman, ball_trace, update_roi, value, Dead_line,prev_x):

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
        if area > 200 and area < 400:
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
    

    if flag2 == 1 and value == 0 and Dead_line < predict_x and prev_x > predict_x:
      measured = np.array([[np.float32(predict_x)], [np.float32(predict_y)]])
      prev_x = predict_x
      # 칼만 필터에 측정값 설정
      kalman.correct(measured)
      update_roi = True

      # 공 중심점 표시
      cv2.circle(curr_frame, (predict_x, predict_y), 5, (0, 0, 255), -1)
      ball_trace.append((predict_x, predict_y))
      if select_contour is not None:
        cv2.drawContours(curr_frame, [select_contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)

    elif flag2 == 1 and value == 1 and Dead_line > predict_x and prev_x < predict_x:
      measured = np.array([[np.float32(predict_x)], [np.float32(predict_y)]])
      prev_x = predict_x
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
    
    return curr_frame, roi_x, roi_y,update_roi, prev_x



polynomial_1 = None
select_point_1 = False
frame_1_points = []
Dead_line_1 = None


polynomial_2 = None
select_point_2 = False
frame_2_points = []
Dead_line_2 = None


# 직선 계산 함수
def compute_line_points(x0, y0, slope, y_min, y_max):
    y_vals = np.linspace(y_min, y_max, 1000)
    x_vals = x0 + (y_vals - y0) / slope
    return x_vals, y_vals


#3d 좌표 계산산
def calculate_3d_points(angle1_az_deg_homplate, angle1_az_deg_baseball, angle2_az_deg_homplate, angle2_az_deg_baseball,angle1_el_deg_homeplate,angle1_el_deg_baseball,angle2_el_deg_homeplate,angle2_el_deg_baseball):
    print(f"\n [CAMERA]")
    print(f"  angle1_az_deg_homeplate:  {angle1_az_deg_homplate:+.2f}°")
    print(f"  angle1_az_deg_baseball:  {angle1_az_deg_baseball:+.2f}°")
    print(f"  angle2_az_deg_homplate:  {angle2_az_deg_homplate:+.2f}°")
    print(f"  angle2_az_deg_baseball:  {angle2_az_deg_baseball:+.2f}°")
    print(f"  angle1_el_deg_homeplate:  {angle1_el_deg_homeplate:+.2f}°")
    print(f"  angle1_el_deg_baseball:  {angle1_el_deg_baseball:+.2f}°")
    print(f"  aangle2_el_deg_homeplate:  {angle2_el_deg_homeplate:+.2f}°")
    print(f"  angle2_el_deg_baseball:  {angle2_el_deg_baseball:+.2f}°")
    
    
    
    angle1_deg = 60 - abs(angle1_az_deg_homplate) + abs(angle1_az_deg_baseball)
    angle2_deg = 60 - abs(angle2_az_deg_homplate) + abs(angle2_az_deg_baseball)

    # 기울기 계산 (90 - angle), 왼쪽 카메라는 음수
    slope1 = np.tan(np.radians(90 - angle1_deg))      # 오른쪽 카메라: 양수 기울기
    slope2 = -np.tan(np.radians(90 - angle2_deg))     # 왼쪽 카메라: 음수 기울기

    # 시작점
    x1_start, y1_start = 500 * np.sqrt(3), 500   # 수빈 카메라
    x2_start, y2_start = -500 * np.sqrt(3), 500  # 현서 카메라
    
    # y값 범위 제한
    y_min, y_max = 0, 500
    # x,y 직선 좌표 계산
    x1_vals, y1_vals = compute_line_points(x1_start, y1_start, slope1, y_min, y_max)
    x2_vals, y2_vals = compute_line_points(x2_start, y2_start, slope2, y_min, y_max)
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


def visualize_3d_point(point_1, point_2):
    x, y, z = point_1
    xx, yy, zz = point_2

    # 카메라 시작점
    x1_start, y1_start, z1 = 500 * np.sqrt(3), 500, 104
    x2_start, y2_start, z2 = -500 * np.sqrt(3), 500, 104

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 카메라 위치
    ax.scatter(x1_start, y1_start, z1, color='blue', label='Subin Camera')
    ax.scatter(x2_start, y2_start, z2, color='green', label='Hyunseo Camera')

    # 카메라 높이 표시
    ax.plot([x1_start, x1_start], [y1_start, y1_start], [0, z1], color='blue', linestyle='dotted')
    ax.plot([x2_start, x2_start], [y2_start, y2_start], [0, z2], color='green', linestyle='dotted')

    # 시선 벡터
    ax.plot([x1_start, x], [y1_start, y], [z1, z], color='blue', linestyle='dotted')
    ax.plot([x2_start, x], [y2_start, y], [z2, z], color='green', linestyle='dotted')

    ax.plot([x1_start, xx], [y1_start, yy], [z1, zz], color='blue', linestyle='dotted')
    ax.plot([x2_start, xx], [y2_start, yy], [z2, zz], color='green', linestyle='dotted')

    # 교차점
    ax.scatter(x, y, z, color='purple', s=50, label='3D Intersection Point')
    ax.text(x + 10, y + 10, z + 10, f"({x:.2f}, {y:.2f}, {z:.2f})", color='purple')
    ax.scatter(xx, yy, zz, color='purple', s=50, label='3D Intersection Point')
    ax.text(xx + 10, yy + 10, zz + 10, f"({xx:.2f}, {yy:.2f}, {zz:.2f})", color='purple')
    
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


# 카메라 월드 각도 구하기
#카메라 - 홈플레이트 각도
angle1_az_deg_homplate, angle1_el_deg_homeplate = compute_angles('left', middle_point_1)
angle2_az_deg_homplate, angle2_el_deg_homeplate = compute_angles('right', middle_point_2)

yolo_enabled = True
#타자 키
min_batter_height = height2
while True:
    
    ret1, next_frame_1 = cap1.read()
    ret2, next_frame_2 = cap2.read()

    if not ret1 and not ret2:
        break
    
    
    if yolo_enabled:
        results = model(frame2, verbose=False)

        batter_box = None
        for bounding_box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            if conf >= 0.7:
                x1, y1, x2, y2 = map(int, bounding_box.tolist())
                if int(cls) == 0:  # batter만 처리
                            batter_box = (x1, y1, x2, y2)
                            # (선택) batter 시각화가 필요하면 아래 유지
                            color = (255, 0, 0)
                            text = f"batter {conf:.2f}"
                            cv2.rectangle(curr_frame_2, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(curr_frame_2, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            batter_height = y2 - y1
                            if batter_height < min_batter_height:
                                min_batter_height = batter_height
                            
    else:
        pass    # YOLO off -> pass

    if len(homeplate_points_2) == 2:
        pt1 = homeplate_points_2[0][0], homeplate_points_2[0][1]
        pt2 = homeplate_points_2[1][0], homeplate_points_2[1][1]

                # Top
        cv2.line(curr_frame_2, (pt1[0], pt1[1] - int(min_batter_height * 0.2764)), (pt2[0], pt2[1] - int(min_batter_height * 0.2764)), (255, 255, 255), 2)
        
        # Bottom
        cv2.line(curr_frame_2, (pt1[0], pt1[1] - int(min_batter_height * 0.5635)), (pt2[0], pt2[1] - int(min_batter_height * 0.5635)), (255, 255, 255), 2)

        # Left
        cv2.line(curr_frame_2, (pt1[0], pt1[1] - int(min_batter_height * 0.5635)), (pt1[0], pt1[1] - int(min_batter_height * 0.2764)), (255, 255, 255), 2)
        
        # Right
        cv2.line(curr_frame_2, (pt2[0], pt2[1] - int(min_batter_height * 0.5635)), (pt2[0], pt2[1] - int(min_batter_height * 0.2764)), (255, 255, 255), 2)
    
    Dead_line_1 = homeplate_points_1[0][0]
    Dead_line_2 = homeplate_points_2[1][0]

    #관심영역 활성화화
    if roi_1_enable:
        frame1, roi_x_1, roi_y_1, update_roi_1, prev_x_1 = track_ball(prev_frame_1, curr_frame_1, next_frame_1, roi_x_1, roi_y_1, roi_width_1, roi_height_1, width1, height1, kalman_1, ball_trace_1, update_roi_1, 0, Dead_line_1, prev_x_1) 
    
    #관심영역 활성화화
    if roi_2_enable:
        frame2, roi_x_2, roi_y_2, update_roi_2, prev_x_2 = track_ball(prev_frame_2, curr_frame_2, next_frame_2, roi_x_2, roi_y_2, roi_width_2, roi_height_2, width2, height2, kalman_2, ball_trace_2, update_roi_2, 1, Dead_line_2, prev_x_2) 
    
    
    
    key = cv2.waitKey(int(1000/fps)) & 0xFF

    # 3차 다항식 그리기기
    if polynomial_1:
            frame1 = draw_predicted_trajectory(curr_frame_1, polynomial_1,width1,height1)

    # 3차 다항식 그리기기
    if polynomial_2:
            frame2 = draw_predicted_trajectory(curr_frame_2, polynomial_2,width2,height2)
    

    # 카메라 - 야구공 각도
    angle1_az_deg_baseball_1, angle1_el_deg_baseball_1 = None, None
    angle1_az_deg_baseball_2, angle1_el_deg_baseball_2 = None, None
    angle2_az_deg_baseball_1, angle2_el_deg_baseball_1 = None, None
    angle2_az_deg_baseball_2, angle2_el_deg_baseball_2 = None, None

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
        #해당 좌표 월드 각도 계산
        # 카메라 - 야구공 각도
        angle1_az_deg_baseball_1, angle1_el_deg_baseball_1 = compute_angles('left', [(x_value_1, y_value_1)])
        angle1_az_deg_baseball_2, angle1_el_deg_baseball_2 = compute_angles('left', [(x_value_2, y_value_2)])
 
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
        # 카메라 - 야구공 각도
        angle2_az_deg_baseball_1, angle2_el_deg_baseball_1 = compute_angles('right', [(x_value_1, y_value_1)])
        angle2_az_deg_baseball_2, angle2_el_deg_baseball_2 = compute_angles('right', [(x_value_2, y_value_2)])
    
    # 월드 좌표계에서의의 각도 (degree)
    if angle1_az_deg_baseball_1 is not None and angle2_az_deg_baseball_1 is not None:    
        point_1 = calculate_3d_points(angle1_az_deg_homplate, angle1_az_deg_baseball_1, angle2_az_deg_homplate, angle2_az_deg_baseball_1,angle1_el_deg_homeplate,angle1_el_deg_baseball_1,angle2_el_deg_homeplate,angle2_el_deg_baseball_1)

    # 월드 좌표계에서의의 각도 (degree)
    if angle1_az_deg_baseball_2 is not None and angle2_az_deg_baseball_2 is not None:    
        point_2 = calculate_3d_points(angle1_az_deg_homplate, angle1_az_deg_baseball_2, angle2_az_deg_homplate, angle2_az_deg_baseball_2,angle1_el_deg_homeplate,angle1_el_deg_baseball_2,angle2_el_deg_homeplate,angle2_el_deg_baseball_2)



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
        ball_trace_1.clear()
        prev_x_1 = width1
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
        ball_trace_2.clear()
        prev_x_2 = 0
    elif key == ord('v'):
        # ball_trace에 저장된 좌표로 3차 다항식 회귀
        polynomial_2 = polyfit_3rd_degree(ball_trace_2)
        roi_2_enable = False
        ball_trace_2.clear()
    elif key == ord('k'):   
        if point_1 and point_2:
            visualize_3d_point(point_1, point_2)       
    elif key == ord('y'):
        # yolo on/off
        if yolo_enabled:
            yolo_enabled = False

        else:
            yolo_enabled = True
            min_batter_height = height2
    elif key == ord('i'):
        #홈플레이트 좌표 초기화 -> 만약 카메라가 넘어져서 처음부터 시작할 때 누르기.
        homeplate_points_1 = None
        homeplate_points_2 = None
    cv2.imshow("Camera 1 Real-Time Tracking", frame1)
    cv2.imshow("Camera 2 Real-Time Tracking", frame2)

    prev_frame_1, curr_frame_1 = curr_frame_1, next_frame_1
    prev_frame_2, curr_frame_2 = curr_frame_2, next_frame_2
cap1.release()
cap2.release()

cv2.destroyAllWindows()