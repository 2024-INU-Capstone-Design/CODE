#
# Merging all functional modules for the low-cost Automated Ball-Strike (ABS) system
#
# 1 Detect batter and home plate using YOLOv11
# 2 Draw the strike zone based on bounding boxes
# 3 Tracing baseball
# 4 Estimate the baseball's 3D position
#   4-1 Calculate direction vectors
#   4-2 Triangulation

import os
import cv2
import datetime
import numpy as np
from ultralytics import YOLO

# pt 파일 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
pt_file_path = os.path.join(current_dir, './YOLO_model/YOLO11_batter_homeplate.pt')
model = YOLO(pt_file_path)

# iVCam 카메라 연결
cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 비디오 설정
fps = 60
width1 = int(cam1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cam1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
display_scale = 1.0

recording = False
out1 = None

# 커서 및 포인트 설정
cursor_x, cursor_y = 300, 600
homeplate_points = []   # 홈플레이트 양 끝 좌표 저장
middle_point = []       # 홈플레이트 양 끝 점의 중앙점
move_step = 5

batter_height = 0

# 추가1
yolo_enabled = True

while cam1.isOpened():
    ret1, frame = cam1.read()
    if not ret1:
        break

    if yolo_enabled:
        results = model(frame, verbose=False)
        
        batter_box = None
        for bounding_box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            if conf >= 0.7:
                x1, y1, x2, y2 = map(int, bounding_box.tolist())
                if int(cls) == 0:  # batter만 처리
                    batter_box = (x1, y1, x2, y2)
                    # (선택) batter 시각화
                    color = (255, 0, 0)
                    text = f"batter {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    batter_height = y2 - y1
    else:
        # YOLO off 일 때
        pass
        
    # 홈플레이트 영역 수동 저장 (2개 점이 선택됐을 때 사각형 처리)
    if len(homeplate_points) == 2:
        pt1 = homeplate_points[0]
        pt2 = homeplate_points[1]

        # Top
        cv2.line(frame, (pt1[0], pt1[1] - int(batter_height * 0.2764)),
                 (pt2[0], pt2[1] - int(batter_height * 0.2764)), (255, 255, 255), 2)
        # Bottom
        cv2.line(frame, (pt1[0], pt1[1] - int(batter_height * 0.5635)),
                 (pt2[0], pt2[1] - int(batter_height * 0.5635)), (255, 255, 255), 2)
        # Left
        cv2.line(frame, (pt1[0], pt1[1] - int(batter_height * 0.5635)),
                 (pt1[0], pt1[1] - int(batter_height * 0.2764)), (255, 255, 255), 2)
        # Right
        cv2.line(frame, (pt2[0], pt2[1] - int(batter_height * 0.5635)),
                 (pt2[0], pt2[1] - int(batter_height * 0.2764)), (255, 255, 255), 2)

    # 기준점(middle_point) 시각화
    if middle_point:
        cv2.circle(frame, middle_point[0], 5, (0, 255, 255), -1)
        cv2.putText(frame, f"REF ({middle_point[0][0]}, {middle_point[0][1]})", 
                    (middle_point[0][0] + 10, middle_point[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 커서 표시
    cv2.circle(frame, (cursor_x, cursor_y), 3, (0, 0, 255), -1)

    # 화면 출력
    small_frame1 = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
    cv2.imshow("Camera 1", small_frame1)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('y'):
        if yolo_enabled:
            print("이미 YOLO가 켜져 있음")
        else:
            yolo_enabled = True
            homeplate_points = []
            middle_point = []
            print("YOLO ON + 홈플레이트 점 초기화")
    elif key == 32:  # Space
        recording = not recording
        if recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out1 = cv2.VideoWriter(f"cam1_{timestamp}.mp4", fourcc, fps, (width1, height1))
            print("녹화 시작")
        else:
            out1.release()
            print("녹화 중지")
    elif key == ord('a'): cursor_x = max(cursor_x - move_step, 0)
    elif key == ord('w'): cursor_y = max(cursor_y - move_step, 0)
    elif key == ord('d'): cursor_x = min(cursor_x + move_step, width1 - 1)
    elif key == ord('s'): cursor_y = min(cursor_y + move_step, height1 - 1)
    elif key == 13:  # Enter
        if len(homeplate_points) < 2:
            homeplate_points.append((cursor_x, cursor_y))
            print("홈플레이트 점:", homeplate_points)
            if len(homeplate_points) == 2:
                mid_x = int((homeplate_points[0][0] + homeplate_points[1][0]) / 2)
                mid_y = int((homeplate_points[0][1] + homeplate_points[1][1]) / 2)
                middle_point.append((mid_x, mid_y))
                print("스트라이크 존 고정됨, YOLO OFF")
                print("중앙점 (기준점):", (mid_x, mid_y))
                yolo_enabled = False
        else:
            print("이미 점 2개 선택됨")

    if recording:
        out1.write(frame)

# 종료
if out1:
    out1.release()
cam1.release()
cv2.destroyAllWindows()
