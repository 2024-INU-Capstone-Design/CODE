import cv2
import numpy as np
from ultralytics import YOLO

# 모델 로드 (로컬에 있는 모델 경로로 변경!)
model = YOLO('C:/Users/ASUS/OneDrive/바탕 화면/realtime/final_best.pt')

# 입력/출력 동영상 경로
input_video_path = 'C:/Users/ASUS/OneDrive/바탕 화면/realtime/IMG_9944.MOV'
output_video_path = 'C:/Users/ASUS/OneDrive/바탕 화면/realtime/Draw_strike_zone.mp4'

# 동영상 열기
cap = cv2.VideoCapture(input_video_path)

# 속성 읽기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 결과 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 예측
    results = model(frame)
    annotated_frame = results[0].plot()

    batter_height = None
    homeplate_width = None
    homeplate_height = None

    for obj in results[0].boxes:
        cls = int(obj.cls.item())
        x1, y1, x2, y2 = map(int, obj.xyxy[0])

        if cls == 0:  # batter
            batter_height = y2 - y1
            batter_bottom = y2

        if cls == 1:  # home plate
            homeplate_width = x2 - x1
            homeplate_height = y2 - y1
            homeplate_left = x1
            homeplate_right = x2

    if batter_height and homeplate_width and homeplate_height:
        # 스트라이크 존 좌표 계산
        top_y = int(batter_bottom - batter_height * 0.5635 + homeplate_height)
        bottom_y = int(batter_bottom - batter_height * 0.2764 + homeplate_height)
        left_x = int(homeplate_left - homeplate_width * (2 / 43.18) + homeplate_width * 0.60)
        right_x = int(homeplate_right + homeplate_width * (2 / 43.18))

        # 스트라이크 존 선 그리기
        cv2.line(annotated_frame, (left_x, top_y), (right_x, top_y), (255, 255, 255), 2)   # Top
        cv2.line(annotated_frame, (left_x, bottom_y), (right_x, bottom_y), (255, 255, 255), 2)  # Bottom
        cv2.line(annotated_frame, (left_x, top_y), (left_x, bottom_y), (255, 255, 255), 2)  # Left
        cv2.line(annotated_frame, (right_x, top_y), (right_x, bottom_y), (255, 255, 255), 2)  # Right

    # 결과 저장
    out.write(annotated_frame)

# 자원 해제
cap.release()
out.release()
print(" 분석 및 저장 완료:", output_video_path)
