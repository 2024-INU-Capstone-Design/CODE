import os
import cv2
import logging
from ultralytics.utils import LOGGER
from ultralytics import YOLO
import numpy as np
import time

# 홈플레이트 + 타자 인식 정확도 동시에 표시

LOGGER.setLevel(logging.ERROR)

# 경로 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, "final_best.pt")

# 모델 로드
model = YOLO(path)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if fps == 0:
    fps = 30

# 실시간 처리
prev_frame = None
while prev_frame is None:
    ret, prev_frame = cap.read()
    if not ret:
        print("유효한 첫 번째 프레임을 기다리는 중...")
        continue

curr_frame = None
while curr_frame is None:
    ret, curr_frame = cap.read()
    if not ret:
        print("유효한 두 번째 프레임을 기다리는 중...")
        continue

print("첫 번째와 두 번째 프레임이 성공적으로 초기화되었습니다!")

# 정확도 계산용 변수
total_frames = 0
both_detected_frames = 0
start_time = time.time()
display_accuracy = 0.0

while cap.isOpened():
    ret, next_frame = cap.read()
    if not ret:
        break

    total_frames += 1

    results = model(next_frame)

    homeplate_box = None
    batter_box = None

    for bbox, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        if conf >= 0.5:
            x1, y1, x2, y2 = map(int, bbox.tolist())
            label = "batter" if int(cls) == 0 else "homeplate"

            if label == "homeplate":
                homeplate_box = (x1, y1, x2, y2)
                cv2.rectangle(next_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.rectangle(next_frame, (x1, y1 - 35), (x1 + 180, y1 - 5), (0, 255, 0), -1)
                cv2.putText(next_frame, "Homeplate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            elif label == "batter":
                batter_box = (x1, y1, x2, y2)
                cv2.rectangle(next_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.rectangle(next_frame, (x1, y1 - 35), (x1 + 100, y1 - 5), (0, 0, 255), -1)
                cv2.putText(next_frame, "Batter", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 둘 다 감지되었을 경우 카운트 증가
    if homeplate_box and batter_box:
        both_detected_frames += 1

    # 3초마다 정확도 출력
    elapsed_time = time.time() - start_time

    if elapsed_time >= 3:
        both_accuracy = (both_detected_frames / total_frames) * 100
        display_accuracy = both_accuracy

        print(f"\n📊 3초 요약:")
        print(f" - 홈플레이트 + 타자 동시 인식 정확도: {both_accuracy:.2f}%")
        print(f" - 총 프레임 수: {total_frames}")

        # 초기화
        total_frames = 0
        both_detected_frames = 0
        start_time = time.time()
    
    cv2.rectangle(next_frame, (5, 5), (950, 90), (0, 0, 0), -1)
    cv2.putText(next_frame, f"Detection Accuracy : {display_accuracy:.2f}%", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    cv2.imshow("Homplate-Batter-Detection", next_frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev_frame = curr_frame
    curr_frame = next_frame
