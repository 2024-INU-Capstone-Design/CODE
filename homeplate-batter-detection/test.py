import os
import cv2
import logging
from ultralytics.utils import LOGGER
from ultralytics import YOLO
import time

# 로그 레벨 낮추기
LOGGER.setLevel(logging.ERROR)

# 모델 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "final_best.pt")
model = YOLO(model_path)

# 동영상 경로 설정
input_video_path = os.path.join(current_dir, 'ball_90.mp4')
output_video_path = os.path.join(current_dir, 'result_detect.MOV')

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:", fps)

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 정확도 계산용 변수
frame_counter = 0
homeplate_detected_frames = 0
batter_detected_frames = 0
display_homeplate_acc = 0.0
display_batter_acc = 0.0

# 3초마다 확인할 기준 프레임 수
frames_per_cycle = fps * 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    results = model(frame)

    homeplate_detected = False
    batter_detected = False

    for bbox, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        if conf >= 0.85:
            x1, y1, x2, y2 = map(int, bbox.tolist())
            label = "batter" if int(cls) == 0 else "homeplate"

            if label == "homeplate":
                homeplate_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.rectangle(frame, (x1, y1 - 35), (x1 + 180, y1 - 5), (0, 255, 0), -1)
                cv2.putText(frame, "Homeplate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            elif label == "batter":
                batter_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.rectangle(frame, (x1, y1 - 35), (x1 + 100, y1 - 5), (0, 0, 255), -1)
                cv2.putText(frame, "Batter", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if homeplate_detected:
        homeplate_detected_frames += 1
    if batter_detected:
        batter_detected_frames += 1

    # 3초치 프레임 다 봤으면 정확도 출력
    if frame_counter >= frames_per_cycle:
        display_homeplate_acc = (homeplate_detected_frames / frame_counter) * 100
        display_batter_acc = (batter_detected_frames / frame_counter) * 100

        print(f"\n📊 3초 요약:")
        print(f" - 홈플레이트 인식 정확도: {display_homeplate_acc:.2f}%")
        print(f" - 타자 인식 정확도: {display_batter_acc:.2f}%")
        print(f" - 총 프레임 수: {frame_counter}")

        # 리셋
        frame_counter = 0
        homeplate_detected_frames = 0
        batter_detected_frames = 0

    # 정확도 화면 출력
    cv2.rectangle(frame, (5, 5), (1000, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Homeplate Accuracy : {display_homeplate_acc:.2f}%", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f"Batter Accuracy     : {display_batter_acc:.2f}%", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # 저장
    out.write(frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()