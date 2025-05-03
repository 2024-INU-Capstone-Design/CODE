import os
import cv2
import logging
from ultralytics.utils import LOGGER
from ultralytics import YOLO
import time

# 홈플레이트 / 타자 인식 정확도 각각 표시 (프레임단위)
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

    # YOLO 모델로 객체 감지 수행
    results = model(frame)

    # YOLO 기본 스타일의 바운딩 박스
    annotated_frame = results[0].plot()

    homeplate_detected = False
    batter_detected = False

    # 클래스 및 confidence 정보만 추출하여 검출 여부만 판단
    for bbox, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        if conf >= 0.6:
            label = "batter" if int(cls) == 0 else "homeplate"
            if label == "homeplate":
                homeplate_detected = True
            elif label == "batter":
                batter_detected = True

    # 프레임 카운트
    if homeplate_detected:
        homeplate_detected_frames += 1
    if batter_detected:
        batter_detected_frames += 1

    # 3초마다 정확도 계산
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
    # Homeplate Accuracy: 하늘색
    cv2.putText(annotated_frame, f"Homeplate Accuracy : {display_homeplate_acc:.2f}%", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 102), 2)

    # Batter Accuracy: 파란색
    cv2.putText(annotated_frame, f"Batter Accuracy     : {display_batter_acc:.2f}%", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    ## 창 크기 고정 + 영상 표시
    cv2.namedWindow("Homplate-Batter-Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Homplate-Batter-Detection", 800, 600)  # 필요시 크기 조정
    cv2.imshow("Homplate-Batter-Detection", annotated_frame)

    # 저장
    out.write(annotated_frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()