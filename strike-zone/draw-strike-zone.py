import os
import sys
import time
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))

# ultralytics 폴더가 없는 경우 git clone
ultralytics_dir = os.path.join(current_dir, "ultralytics")
if not os.path.exists(ultralytics_dir):
    os.system("git clone https://github.com/ultralytics/ultralytics")

# ultralytics 폴더로 이동 및 설치
os.chdir(ultralytics_dir)
os.system("pip install -e .")
sys.path.append(ultralytics_dir)

from ultralytics import YOLO

# 모델 로드
model_path = os.path.join(current_dir, "final_best.pt")
model = YOLO(model_path)

# 동영상 경로 설정
input_video_path = os.path.join(current_dir, 'IMG_9944.MOV')
output_video_path = os.path.join(current_dir, 'result_diamond.mp4')

# 동영상 열기
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 출력 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 고정 좌표 관련 변수
update_interval = 4  # 초
start_time = time.monotonic()
fixed_strike_zone = None

# 프레임 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if fixed_strike_zone is None:
        results = model(frame)
        annotated_frame = results[0].plot()

        current_batter = None
        current_homeplate = None

        for obj in results[0].boxes:
            if obj.cls == 0:  # batter
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                current_batter = (x1, y1, x2, y2)
            if obj.cls == 1:  # home plate
                x3, y3, x4, y4 = map(int, obj.xyxy[0])
                current_homeplate = (x3, y3, x4, y4)

        elapsed_time = time.monotonic() - start_time
        if current_batter and current_homeplate and elapsed_time >= update_interval:
            fixed_strike_zone = (*current_batter, *current_homeplate)
    else:
        annotated_frame = frame

    if fixed_strike_zone:
        x1, y1, x2, y2, x3, y3, x4, y4 = fixed_strike_zone
        batter_height = y2 - y1
        homeplate_width = x4 - x3
        homeplate_height = y4 - y3

        # 마름모형 스트라이크 존 그리기
        # Top
        annotated_frame = cv2.line(annotated_frame, 
            (int(x3 - homeplate_width * (2 / 43.18) + homeplate_width * 0.60), int(y2 - batter_height * 0.5635 + homeplate_height)),
            (int(x4 + homeplate_width * (2 / 43.18)), int(y2 - batter_height * 0.5635)),
            (255, 255, 255), 2)

        # Bottom
        annotated_frame = cv2.line(annotated_frame,
            (int(x3 - homeplate_width * (2 / 43.18) + homeplate_width * 0.60), int(y2 - batter_height * 0.2764 + homeplate_height)),
            (int(x4 + homeplate_width * (2 / 43.18)), int(y2 - batter_height * 0.2764)),
            (255, 255, 255), 2)

        # Left
        annotated_frame = cv2.line(annotated_frame,
            (int(x3 - homeplate_width * (2 / 43.18) + homeplate_width * 0.60), int(y2 - batter_height * 0.5635 + homeplate_height)),
            (int(x3 - homeplate_width * (2 / 43.18) + homeplate_width * 0.60), int(y2 - batter_height * 0.2764 + homeplate_height)),
            (255, 255, 255), 2)

        # Right
        annotated_frame = cv2.line(annotated_frame,
            (int(x4 + homeplate_width * (2 / 43.18)), int(y2 - batter_height * 0.5635)),
            (int(x4 + homeplate_width * (2 / 43.18)), int(y2 - batter_height * 0.2764)),
            (255, 255, 255), 2)

    out.write(annotated_frame)

cap.release()
out.release()