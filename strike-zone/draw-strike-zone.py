import os
import sys
import time
import cv2

# 현재 Directory에 YOLO 모델 설치 및 설정
os.system("git clone https://github.com/ultralytics/ultralytics")

# ultralytics 폴더로 이동
os.chdir("ultralytics")

# 현재 위치에 "editable" 설정으로 패키지 설치
os.system("pip install -e .")

# ultralytics 모듈을 인식할 수 있도록 Windows 환경에 설치 경로 추가
sys.path.append("C:/Users/USER/Desktop/ABS Challenge/CODE/strike-zone/ultralytics")

from ultralytics import YOLO

# model 불러와서 사용하기 (가중치 파일)
model = YOLO(r'C:\Users\USER\Desktop\ABS Challenge\CODE\strike-zone\yolo11_best.pt')

# baseball2 동영상으로
# 입력 및 출력 동영상 경로 설정
# input_video_path = r'C:\Users\USER\Desktop\ABS Challenge\CODE\strike-zone\baseball.mp4'
# output_video_path = r'C:\Users\USER\Desktop\ABS Challenge\CODE\strike-zone\result.mp4'

# baseball 동영상으로
# 입력 및 출력 동영상 경로 설정
input_video_path = r'C:\Users\USER\Desktop\ABS Challenge\CODE\strike-zone\baseball2.mp4'
output_video_path = r'C:\Users\USER\Desktop\ABS Challenge\CODE\strike-zone\result2.mp4'

# 동영상 파일 로드
cap = cv2.VideoCapture(input_video_path)

# 동영상 파일의 속성 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 결과를 저장할 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 좌표 고정 주기 (초 단위)
update_interval = 4
start_time = time.monotonic()

# 고정된 스트라이크 존 좌표 (초기화)
fixed_strike_zone = None

# 동영상 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 모델로 스트라이크 존이 고정되지 않았을 때만 예측 수행
    if fixed_strike_zone is None:
        # 모델로 프레임에 대한 예측 수행
        results = model(frame)
        annotated_frame = results[0].plot()

        # 현재 프레임에서 수집된 좌표
        current_batter = None
        current_homeplate = None

        # 매 프레임마다 좌표 수집
        for obj in results[0].boxes:
            if obj.cls == 0:  # 'batter' 클래스만 필터링
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                current_batter = (x1, y1, x2, y2)

            if obj.cls == 1:  # 'home plate' 클래스만 필터링
                x3, y3, x4, y4 = map(int, obj.xyxy[0])
                current_homeplate = (x3, y3, x4, y4)

        # 고정 시간(초)이 경과했는지 확인
        elapsed_time = time.monotonic() - start_time
        if current_batter and current_homeplate and elapsed_time >= update_interval:
            # 고정된 스트라이크 존 좌표 설정
            fixed_strike_zone = (*current_batter, *current_homeplate)
    else:
        # YOLO를 중지하고, 고정된 스트라이크 존만 그리기
        annotated_frame = frame

    # 고정된 스트라이크 존 그리기
    if fixed_strike_zone:
        x1, y1, x2, y2, x3, y3, x4, y4 = fixed_strike_zone

        # 스트라이크 존 그리기
        annotated_frame = cv2.line(annotated_frame, (int(x3-(x4-x3)*(2/43.18)), int(y2-(y2-y1)*(0.5635))), 
                                   (int(x4+(x4-x3)*(2/43.18)), int(y2-(y2-y1)*(0.5635))), (255, 255, 255), 2)
        annotated_frame = cv2.line(annotated_frame, (int(x3-(x4-x3)*(2/43.18)), int(y2-(y2-y1)*(0.2764))), 
                                   (int(x4+(x4-x3)*(2/43.18)), int(y2-(y2-y1)*(0.2764))), (255, 255, 255), 2)
        annotated_frame = cv2.line(annotated_frame, (int(x3-(x4-x3)*(2/43.18)), int(y2-(y2-y1)*(0.5635))), 
                                   (int(x3-(x4-x3)*(2/43.18)), int(y2-(y2-y1)*(0.2764))), (255, 255, 255), 2)
        annotated_frame = cv2.line(annotated_frame, (int(x4+(x4-x3)*(2/43.18)), int(y2-(y2-y1)*(0.5635))), 
                                   (int(x4+(x4-x3)*(2/43.18)), int(y2-(y2-y1)*(0.2764))), (255, 255, 255), 2)

    # 결과 프레임 저장
    out.write(annotated_frame)

# 자원 해제
cap.release()
out.release()