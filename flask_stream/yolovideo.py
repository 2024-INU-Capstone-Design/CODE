import os
import cv2
import logging
from ultralytics.utils import LOGGER
from ultralytics import YOLO
import numpy as np
import time

# 홈플레이트 / 타자 인식 정확도 각각 표시

LOGGER.setLevel(logging.ERROR)

# flask 서버에 넘길 전역 변수 선언
latest_detection = {
    "ball": None,
    "batter": None,
    "homeplate": None,
    "is_strike": False
}

# 경로 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, "final_best.pt")

# 모델 로드
model = YOLO(path)

cap = cv2.VideoCapture(0)

def generate_frames():

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
    homeplate_detected_frames = 0
    batter_detected_frames = 0
    start_time = time.time()
    display_homeplate_acc = 0.0
    display_batter_acc = 0.0

    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            break

        total_frames += 1

        results = model(next_frame)

        homeplate_detected = False
        batter_detected = False

        for bbox, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            if conf >= 0.5:
                x1, y1, x2, y2 = map(int, bbox.tolist())
                label = "batter" if int(cls) == 0 else "homeplate"

                if label == "homeplate":
                    homeplate_detected = True
                    homeplate_box = (x1, y1, x2, y2)
                    cv2.rectangle(next_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    cv2.rectangle(next_frame, (x1, y1 - 35), (x1 + 180, y1 - 5), (0, 255, 0), -1)
                    cv2.putText(next_frame, "Homeplate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                elif label == "batter":
                    batter_detected = True
                    batter_box = (x1, y1, x2, y2)
                    cv2.rectangle(next_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cv2.rectangle(next_frame, (x1, y1 - 35), (x1 + 100, y1 - 5), (0, 0, 255), -1)
                    cv2.putText(next_frame, "Batter", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 감지되었을 경우 카운트 증가
        if homeplate_detected:
            homeplate_detected_frames += 1
        if batter_detected:
            batter_detected_frames += 1

        # 3초마다 정확도 출력
        elapsed_time = time.time() - start_time

        if elapsed_time >= 3:
            if total_frames > 0:
                display_homeplate_acc = (homeplate_detected_frames / total_frames) * 100
                display_batter_acc = (batter_detected_frames / total_frames) * 100

            print(f"\n📊 3초 요약:")
            print(f" - 홈플레이트 인식 정확도: {display_homeplate_acc:.2f}%")
            print(f" - 타자 인식 정확도: {display_batter_acc:.2f}%")
            print(f" - 총 프레임 수: {total_frames}")

            # 초기화
            total_frames = 0
            homeplate_detected_frames = 0
            batter_detected_frames = 0
            start_time = time.time()

        # 정확도 화면 출력
        cv2.rectangle(next_frame, (5, 5), (1000, 120), (0, 0, 0), -1)
        cv2.putText(next_frame, f"Homeplate Accuracy : {display_homeplate_acc:.2f}%", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(next_frame, f"Batter Accuracy     : {display_batter_acc:.2f}%", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # ============ [ 미니맵 영역 시작 ] ============

        # 1. 미니맵 사이즈 설정
        mini_map_h, mini_map_w = 150, 120
        mini_map = np.zeros((mini_map_h, mini_map_w, 3), dtype=np.uint8)

        # 2. 미니맵 스트라이크 존 (비율에 맞게 설정)
        zone_tl = (30, 30)
        zone_br = (90, 120)
        cv2.rectangle(mini_map, zone_tl, zone_br, (255, 255, 255), 2)

        # 3. 공 좌표 표시 (공 클래스 번호가 2번이라고 가정 — 네 모델에 맞게 수정)
        for bbox, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            if conf >= 0.5 and int(cls) == 2:  # ← 공 클래스 번호 바꿔도 됨
                x1, y1, x2, y2 = map(int, bbox.tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # 메인 프레임 → 미니맵 비율로 좌표 변환
                mini_x = int(cx * mini_map_w / width)
                mini_y = int(cy * mini_map_h / height)

                cv2.circle(mini_map, (mini_x, mini_y), 5, (0, 255, 0), -1)

            # 4. 미니맵을 오른쪽 아래에 붙이기
            h, w, _ = next_frame.shape
            x_offset = w - mini_map_w - 10
            y_offset = h - mini_map_h - 10
            next_frame[y_offset:y_offset+mini_map_h, x_offset:x_offset+mini_map_w] = mini_map

            # ============ [ 미니맵 영역 끝 ] ============

        # cv2.imshow("Homplate-Batter-Detection", next_frame)
        
        # ESC 키를 누르면 종료
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

        ret, buffer = cv2.imencode('.jpg', next_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        prev_frame = curr_frame
        curr_frame = next_frame
