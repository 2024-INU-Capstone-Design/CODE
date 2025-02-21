import cv2
from ultralytics import YOLO
import numpy as np
import time

# 모델 로드 - 홈플레이트/타자 인식 모델
#model = YOLO('D:/python/best.pt')  # 경로에 맞게 수정
model = YOLO('/Users/gyuri/Documents/python/Capstone-Design/best.pt')

# 웹캠에서 비디오 스트림 불러오기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠, 다른 ID를 사용하려면 1, 2 등을 사용

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

strike_zone_active = False  # 스트라이크 존이 화면에 표시되고 있는지 여부
strike_zone_time = 0  # 스트라이크 존을 마지막으로 그린 시간

# 비디오 파일 불러오기
# cap = cv2.VideoCapture('D:/python/baseball.mp4')
# output_video_path = 'D:/python/video/resultdiff_with_kalman.mp4'

# 비디오 속성 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 출력 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# ROI (공이 출현할 것으로 예상되는 영역) 설정
#roi_x, roi_y, roi_width, roi_height = 700, 400, 100, 500
# ROI 초기 설정 (공이 출현할 것으로 예상되는 영역)
#initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height = 0, 400, 200, 500
#임시 주석
#initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height = 700, 400, 100, 500
initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height = 150, 100, 70, 300
#initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height = 600, 400, 100, 500
#initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height = 0, 0, 200, 500
roi_x, roi_y, roi_width, roi_height = initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height
roi_rect_color = (255, 0, 0)
updata_roi = False  # ROI 업데이트 여부 플래그

# 칼만 필터 초기화 (6개의 상태 변수: x, y 위치, 속도, 가속도 포함)
kalman = cv2.KalmanFilter(6, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
                                    [0, 1, 0, 1, 0, 0.5],
                                    [0, 0, 1, 0, 1, 0],
                                    [0, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]], np.float32)

# 프로세스 잡음 공분산 증가
kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1

# 측정 잡음 공분산 감소
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
kalman.errorCovPost = np.eye(6, dtype=np.float32)

# 첫 번째 프레임이 유효할 때까지 반복해서 읽기
prev_frame = None
while prev_frame is None:
    ret, prev_frame = cap.read()
    if not ret:
        print("유효한 첫 번째 프레임을 기다리는 중...")
        continue

# 두 번째 프레임 가져오기
curr_frame = None
while curr_frame is None:
    ret, curr_frame = cap.read()
    if not ret:
        print("유효한 두 번째 프레임을 기다리는 중...")
        continue

print("첫 번째와 두 번째 프레임이 성공적으로 초기화되었습니다!")

while cap.isOpened():
    # 다음 프레임 읽기
    ret, next_frame = cap.read()
    if not ret:
        #print("------------------------------------")
        #print("웹캠에서 프레임을 읽을 수 없습니다.")
        break

    # 모델로 프레임에 대한 예측 수행
    results = model(next_frame)

    homeplate_box = None
    batter_box = None

        # 탐지된 객체 확인 (홈플레이트와 타자만 저장)
    for bbox, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        if conf >= 0.2:  # 신뢰도 20% 이상
            x1, y1, x2, y2 = map(int, bbox.tolist())
            label = "batter" if int(cls) == 0 else "homeplate"

            # 홈플레이트와 타자의 바운딩 박스 저장 (화면에는 표시하지 않음)
            if label == "homeplate":
                homeplate_box = (x1, y1, x2, y2)
                print("[LOG] Homeplate detected")  # 홈플레이트 감지 로그

            elif label == "batter":
                batter_box = (x1, y1, x2, y2)
                print("[LOG] Batter detected")  # 타자 감지 로그

    # 홈플레이트와 타자가 감지된 경우 스트라이크 존 계산
    if homeplate_box and batter_box:
        hx1, hy1, hx2, hy2 = homeplate_box  # 홈플레이트 좌표
        bx1, by1, bx2, by2 = batter_box  # 타자 좌표

        # 홈플레이트 중심 좌표
        homeplate_center_x = (hx1 + hx2) // 2
        homeplate_width = hx2 - hx1

        # 타자의 신장 계산
        batter_height = by2 - by1

        # 🔹 스트라이크 존 크기 설정
        strike_zone_width = int(homeplate_width * 1.5)  # 홈플레이트 너비의 1.5배
        strike_zone_height = int(batter_height * 0.6)  # 타자 키의 60%

        # 🔹 스트라이크 존 위치 설정
        strike_zone_x1 = homeplate_center_x - strike_zone_width // 2
        strike_zone_x2 = homeplate_center_x + strike_zone_width // 2
        strike_zone_y1 = by1 + int(batter_height * 0.2)  # 타자의 20% 지점부터 시작
        strike_zone_y2 = strike_zone_y1 + strike_zone_height

        # 🔹 스트라이크 존을 10초간 유지
        current_time = time.time()
        if not strike_zone_active or (current_time - strike_zone_time > 10):
            print("[LOG] Strike zone drawn")  # 스트라이크 존 감지 로그
            strike_zone_active = True
            strike_zone_time = current_time

        # 🔹 스트라이크 존 그리기 (초록색 박스)
        cv2.rectangle(next_frame, (strike_zone_x1, strike_zone_y1), (strike_zone_x2, strike_zone_y2), (0, 255, 0), 1)
        cv2.putText(next_frame, "Strike Zone", (strike_zone_x1, strike_zone_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    
    # 스트라이크 존이 10초 이상 유지되었으면 제거
    elif strike_zone_active and (time.time() - strike_zone_time > 10):
        strike_zone_active = False  # 스트라이크 존 초기화

    # 관심 영역(ROI) 부분 추출
    prev_roi = prev_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    curr_roi = curr_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    next_roi = next_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # 연속된 프레임 차이 계산 (ROI 내에서만 수행)
    diff1 = cv2.absdiff(prev_roi, curr_roi)
    diff2 = cv2.absdiff(curr_roi, next_roi)
    combined_diff = cv2.bitwise_and(diff1, diff2)

    # 그레이스케일 변환 및 이진화
    gray_diff = cv2.cvtColor(combined_diff, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(gray_diff, 26, 255, cv2.THRESH_BINARY) # 20보다 작게 하면 더 많은 노이즈가 생김

    # ROI 영역을 비디오에 표시
    cv2.rectangle(curr_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), roi_rect_color, 2)

    # 관심 영역(ROI) 내에서 윤곽선 찾기
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measured = None  # 측정 위치 초기화
    contours_found = False

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if w > h else h / w
        # 야구공이 크면 area > 50으로
        # baseballlong은 40
        if area > 50 and 0.5 <= aspect_ratio <= 2.0:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # ROI 내의 좌표를 전체 프레임의 좌표로 변환
            global_cx = cx + roi_x
            global_cy = cy + roi_y

            measured = np.array([[np.float32(global_cx)], [np.float32(global_cy)]])
            contours_found = True

            # 칼만 필터에 측정값 설정
            kalman.correct(measured)

            # 공 중심점 표시
            cv2.circle(curr_frame, (global_cx, global_cy), 5, (0, 0, 255), -1)
            cv2.drawContours(curr_frame, [contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)

            updata_roi = True

    # 칼만 필터 예측값 계산
    prediction = kalman.predict()
    predicted_cx, predicted_cy = int(prediction[0].item()), int(prediction[1].item())

    # 윤곽선이 추출되지 않으면 초기 ROI 위치로 복구
    if not contours_found:
        roi_x, roi_y, roi_width, roi_height = initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height
        predicted_cx, predicted_cy = None, None  # 예측 원을 지우기 위해 초기화

    # 예측 위치에 원 그리기 (예측 위치가 있을 때만)
    if predicted_cx is not None and predicted_cy is not None:
        cv2.circle(curr_frame, (predicted_cx, predicted_cy), 5, (255, 255, 0), -1)  # 노란색 예측 점
    if updata_roi:
      roi_x = max(0, min(predicted_cx - 50, width - 100))
      roi_y = max(0, min(predicted_cy - 50, height - 100))
      roi_width, roi_height = 100, 100
      updata_roi = False

    # # 예측 결과를 프레임에 그리기
    # annotated_frame = results[0].plot()

    # # 특정 영역에 추가적인 사각형 그리기 (필요 시 수정)
    # cv2.rectangle(annotated_frame, (1000, 1000), (500, 500), (0, 255, 0), 10)

    # # YOLO 결과로 curr_frame 덮어쓰기
    # curr_frame = annotated_frame

    # bounding box 좌표 출력 보류
    # if results[0].boxes and len(results[0].boxes) > 0:
    #     detected = False  # 감지 여부 플래그

    #     for bbox, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
    #         if conf >= 0.3:  # 정확도가 50% 이상인 경우만 처리
    #             detected = True
    #             x1, y1, x2, y2 = map(int, bbox.tolist())
    #             label = "batter" if int(cls) == 0 else "homeplate"  # 클래스 이름 결정
    #             ## cv2.rectangle(curr_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             ## cv2.putText(curr_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
    #             ##      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #             print("감지O")
    #             print(f"객체: {label}, 정확도: {conf * 100:.2f}%, 좌표: {bbox.tolist()}")

    #     if not detected:
    #         print("감지된 객체가 없습니다.")  # 50% 이상인 객체가 없을 때 로그 출력
    # else:
    #     print("없음")  # 전체적으로 감지되지 않은 경우

    #combined_frame = cv2.addWeighted(curr_frame, 0.5, annotated_frame, 0.5, 0)

    # 결과 프레임을 화면에 표시
    cv2.imshow("Ball Tracking & Detect", curr_frame)  # 추적 결과를 실시간으로 보여줌

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 결과 프레임 저장
    # out.write(curr_frame)

    # 다음 프레임 준비
    prev_frame = curr_frame
    curr_frame = next_frame


# 비디오 캡처 및 출력 해제
# cap.release()
# out.release()

# print(f"Video saved as: {output_video_path}")
