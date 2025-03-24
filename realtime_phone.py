import cv2
import numpy as np

# IVCam 연결 (디바이스 ID 확인 후 변경)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Unable to open video capture device.")
    cap.release()
    exit()

# FPS, 해상도 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:  # FPS 값이 유효하지 않으면 기본값 설정
    fps = 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ROI 초기 설정 (공이 출현할 것으로 예상되는 영역)
initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height = 200, 100, 80, 300
roi_x, roi_y, roi_width, roi_height = initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height
roi_rect_color = (255, 0, 0)  # ROI 테두리 색상
update_roi = False  # ROI 업데이트 플래그

# 칼만 필터 초기화
kalman = cv2.KalmanFilter(6, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
                                    [0, 1, 0, 1, 0, 0.5],
                                    [0, 0, 1, 0, 1, 0],
                                    [0, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.05  # 프로세스 잡음 공분산
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0  # 측정 잡음 공분산
kalman.errorCovPost = np.eye(6, dtype=np.float32)

# 이전 프레임 초기화
ret, prev_frame = cap.read()
if not ret:
    print("Error: Unable to read initial frame.")
    cap.release()
    exit()

ret, curr_frame = cap.read()
if not ret:
    print("Error: Unable to read second frame.")
    cap.release()
    exit()

# 메인 루프
while cap.isOpened():
    ret, next_frame = cap.read()
    if not ret:
        break

    # 관심 영역(ROI) 설정
    roi_x = max(0, min(roi_x, width - roi_width))
    roi_y = max(0, min(roi_y, height - roi_height))

    # ROI 추출
    prev_roi = prev_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    curr_roi = curr_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    next_roi = next_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # 연속된 프레임 차이 계산
    diff1 = cv2.absdiff(prev_roi, curr_roi)
    diff2 = cv2.absdiff(curr_roi, next_roi)
    combined_diff = cv2.bitwise_and(diff1, diff2)

    # 그레이스케일 변환 및 이진화
    gray_diff = cv2.cvtColor(combined_diff, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

    # ROI 영역을 비디오에 표시
    cv2.rectangle(curr_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), roi_rect_color, 2)

    # 윤곽선 감지
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measured = None
    contours_found = False

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if w > h else h / w
        if area > 50 and 0.3 <= aspect_ratio <= 3.0:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # ROI 내 좌표를 전체 프레임 좌표로 변환
            global_cx = cx + roi_x
            global_cy = cy + roi_y
            measured = np.array([[np.float32(global_cx)], [np.float32(global_cy)]])
            kalman.correct(measured)

            # 공 중심점 표시
            cv2.circle(curr_frame, (global_cx, global_cy), 5, (0, 0, 255), -1)
            cv2.drawContours(curr_frame, [contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)
            contours_found = True
            update_roi = True

    # 칼만 필터 예측값
    prediction = kalman.predict()
    predicted_cx, predicted_cy = int(prediction[0]), int(prediction[1])

    # ROI 업데이트
    if update_roi and contours_found:
        roi_x = max(0, min(predicted_cx - roi_width // 2, width - roi_width))
        roi_y = max(0, min(predicted_cy - roi_height // 2, height - roi_height))
        update_roi = False

    # 결과를 화면에 출력
    cv2.imshow("Real-Time Ball Tracking", curr_frame)

    # FPS 딜레이 및 종료 조건
    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:  # ESC 키
        break

    # 다음 프레임 준비
    prev_frame = curr_frame
    curr_frame = next_frame

# 자원 해제
cap.release()
cv2.destroyAllWindows()
