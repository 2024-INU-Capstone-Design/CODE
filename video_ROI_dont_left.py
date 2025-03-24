import cv2
import numpy as np


# 웹캠 입력 설정 (0은 기본 웹캠)
cap = cv2.VideoCapture(0)

# 비디오 파일 경로
# video_path = 'C:/baseball.mp4'  # 여기에서 테스트할 비디오 파일 경로를 입력하세요.
# cap = cv2.VideoCapture(video_path)

# 이전 global_cx 값 초기화
previous_global_cx = None

# 웹캠 속성 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ROI 초기 설정 (공이 출현할 것으로 예상되는 영역)
#initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height = 700, 400, 100, 500
initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height = 200, 100, 80, 300
roi_x, roi_y, roi_width, roi_height = initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height
roi_rect_color = (255, 0, 0)
updata_roi = False  # ROI 업데이트 여부 플래그

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

# 프로세스 잡음 공분산 증가
kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1

# 측정 잡음 공분산 감소
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
kalman.errorCovPost = np.eye(6, dtype=np.float32)

# 첫 프레임 읽기
ret, prev_frame = cap.read()
if not ret:
    print("Error: Unable to access the webcam.")
    cap.release()
    exit()

ret, curr_frame = cap.read()
if not ret:
    print("Error: Unable to access the webcam.")
    cap.release()
    exit()

while cap.isOpened():
    # 다음 프레임 읽기
    ret, next_frame = cap.read()
    if not ret:
        break

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
    _, roi_thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

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
        if area > 50 and 0.5 <= aspect_ratio <= 2.0:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # ROI 내의 좌표를 전체 프레임의 좌표로 변환
            global_cx = cx + roi_x
            global_cy = cy + roi_y
             # ROI가 왼쪽으로 이동하지 않도록 조건 추가
            # 이전 global_cx와 비교하여 왼쪽으로 이동하면 무시
            if previous_global_cx is None or global_cx >= previous_global_cx + 10:
                measured = np.array([[np.float32(global_cx)], [np.float32(global_cy)]])
                contours_found = True
                previous_global_cx = global_cx
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
        predicted_cx, predicted_cy = None, None
     # ROI 업데이트 (왼쪽으로 가지 않도록 설정)
    if updata_roi and predicted_cx is not None and predicted_cx > roi_x:
        roi_x = max(0, min(predicted_cx - 50, width - 100))
        roi_y = max(0, min(predicted_cy - 50, height - 100))
        roi_width, roi_height = 100, 100
        updata_roi = False


    # 결과를 실시간으로 화면에 출력
    cv2.imshow("Real-Time Ball Tracking", curr_frame)

    delay = int(1000 / fps)  # FPS에 기반한 프레임 딜레이 계산
    if cv2.waitKey(delay) & 0xFF == 27:  # ESC 키
        break
    # 다음 프레임 준비
    prev_frame = curr_frame
    curr_frame = next_frame

# 자원 해제
cap.release()
cv2.destroyAllWindows()