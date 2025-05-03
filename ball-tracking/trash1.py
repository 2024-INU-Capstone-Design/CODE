import cv2
import numpy as np
import math

# 비디오 파일 불러오기
video_path = 'C:/Users/ASUS/OneDrive/바탕 화면/realtime/university_ground_seokjin.mp4'  # 여기에서 테스트할 비디오 파일 경로를 입력하세요.
cap = cv2.VideoCapture(video_path)

# 비디오 속성 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ROI (공이 출현할 것으로 예상되는 영역) 설정

# my 캠
initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height = width - 200, 50, 500, 500
roi_x, roi_y, roi_width, roi_height = initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height

roi_rect_color = (255, 0, 0)
updata_roi = False # ROI 업데이트 여부 플래그


# 칼만 필터 초기화
kalman = cv2.KalmanFilter(4, 2)  # 4개의 상태 변수, 2개의 측정 변수 (x, y 위치)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
kalman.errorCovPost = np.eye(4, dtype=np.float32)
initial_x = width
initial_y = 0

kalman.statePre = np.array([[initial_x], [initial_y], [0], [0]], np.float32)
kalman.statePost = np.array([[initial_x], [initial_y], [0], [0]], np.float32)

# 첫 두 프레임 읽기
ret, prev_frame = cap.read()
if not ret:
    print("Error: Unable to read video file.")
    cap.release()
    exit()

ret, curr_frame = cap.read()
if not ret:
    print("Error: Unable to read second frame.")
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
    blurred = cv2.GaussianBlur(gray_diff, (9, 9), 0)
    _, roi_thresh = cv2.threshold(blurred, 3, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel)
    roi_thresh = cv2.dilate(roi_thresh, kernel, iterations=1)
    
    # ROI 영역을 비디오에 표시
    cv2.rectangle(curr_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), roi_rect_color, 2)

    # 칼만 필터 예측값 계산
    prediction = kalman.predict()
    predicted_cx, predicted_cy = int(prediction[0].item()), int(prediction[1].item())
    
    # 관심 영역(ROI) 내에서 윤곽선 찾기
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measured = None  # 측정 위치 초기화

    # 윤관석이 1개 이상 검출 되었는지 확인. 왜냐하면 공은 하나니깐 여러개 검출되면 처리할려고
    flag = 0
    if len(contours) > 1:
      flag = 1
      

    # 칼만 필터가 예측한 좌표와 제일 가까운 좌표 저장 변수.
    predict_x = 0
    predict_y = 0
    min_distance = 100.0
    select_contour = None

    # 야구공 윤곽선 추출 되었는지 확인
    flag2 = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # 야구공이 크면 area > 50으로
        if area > 600 and area < 800:
            flag2 = 1
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            print(f"area = {area}")
            # ROI 내의 좌표를 전체 프레임의 좌표로 변환
            global_cx = cx + roi_x
            global_cy = cy + roi_y

            # 제일 가까운 좌표 구하기
            if flag == 1:
              distance = math.sqrt((global_cx - predicted_cx)**2 + 0.8 * (global_cy - predicted_cy)**2)
              if distance < min_distance:
                min_distance = distance
                predict_x = global_cx
                predict_y = global_cy
                select_contour = contour
            else:
              predict_x = global_cx
              predict_y = global_cy
              select_contour = contour


    if flag2 == 1:
      measured = np.array([[np.float32(predict_x)], [np.float32(predict_y)]])

      # 칼만 필터에 측정값 설정
      kalman.correct(measured)
      updata_roi = True

      # 공 중심점 표시
      cv2.circle(curr_frame, (predict_x, predict_y), 5, (0, 0, 255), -1)
      if select_contour is not None:
        cv2.drawContours(curr_frame, [select_contour + (roi_x, roi_y)], -1, (0, 255, 0), 2)


    
    # 예측 위치에 원 그리기
    cv2.circle(curr_frame, (predicted_cx, predicted_cy), 5, (255, 255, 0), -1)  # 노란색 예측 점

    if updata_roi:
      roi_x = max(0, min(predicted_cx - roi_width // 2, width - roi_width))
      roi_y = max(0, min(predicted_cy - roi_height // 2, height - roi_height))

      roi_width, roi_height = 500, 500
      

    

     # 결과를 실시간으로 화면에 출력
    cv2.imshow("Real-Time Ball Tracking", curr_frame)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('r'):
        roi_x, roi_y, roi_width, roi_height = initial_roi_x, initial_roi_y, initial_roi_width, initial_roi_height
        predicted_cx, predicted_cy = initial_x, initial_y
        updata_roi = False
         # 칼만 필터 초기화
        kalman.statePre = np.zeros((4, 1), np.float32)
        kalman.statePost = np.zeros((4, 1), np.float32)
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        kalman.statePre = np.array([[initial_x], [initial_y], [0], [0]], np.float32)
        kalman.statePost = np.array([[initial_x], [initial_y], [0], [0]], np.float32)
    # 다음 프레임 준비
    prev_frame = curr_frame
    curr_frame = next_frame

# 자원 해제
cap.release()
cv2.destroyAllWindows()

