import cv2
import datetime

# 두 대의 IVCam 카메라 연결
cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW) # subin
cam2 = cv2.VideoCapture(0, cv2.CAP_DSHOW) # hynseo
roi_rect_color = (255, 0, 0)

cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 카메라 확인 (하나라도 열리지 않으면 종료)
if not cam1.isOpened() and not cam2.isOpened():
    print("Failed to open one or both cameras")
    exit()

# 비디오 설정
fps = 60
width1, height1 = int(cam1.get(3)), int(cam1.get(4))
width2, height2 = int(cam2.get(3)), int(cam2.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
recording = False
out1 = None
out2 = None

Draw_line = True
Dead_line =  500
homeplate_x = 423
homeplate_y = 347
homeplate_width = 55
homeplate_height = 12

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 and not ret2:
        print("Failed to grab frames")
        break

    key = cv2.waitKey(1) & 0xFF

    # 선 및 홈플레이트 표시
    if Draw_line:
        cv2.line(frame1, (Dead_line, 0), (Dead_line, height1), (0, 255, 255), 2)
        cv2.line(frame2, (width2 - Dead_line, 0), (width2 - Dead_line, height2), (0, 255, 255), 2)

        cv2.rectangle(frame1, (homeplate_x, homeplate_y),
                      (homeplate_x + homeplate_width, homeplate_y + homeplate_height), roi_rect_color, 2)
        cv2.rectangle(frame2, (width2 - homeplate_x, homeplate_y),
                      (width2 - homeplate_x - homeplate_width, homeplate_y + homeplate_height), roi_rect_color, 2)

    # 영상 출력
    
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if key == 27:  # ESC
        break
    elif key == 32:  # 스페이스바 눌러서 녹화 시작/중지
        recording = not recording
        if recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out1 = cv2.VideoWriter(f"cam1_{timestamp}.mp4", fourcc, fps, (width1, height1))
            out2 = cv2.VideoWriter(f"cam2_{timestamp}.mp4", fourcc, fps, (width2, height2))
            print("녹화 시작")
        else:
            if out1: out1.release()
            if out1: out2.release()
            out1= None
            out2= None
            print("녹화 중지")
    elif key == ord('r'):
        Draw_line = False
        print(f"Camera 1 width: {width1}")
        print(f"Camera 1 height: {height1}")
        print(f"Camera 2 width: {width2}")
        print(f"Camera 2 height: {height2}")
    # 녹화 중일 때 저장
    if recording and out1:
        out1.write(frame1)
        out2.write(frame2)

# 종료 처리
if out1: out1.release()
if out2: out2.release()
cam1.release()
cam2.release()
cv2.destroyAllWindows()
