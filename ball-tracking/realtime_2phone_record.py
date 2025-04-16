import cv2
import datetime

# 두 대의 IVCam 카메라 연결
cam1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 첫 번째 카메라
cam2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 두 번째 카메라

if not cam1.isOpened() or not cam2.isOpened():
    print("Failed to open one or both cameras")
    exit()

# 비디오 설정
fps = 30
width1, height1 = int(cam1.get(3)), int(cam1.get(4))
width2, height2 = int(cam2.get(3)), int(cam2.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

recording = False
out1, out2 = None, None

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("Failed to grab frames")
        break

    # 두 개의 카메라 영상을 나란히 보여줌
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 32:
        recording = not recording
        if recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out1 = cv2.VideoWriter(f"cam1_{timestamp}.mp4", fourcc, fps, (width1, height1))
            out2 = cv2.VideoWriter(f"cam2_{timestamp}.mp4", fourcc, fps, (width2, height2))
            print("녹화 시작")
        else:
            out1.release()
            out2.release()
            print("녹화 중지")

    # 녹화 중일 때 저장
    if recording:
        out1.write(frame1)
        out2.write(frame2)

# 종료 처리
if out1: out1.release()
if out2: out2.release()  
cam1.release()
cam2.release()
cv2.destroyAllWindows()
