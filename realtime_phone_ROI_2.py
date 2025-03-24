import cv2

# 두 대의 IVCam 카메라 연결
cam1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 첫 번째 카메라
cam2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 두 번째 카메라

if not cam1.isOpened() or not cam2.isOpened():
    print("Failed to open one or both cameras")
    exit()

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("Failed to grab frames")
        break

    # 두 개의 카메라 영상을 나란히 보여줌
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()
