import cv2
import numpy as np

def select_pixel_with_keys(image, side='left'):
    clone = image.copy()
    selected = [image.shape[1] // 2, image.shape[0] // 2 + 200]

    def draw_cursor(img, pos):
        temp = img.copy()
        cv2.drawMarker(temp, tuple(pos), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
        return temp

    cv2.imshow(f"{side} Select", draw_cursor(clone, selected))

    while True:
        key = cv2.waitKeyEx(0)
        if key == 27:  # ESC
            print(f"[{side}] 선택 취소")
            cv2.destroyWindow(f"{side} Select")
            return None
        elif key in [13, 10]:  # Enter
            break
        elif key == 2490368:  # ↑
            selected[1] = max(0, selected[1] - 1)
        elif key == 2621440:  # ↓
            selected[1] = min(image.shape[0] - 1, selected[1] + 1)
        elif key == 2424832:  # ←
            selected[0] = max(0, selected[0] - 1)
        elif key == 2555904:  # →
            selected[0] = min(image.shape[1] - 1, selected[0] + 1)

        cv2.imshow(f"{side} Select", draw_cursor(clone, selected))

    cv2.destroyWindow(f"{side} Select")
    print(f"[{side}] 선택된 픽셀: (u={selected[0]}, v={selected[1]})")
    return tuple(selected)

def capture_from_camera(camera_id, side='left'):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[{side}] 카메라 {camera_id} 열기 실패")
        return None

    print(f"[{side}] 스페이스바로 캡처, ESC로 취소")
    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{side}] 프레임 읽기 실패")
            break

        cv2.imshow(f"{side} Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print(f"[{side}] 취소됨")
            break
        elif key == 32:  # Space
            captured_frame = frame.copy()
            print(f"[{side}] 캡처 완료")
            break

    cap.release()
    cv2.destroyWindow(f"{side} Camera")
    if captured_frame is not None:
        return select_pixel_with_keys(captured_frame, side)
    return None

if __name__ == '__main__':
    print("🔴 왼쪽 카메라 (ID=0)에서 캡처")
    left_point = capture_from_camera(0, 'left')

    print("\n🔵 오른쪽 카메라 (ID=1)에서 캡처")
    right_point = capture_from_camera(1, 'right')

    print("\n=== 최종 선택된 좌표 ===")
    if left_point:
        print(f"왼쪽 카메라:  (u={left_point[0]}, v={left_point[1]})")
    else:
        print("왼쪽 카메라: 선택 안됨")
    if right_point:
        print(f"오른쪽 카메라: (u={right_point[0]}, v={right_point[1]})")
    else:
        print("오른쪽 카메라: 선택 안됨")

    # 배열에 저장
    left_points = [left_point] if left_point else []
    right_points = [right_point] if right_point else []
