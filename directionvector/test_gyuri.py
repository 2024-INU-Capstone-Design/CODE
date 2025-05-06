import cv2
import numpy as np
import platform

# 1) Intrinsic + Distortion
def get_K_dist(side):
    if side == 'left':
        K = np.array([[1777.39872361239, 0, 643.444883498561],
                      [0, 1775.07100197172, 370.062585668853],
                      [0, 0, 1]], dtype=np.float32)
        dist = np.array([0.0262313355787859, 0.181622848916271, 0, 0, 0], dtype=np.float32)
    else:
        K = np.array([[1707.52273302496, 0, 640.928904709027],
                      [0, 1707.62821582339, 343.451657880970],
                      [0, 0, 1]], dtype=np.float32)
        dist = np.array([0.174451816407887, -0.516395174012713, 0, 0, 0], dtype=np.float32)
    return K, dist

# 2) Extrinsic via fixed azimuth angles
def get_extrinsics_by_angle(side, distance=10.0, height=1.0, angle_deg=60.0):
    θ = np.deg2rad(angle_deg)
    if side == 'left':
        az = θ
    else:
        az = -θ

    # 카메라 위치
    C = np.array([distance * np.sin(az), height, distance * np.cos(az)], dtype=np.float32)

    # 카메라가 바라보는 방향 (월드 기준)
    forward = np.array([-np.sin(az), 0, -np.cos(az)], dtype=np.float32)  # Z축 기준 -60도, +60도 방향
    forward /= np.linalg.norm(forward)

    up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)

    R_cam2world = np.stack([right, true_up, forward], axis=1)
    return C, R_cam2world

# 3) 픽셀 → 카메라 로컬 방향벡터
def pixel_to_cam_dir(u, v, K, dist):
    pt = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.undistortPoints(pt, K, dist, P=None)
    x_norm, y_norm = und[0, 0]
    dir_cam = np.array([x_norm, -y_norm, 1.0], dtype=np.float32)
    return dir_cam / np.linalg.norm(dir_cam)

# 4) 키보드로 점 이동하며 픽셀 선택 → 방향 오차 계산
def click_and_compute(image_path, side):
    is_windows = platform.system() == 'Windows' # 플랫폼에 따라 입력 방식을 분기

    K, dist = get_K_dist(side) # 내부 파라미터 및 왜곡계수 로드
    C, Rcw = get_extrinsics_by_angle(side) # 카메라 위치와 월드->카메라 회전 행렬

    img = cv2.imread(image_path) # 이미지 로드
    if img is None:
        raise FileNotFoundError(image_path) # 이미지가 없으면 예외 발생

    h, w = img.shape[:2]
    x, y = w // 2, h // 2 # 중앙 시작
    
    print("←↑↓→ 또는 WASD로 빨간 점 이동 / Enter 선택 / ESC 종료")

    # 마우스 클릭 코드
    # coords = []
    # def on_click(evt, x, y, flags, param):
    #     if evt == cv2.EVENT_LBUTTONDOWN:
    #         coords.append((x, y))
    #         cv2.destroyAllWindows()

    # cv2.imshow(f"{side} Click", img)
    # cv2.setMouseCallback(f"{side} Click", on_click)
    # cv2.waitKey(0)

    # if not coords:
    #     print(f"[{side}] 클릭 없음")
    #     return None

    # u, v = coords[0]

    while True:
        img_disp = img.copy() # 원본 이미지 복사
        cv2.circle(img_disp, (x, y), 5, (0, 0, 255), -1) # 빨간 점 그리기
        cv2.imshow(f"{side} Select", img_disp) # 이미지 출력

        key = cv2.waitKeyEx(0) if is_windows else cv2.waitKey(0) & 0xFF

        if key in [13, 10]: # Enter 종료
            break
        elif key == 27:
            print(f"[{side}] 선택 취소") # ESC 취소
            cv2.destroyAllWindows()
            return None

        # if is_windows:
        #     if key == 2490368 or key == ord('w'): y = max(0, y - 3)
        #     elif key == 2621440 or key == ord('s'): y = min(h - 1, y + 3)
        #     elif key == 2424832 or key == ord('a'): x = max(0, x - 3)
        #     elif key == 2555904 or key == ord('d'): x = min(w - 1, x + 3)
        # else:
        #     if key == ord('w'): y = max(0, y - 3)
        #     elif key == ord('s'): y = min(h - 1, y + 3)
        #     elif key == ord('a'): x = max(0, x - 3)
        #     elif key == ord('d'): x = min(w - 1, x + 3)

        # 방향키 및 WASD 입력 처리
        if is_windows:
            if key == 2490368 or key == ord('w'):  # ↑ 또는 W
                y = max(0, y - 3)
            elif key == 2621440 or key == ord('s'):  # ↓ 또는 S
                y = min(h - 1, y + 3)
            elif key == 2424832 or key == ord('a'):  # ← 또는 A
                x = max(0, x - 3)
            elif key == 2555904 or key == ord('d'):  # → 또는 D
                x = min(w - 1, x + 3)
        else:
            # macOS는 방향키 대신 WASD만 처리
            if key == ord('w'):
                y = max(0, y - 3)
            elif key == ord('s'):
                y = min(h - 1, y + 3)
            elif key == ord('a'):
                x = max(0, x - 3)
            elif key == ord('d'):
                x = min(w - 1, x + 3)

    cv2.destroyAllWindows()

    u, v = x, y # 최종 선택 위치

    cam_dir = pixel_to_cam_dir(u, v, K, dist)
    world_dir = Rcw @ cam_dir
    world_dir /= np.linalg.norm(world_dir)

    # 이상적 방향벡터 (카메라 위치에서 홈플레이트를 향한 벡터)
    ideal_vec = -C
    ideal_dir = ideal_vec / np.linalg.norm(ideal_vec)

    az_world = np.arctan2(world_dir[0], world_dir[2])
    az_ideal = np.arctan2(ideal_dir[0], ideal_dir[2])
    delta_deg = np.degrees(az_world - az_ideal)
    delta_deg = (delta_deg + 180) % 360 - 180

    print(f"\n[{side.upper()} CAMERA]")
    print(f" 클릭 픽셀: (u={u}, v={v})")
    print(f" cam_dir:    {cam_dir}")
    print(f" world_dir:  {world_dir}")
    print(f" ideal_dir:  {ideal_dir}")
    print(f" azimuth → world={np.degrees(az_world):.2f}°, ideal={np.degrees(az_ideal):.2f}°")
    print(f" 방위각 오차: {delta_deg:+.2f}°")
    return delta_deg

if __name__ == '__main__':
    eL = click_and_compute('Capstone-Design/directionvector/left2.png', 'left')
    eR = click_and_compute('Capstone-Design/directionvector/right2.png', 'right')
    print("\n=== 최종 방위 오차 ===")
    if eL is not None:
        print(f" 왼쪽 카메라:  {eL:.3f}°")
    if eR is not None:
        print(f" 오른쪽 카메라: {eR:.3f}°")
