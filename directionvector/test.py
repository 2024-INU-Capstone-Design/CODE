import cv2
import numpy as np

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

    C = np.array([distance * np.sin(az), height, distance * np.cos(az)], dtype=np.float32)
    forward = np.array([np.sin(az), 0, np.cos(az)], dtype=np.float32)
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

# 4) 클릭 → 오차 계산
def click_and_compute(image_path, side):
    K, dist = get_K_dist(side)
    C, Rcw = get_extrinsics_by_angle(side)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    clone = img.copy()
    selected = [img.shape[1] // 2, img.shape[0] // 2 + 240]  # 초기 위치 (중앙)

    def draw_cursor(img, pos):
        temp = img.copy()
        cv2.drawMarker(temp, tuple(pos), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=15, thickness=1)
        return temp

    cv2.imshow(f"{side} Click", draw_cursor(clone, selected))

    while True:
        key = cv2.waitKeyEx(0)
        if key == 27:  # ESC
            print(f"[{side}] 취소됨")
            cv2.destroyAllWindows()
            return None
        elif key in [13, 10]:  # Enter
            break
        elif key == 2490368:  # ↑
            selected[1] = max(0, selected[1] - 1)
        elif key == 2621440:  # ↓
            selected[1] = min(img.shape[0] - 1, selected[1] + 1)
        elif key == 2424832:  # ←
            selected[0] = max(0, selected[0] - 1)
        elif key == 2555904:  # →
            selected[0] = min(img.shape[1] - 1, selected[0] + 1)

        cv2.imshow(f"{side} Click", draw_cursor(clone, selected))

    cv2.destroyAllWindows()
    u, v = selected
    cam_dir = pixel_to_cam_dir(u, v, K, dist)
    world_dir = Rcw @ cam_dir
    world_dir /= np.linalg.norm(world_dir)

    if side == 'left':
        ideal_dir = np.array([+8.66, 1.0, 5.0], dtype=np.float32)
    else:
        ideal_dir = np.array([-8.66, 1.0, 5.0], dtype=np.float32)
    ideal_dir /= np.linalg.norm(ideal_dir)

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
    eL = click_and_compute('directionvector/left2.png', 'left')
    eR = click_and_compute('directionvector/right2.png', 'right')
    print("\n=== 최종 방위 오차 ===")
    if eL is not None:
        print(f" 왼쪽 카메라:  {eL:.3f}°")
    if eR is not None:
        print(f" 오른쪽 카메라: {eR:.3f}°")
