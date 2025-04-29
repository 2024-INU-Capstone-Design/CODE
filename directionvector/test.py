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

# 2) Extrinsic via Look-At (camera→world)
def get_extrinsics_lookat(side, X=10.0, H=1.0):
    if side == 'left':
        C = np.array([-X, H, 0.0], dtype=np.float32)
    else:
        C = np.array([X, H, 0.0], dtype=np.float32)

    forward = -C  # camera → 원점
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
    dir_cam = np.array([x_norm, -y_norm, 1.0], dtype=np.float32)  # y-up 보정
    return dir_cam / np.linalg.norm(dir_cam)

# 4) 클릭 → 오차 계산
def click_and_compute(image_path, side):
    K, dist     = get_K_dist(side)
    C, Rcw_look = get_extrinsics_lookat(side)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    coords = []
    def on_click(evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            cv2.destroyAllWindows()

    cv2.imshow(f"{side} Click", img)
    cv2.setMouseCallback(f"{side} Click", on_click)
    cv2.waitKey(0)

    if not coords:
        print(f"[{side}] 클릭 없음")
        return None

    u, v = coords[0]
    cam_dir   = pixel_to_cam_dir(u, v, K, dist)
    world_dir = Rcw_look @ cam_dir
    world_dir /= np.linalg.norm(world_dir)

    ideal_vec = -C  # 카메라 위치 → 원점
    ideal_dir = ideal_vec / np.linalg.norm(ideal_vec)

    # Azimuth (방위각)
    az_world = np.arctan2(world_dir[0], world_dir[2])
    az_ideal = np.arctan2(ideal_dir[0], ideal_dir[2])
    delta_deg = np.degrees(az_world - az_ideal)
    delta_deg = (delta_deg + 180) % 360 - 180  # [-180, 180] 범위로 정규화

    print(f"\n[{side.upper()} CAMERA]")
    print(f" 클릭 픽셀: (u={u}, v={v})")
    print(f" cam_dir:    {cam_dir}")
    print(f" world_dir:  {world_dir}")
    print(f" ideal_dir:  {ideal_dir}")
    print(f" 절대 azimuth ▶ world={np.degrees(az_world):.2f}°, ideal={np.degrees(az_ideal):.2f}°")
    print(f" 상대 방위 오프셋: {delta_deg:+.2f}°")
    return delta_deg

if __name__ == '__main__':
    eL = click_and_compute('directionvector/left2.png', 'left')
    eR = click_and_compute('directionvector/right2.png', 'right')
    print("\n=== 최종 방위 오프셋 ===")
    if eL is not None:
        print(f" 왼쪽 카메라:  {eL:.3f}°")
    if eR is not None:
        print(f" 오른쪽 카메라: {eR:.3f}°")
