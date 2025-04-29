import cv2
import numpy as np

# 1) Intrinsic + Distortion (MATLAB 결과 반영)
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

# 2) Extrinsic via Look-At (camera → world)
def get_extrinsics_lookat(side, R=10.0, H=1.0):
    θ = np.deg2rad(60.0)
    if side == 'left':
        C = np.array([-R*np.sin(θ), H, R*np.cos(θ)], dtype=np.float32)
    else:
        C = np.array([R*np.sin(θ), H, R*np.cos(θ)], dtype=np.float32)

    forward = (np.zeros(3, dtype=np.float32) - C)
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
    dir_cam = np.array([ x_norm,
                        -y_norm,  # OpenCV y-down → y-up 보정
                         1.0 ], dtype=np.float32)
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
    cam_dir   = pixel_to_cam_dir(u, v, K, dist)  # 카메라 로컬 3D 방향
    world_dir = Rcw_look @ cam_dir                # 월드 좌표계로 변환
    world_dir /= np.linalg.norm(world_dir)

    # 실제 카메라 위치에서 홈플레이트 방향으로 이상적 벡터 계산
    ideal_vec = -C  # 카메라 위치 → 원점(홈플레이트)
    ideal_dir = ideal_vec / np.linalg.norm(ideal_vec)

    # 방향 오차 계산
    err_rad = np.arccos(np.clip(np.dot(world_dir, ideal_dir), -1.0, 1.0))
    err_deg = np.degrees(err_rad)

    print(f"\n[{side.upper()} CAMERA]")
    print(f" 클릭 픽셀: (u={u}, v={v})")
    print(f" cam_dir:    {cam_dir}")
    print(f" world_dir:  {world_dir}")
    print(f" ideal_dir:  {ideal_dir}")
    print(f" 오차:       {err_deg:.3f}°")
    return err_deg

if __name__ == '__main__':
    eL = click_and_compute('directionvector/left2.png', 'left')
    eR = click_and_compute('directionvector/right2.png', 'right')
    print("\n=== 최종 오차 ===")
    if eL is not None:
        print(f" 왼쪽 카메라:  {eL:.3f}°")
    if eR is not None:
        print(f" 오른쪽 카메라: {eR:.3f}°")
