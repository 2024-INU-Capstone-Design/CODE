import cv2
import numpy as np

# 1) Intrinsic + Distortion
def get_K_dist(side):
    if side=='left':
        K    = np.array([[1676.1, 0,      646.5181],
                         [0,      1674.6, 370.2439],
                         [0,      0,      1      ]], dtype=np.float32)
        dist = np.array([ 0.0274, -0.1214, 0, 0, 0], dtype=np.float32)
    else:
        K    = np.array([[1722.7, 0,      652.8481],
                         [0,      1721.7, 342.1181],
                         [0,      0,      1      ]], dtype=np.float32)
        dist = np.array([-0.0332,  0.2844, 0, 0, 0], dtype=np.float32)
    return K, dist

# 2) Extrinsic via Look-At (camera→world)
def get_extrinsics_lookat(side, R=10.0, H=1.0):
    θ = np.deg2rad(60.0)
    # 카메라 월드 위치
    if side=='left':
        C = np.array([-R*np.sin(θ), H, R*np.cos(θ)], dtype=np.float32)
    else:
        C = np.array([ R*np.sin(θ), H, R*np.cos(θ)], dtype=np.float32)

    # 2-1) 카메라 정면 축 (world) = (원점 − 카메라)
    forward = (np.zeros(3, dtype=np.float32) - C)
    forward /= np.linalg.norm(forward)

    # 2-2) world up
    up = np.array([0, 1, 0], dtype=np.float32)

    # 2-3) 카메라 우측 축 = up × forward
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    # 2-4) 카메라 상단 축 = forward × right
    true_up = np.cross(forward, right)

    # 2-5) Rotation matrix: camera local → world
    # Columns are camera‐frame x, y, z axes in world coords
    R_cam2world = np.stack([right, true_up, forward], axis=1)
    return C, R_cam2world

# 3) 픽셀→카메라 로컬 방향벡터
def pixel_to_cam_dir(u, v, K, dist):
    pt = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.undistortPoints(pt, K, dist, P=None)   # normalized (X/Z, Y/Z)
    x_norm, y_norm = und[0,0]
    dir_cam = np.array([ x_norm,    # 오른쪽 +
                         -y_norm,   # 위쪽 + (OpenCV y-down→y-up)
                          1.0      ], dtype=np.float32)  # 전방
    return dir_cam / np.linalg.norm(dir_cam)

# 4) 클릭→오차 계산
def click_and_compute(image_path, side):
    K, dist      = get_K_dist(side)
    C, Rcw_look  = get_extrinsics_lookat(side)

    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(image_path)

    coords = []
    def on_click(evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            cv2.destroyAllWindows()

    cv2.imshow(f"{side} Click", img)
    cv2.setMouseCallback(f"{side} Click", on_click)
    cv2.waitKey(0)

    if not coords:
        print(f"[{side}] 클릭 없음"); return None

    u, v      = coords[0]
    cam_dir   = pixel_to_cam_dir(u, v, K, dist)       # 카메라 로컬
    world_dir = Rcw_look @ cam_dir                    # world 기준
    world_dir /= np.linalg.norm(world_dir)

    # Ideal = “카메라 위치 → 홈플레이트(0,0,0)”
    ideal_dir = -C
    ideal_dir /= np.linalg.norm(ideal_dir)

    err = np.degrees(np.arccos(np.clip(np.dot(world_dir, ideal_dir), -1, 1)))

    print(f"\n[{side.upper()} CAMERA]")
    print(f" 클릭 픽셀: (u={u}, v={v})")
    print(f" cam_dir:    {cam_dir}")
    print(f" world_dir:  {world_dir}")
    print(f" ideal_dir:  {ideal_dir}")
    print(f" 오차:       {err:.3f}°")
    return err

if __name__=='__main__':
    eL = click_and_compute('directionvector/left.png',  'left')
    eR = click_and_compute('directionvector/right.png', 'right')
    print("\n=== 최종 오차 ===")
    if eL is not None: print(f" 왼쪽 카메라:  {eL:.3f}°")
    if eR is not None: print(f" 오른쪽 카메라: {eR:.3f}°")
