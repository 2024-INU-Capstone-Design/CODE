import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 1. 카메라 내부 파라미터 + 왜곡 계수
def get_K_dist(side):
    if side == 'left':
        K = np.array([[1777.3987, 0, 643.4449],
                      [0, 1775.0710, 370.0626],
                      [0, 0, 1]], dtype=np.float32)
        dist = np.array([0.02623, 0.1816, 0, 0, 0], dtype=np.float32)
    else:
        K = np.array([[1707.5227, 0, 640.9289],
                      [0, 1707.6282, 343.4517],
                      [0, 0, 1]], dtype=np.float32)
        dist = np.array([0.17445, -0.5164, 0, 0, 0], dtype=np.float32)
    return K, dist

# 2. 픽셀 좌표 → 방향 벡터(cam 기준)
def pixel_to_cam_dir(u, v, K, dist):
    pt = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.undistortPoints(pt, K, dist, P=None)
    x_norm, y_norm = und[0, 0]
    dir_cam = np.array([x_norm, -y_norm, 1.0], dtype=np.float32)  # y축 뒤집기 (y-up 보정)
    return dir_cam / np.linalg.norm(dir_cam)

# 3. 방향벡터 → 좌우/위아래 각도 계산
def compute_angle_from_center(cam_dir):
    azimuth_rad = np.arctan2(cam_dir[0], cam_dir[2])  # 좌우
    elevation_rad = np.arctan2(cam_dir[1], np.sqrt(cam_dir[0]**2 + cam_dir[2]**2))  # 위아래
    return np.degrees(azimuth_rad), np.degrees(elevation_rad)

# 4. 전체 파이프라인 실행
def click_and_compute_angles(image_path, side):
    K, dist = get_K_dist(side)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    coords = []
    def on_click(evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            cv2.destroyAllWindows()

    cv2.imshow(f"{side.upper()} CAMERA - CLICK", img)
    cv2.setMouseCallback(f"{side.upper()} CAMERA - CLICK", on_click)
    cv2.waitKey(0)

    if not coords:
        print("클릭이 감지되지 않았습니다.")
        return

    u, v = coords[0]
    cam_dir = pixel_to_cam_dir(u, v, K, dist)
    az_deg, el_deg = compute_angle_from_center(cam_dir)

    print(f"\n [{side.upper()} CAMERA]")
    print(f"  클릭한 픽셀 위치: (u={u}, v={v})")
    print(f"  정규화된 방향벡터: {cam_dir}")
    print(f"  좌우 (Azimuth):  {az_deg:+.2f}°")
    print(f"  위아래 (Elevation): {el_deg:+.2f}°")
    return az_deg, el_deg

# 카메라 - 홈플레이트 각도
angle1_az_deg_homplate, angle1_el_deg_homeplate = click_and_compute_angles('SB_camera.png', 'left')
angle2_az_deg_homplate, angle2_el_deg_homeplate = click_and_compute_angles('HS_camera.png', 'right')

# 카메라 - 야구공 각도
angle1_az_deg_baseball, angle1_el_deg_baseball= click_and_compute_angles('SB_camera.png', 'left')
angle2_az_deg_baseball, angle2_el_deg_baseball = click_and_compute_angles('HS_camera.png', 'right')


# 월드 좌표계에서의의 각도 (degree)
angle1_deg = 60 + abs(angle1_az_deg_homplate) - abs(angle1_az_deg_baseball)  # 수빈 카메라 기준
angle2_deg = 60 + abs(angle2_az_deg_homplate) - abs(angle2_az_deg_baseball) # 현서 카메라 기준

print(angle1_deg)
print(angle2_deg)

# 기울기 계산 (90 - angle), 왼쪽 카메라는 음수
slope1 = np.tan(np.radians(90 - angle1_deg))      # 오른쪽 카메라: 양수 기울기
slope2 = -np.tan(np.radians(90 - angle2_deg))     # 왼쪽 카메라: 음수 기울기

# 시작점
x1_start, y1_start = 500 * np.sqrt(3), 500   # 수빈 카메라
x2_start, y2_start = -500 * np.sqrt(3), 500  # 현서 카메라


# y값 범위 제한
y_min, y_max = 0, 500

# 직선 계산 함수
def compute_line_points(x0, y0, slope):
    y_vals = np.linspace(y_min, y_max, 1000)
    x_vals = x0 + (y_vals - y0) / slope
    return x_vals, y_vals

# x,y 직선 좌표 계산
x1_vals, y1_vals = compute_line_points(x1_start, y1_start, slope1)
x2_vals, y2_vals = compute_line_points(x2_start, y2_start, slope2)

# x,y 교차점 계산
numerator = (slope1 * x1_start - y1_start) - (slope2 * x2_start - y2_start)
denominator = slope1 - slope2

# z값 각도 (degree)
angle1_between = abs(angle1_el_deg_homeplate) - 5.94 # 여기서 5.94는 이제 tan104/1000 계산값값  
angle1_baseball = abs(angle1_el_deg_baseball) - angle1_between

angle2_between = abs(angle2_el_deg_homeplate) - 5.94 # 여기서 5.94는 이제 tan104/1000 계산값값  
angle2_baseball = abs(angle2_el_deg_baseball) - angle2_between

final_target_height = 0
final_target_height_2 = 0
target_z = 0

print(angle1_baseball)
print(angle2_baseball)


if abs(denominator) < 1e-6:
    intersection_point = None
else:
    x_intersect = numerator / denominator
    y_intersect = slope1 * (x_intersect - x1_start) + y1_start
    intersection_point = (x_intersect, y_intersect)
    # 거리
    distance = math.sqrt((x1_start - intersection_point[0])**2 + (y1_start - intersection_point[1])**2)
    target_height = math.tan(math.radians(angle1_baseball)) * distance
    target_height_2 = math.tan(math.radians(angle2_baseball)) * distance
    final_target_height = 104 - target_height
    final_target_height_2 = 104 - target_height_2
    target_z = (final_target_height + final_target_height_2)/2
    print(final_target_height)
    print(final_target_height_2)

# 그래프 시각화
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 교점 좌표
x, y = intersection_point


# 교점 표시 (있을 경우)
if intersection_point is not None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 교점 좌표
    x, y = intersection_point

    # 카메라 좌표들 (z=0으로 표시)
    ax.scatter(x1_start, y1_start, 104, color='blue', label='Subin Camera')
    ax.scatter(x2_start, y2_start, 104, color='green', label='Hyunseo Camera')

    ax.plot([x1_start, x1_start], [y1_start, y1_start], [0, 104], color='blue', linestyle='dotted')
    ax.plot([x2_start, x2_start], [y2_start, y2_start], [0, 104], color='green', linestyle='dotted')

    # 교차점 좌표
    ax.scatter(x, y, target_z, color='purple', s=30, label='3D Intersection Point')
    ax.text(x + 10, y + 10, target_z + 10, f"({x:.2f}, {y:.2f}, {target_z:.2f})", color='purple', fontsize=10)

    # 시선 연결선
    ax.plot([x1_start, x], [y1_start, y], [104,target_z], color='blue', linestyle='dotted')
    ax.plot([x2_start, x], [y2_start, y], [104,target_z], color='green', linestyle='dotted')

#홈플레이트 좌표
point = [[21.6, 0,0], [-21.6, 0,0], [21.6, -30.48,0], [-21.6, -30.48,0],[0,-45.98,0]]

ax.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], [point[0][2], point[1][2]],  color='orange', linestyle='-', linewidth=2, label='Custom Line')
ax.plot([point[0][0], point[2][0]], [point[0][1], point[2][1]], [point[0][2], point[2][2]], color='orange', linestyle='-', linewidth=2)
ax.plot([point[1][0], point[3][0]], [point[1][1], point[3][1]], [point[1][2], point[3][2]], color='orange', linestyle='-', linewidth=2)
ax.plot([point[2][0], point[4][0]], [point[2][1], point[4][1]], [point[2][2], point[4][2]], color='orange', linestyle='-', linewidth=2)
ax.plot([point[3][0], point[4][0]], [point[3][1], point[4][1]], [point[3][2], point[4][2]], color='orange', linestyle='-', linewidth=2)

# 홈플레이트 면 색칠
homeplate_face = [point[0], point[1], point[3], point[4], point[2]]
homeplate_poly = Poly3DCollection([homeplate_face], color='orange', alpha=0.3)
ax.add_collection3d(homeplate_poly)


ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.set_title('3D Triangulation View')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
