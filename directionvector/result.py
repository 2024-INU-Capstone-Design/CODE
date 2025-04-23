import cv2
import numpy as np

# 1. 카메라 파라미터 설정
K_left = np.array([[1676.1, 0, 646.5181],
                   [0, 1674.6, 370.2439],
                   [0, 0, 1]])
dist_left = np.array([0.0274, -0.1214, 0, 0, 0])

K_right = np.array([[1722.7, 0, 652.8481],
                    [0, 1721.7, 342.1181],
                    [0, 0, 1]])
dist_right = np.array([-0.0332, 0.2844, 0, 0, 0])

# 2. 이미지 상의 공 위치 (픽셀)
pt_left = np.array([[781, 861]], dtype=np.float32)
pt_right = np.array([[906, 620]], dtype=np.float32)

# 3. 방향벡터 구하는 함수
def get_direction_vector(image_point, K, dist):
    undistorted = cv2.undistortPoints(image_point, K, dist, P=K)
    x, y = undistorted[0][0]
    direction = np.array([x, y, 1.0])
    return direction / np.linalg.norm(direction)

# 4. 방향벡터 계산
dir_left = get_direction_vector(pt_left, K_left, dist_left)
dir_right = get_direction_vector(pt_right, K_right, dist_right)

print("왼쪽 방향벡터:", dir_left)
print("오른쪽 방향벡터:", dir_right)

# 5. 카메라 위치 설정 (홈플레이트 기준, 좌우 60도 방향, 10m 거리)
angle = np.deg2rad(60)
left_cam_pos = np.array([10 * np.cos(angle), 0, 10 * np.sin(angle)])
right_cam_pos = np.array([10 * np.cos(-angle), 0, 10 * np.sin(-angle)])

# 6. 두 직선의 최근접점 계산 (Triangulation)
def triangulate(p1, d1, p2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    cross_d = np.cross(d1, d2)
    denom = np.linalg.norm(cross_d)**2
    if denom < 1e-6:
        return None  # 병렬이라 계산 불가

    t = np.dot(np.cross((p2 - p1), d2), cross_d) / denom
    u = np.dot(np.cross((p2 - p1), d1), cross_d) / denom

    closest_point1 = p1 + t * d1
    closest_point2 = p2 + u * d2
    midpoint = (closest_point1 + closest_point2) / 2.0
    return midpoint

ball_position = triangulate(left_cam_pos, dir_left, right_cam_pos, dir_right)
print("\n📍 추정된 공의 3D 위치:", ball_position)
