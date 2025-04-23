import numpy as np
import cv2

# 카메라 내부 파라미터
K = np.array([
    [1055.58, 0, 538.80],
    [0, 1056.53, 727.15],
    [0, 0, 1]
])

# 왜곡 계수 (radial + tangential)
dist_coeffs = np.array([0.1576, -0.3916, 0, 0])

def get_direction_vector(u, v):
    """
    왜곡 보정 포함해서 픽셀 좌표 (u, v)를 3D 방향 벡터로 변환
    """
    # 픽셀 좌표 → 정규화된 카메라 좌표 (undistortPoints가 자동으로 왜곡 보정 포함)
    undistorted = cv2.undistortPoints(
        np.array([[[u, v]]], dtype=np.float32),
        cameraMatrix=K,
        distCoeffs=dist_coeffs
    )

    # 결과는 [x, y] 형태이므로 z = 1 붙이고 방향 벡터로 정규화
    x, y = undistorted[0][0]
    vec = np.array([x, y, 1.0])
    return vec / np.linalg.norm(vec)

# 예시: 공이 이미지에서 (550, 720)에 보였을 때
dir_vec = get_direction_vector(781, 860)
print("방향 벡터:", dir_vec)
