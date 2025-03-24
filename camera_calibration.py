import cv2
import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

# 이미지 로드
img1 = cv2.imread('left.jpg')
img2 = cv2.imread('right.jpg')

# 이미지를 흑백으로 변환
gray1 = rgb2gray(img1)
gray2 = rgb2gray(img2)

# 1. 특징점 검출 (ORB 사용)
orb = ORB(n_keypoints=500)
orb.detect_and_extract(gray1)
keypoints1 = orb.keypoints
descriptors1 = orb.descriptors

orb.detect_and_extract(gray2)
keypoints2 = orb.keypoints
descriptors2 = orb.descriptors

# 2. 특징점 매칭
matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

# 매칭된 특징점 좌표 추출
matched_points1 = keypoints1[matches[:, 0]]
matched_points2 = keypoints2[matches[:, 1]]

# 3. RANSAC을 이용한 아웃라이어 제거
model_robust, inliers = ransac(
    (matched_points1, matched_points2),
    ProjectiveTransform,
    min_samples=4,  # 최소 샘플 수
    residual_threshold=2,  # 허용 오차 (픽셀 단위)
    max_trials=1000  # 최대 시도 횟수
)

# 인라이어만 추출
inlier_points1 = matched_points1[inliers]
inlier_points2 = matched_points2[inliers]

# 시각화
result_img = cv2.hconcat([
    cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),
])

for pt1, pt2 in zip(inlier_points1, inlier_points2):
    # pt1과 pt2는 numpy 배열이어야 함
    pt1 = np.array(pt1, dtype=int)  # numpy 배열로 변환
    pt2_adjusted = np.array([pt2[0] + img1.shape[1], pt2[1]], dtype=int)  # 좌표 조정 후 배열로 변환
    
    # 선 그리기
    cv2.line(result_img, tuple(pt1), tuple(pt2_adjusted), (0, 255, 0), 1)

# 창 크기 조정 가능하게 설정
cv2.namedWindow('Matched Points (After RANSAC)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Matched Points (After RANSAC)', 1920, 1080)  # 원하는 크기로 설정

# 결과 출력
cv2.imshow('Matched Points (After RANSAC)', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
