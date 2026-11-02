"""모은 체커보드 사진으로 카메라 내부 파라미터(K)와 왜곡 계수를 구한다.

python scripts/calib_intrinsics.py --square-mm 24.6       # 인쇄된 칸을 자로 잰 값
python scripts/calib_intrinsics.py --cols 8 --rows 5 --square-mm 30

결과: calib/camera_top.yaml, 비교 그림 data/calib_check/undistort.jpg
빈칸: so101/calib.py 의 make_object_points, reprojection_errors
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.calib import make_object_points, reprojection_errors, save_intrinsics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square-mm", type=float, required=True, help="인쇄된 칸 한 변을 자로 잰 길이")
args = ap.parse_args()

pattern = (args.cols, args.rows)
objp = make_object_points(args.cols, args.rows, args.square_mm)
paths = sorted((config.DATA_DIR / "calib").glob("*.jpg"))
if len(paths) < 10:
    raise SystemExit(f"사진이 {len(paths)}장뿐이다. calib_capture.py 로 15장 이상 모을 것")

objpoints, imgpoints, used = [], [], []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
for p in paths:
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(gray, pattern)
    if not found:
        print("코너 못 찾음, 건너뜀:", p.name)
        continue
    # 코너 위치를 픽셀보다 잘게(sub-pixel) 다듬는다
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners)
    used.append(p)
size = gray.shape[::-1]  # (width, height)

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
errs = np.asarray(reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist))

np.set_printoptions(precision=3, suppress=True)
print(f"\n사용한 사진 {len(used)}장, 전체 RMS 재투영 오차 {rms:.3f} px")
print("K =\n", K)
print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}  (영상 중심은 {size[0] / 2:.0f}, {size[1] / 2:.0f})")
print("dist (k1 k2 p1 p2 k3) =", dist.ravel())
print("\n사진별 오차 (큰 순서) — 1 px 넘는 사진은 흔들렸거나 코너가 잘못 잡힌 것. 빼고 다시 돌려 볼 것")
for i in np.argsort(-errs):
    print(f"  {errs[i]:.3f} px  {used[i].name}{'  ←' if errs[i] > 1.0 else ''}")

config.CAMERA_INTRINSICS.parent.mkdir(parents=True, exist_ok=True)
save_intrinsics(config.CAMERA_INTRINSICS, K, dist, size, rms, pattern, args.square_mm, len(used))
print("\n저장:", config.CAMERA_INTRINSICS)

img = cv2.imread(str(used[int(np.argmax(errs))]))
und = cv2.undistort(img, K, dist)
check_dir = config.DATA_DIR / "calib_check"
check_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(check_dir / "undistort.jpg"), np.hstack([img, und]))
print("왼쪽 원본 / 오른쪽 왜곡 보정:", check_dir / "undistort.jpg", "(직선이 더 곧게 펴졌는지 보기)")
