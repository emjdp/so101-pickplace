"""카메라 내부 보정. ★ 실습: TODO 2개를 채운다 ★

확인: pytest tests/test_calib.py -v
      python scripts/calib_intrinsics.py
참고: 문서 '3. 카메라 보정과 블록 검출' 4절, OpenCV 튜토리얼 'Camera Calibration'
"""

from datetime import datetime

import cv2
import numpy as np


def make_object_points(cols: int, rows: int, square: float) -> np.ndarray:
    """체커보드 내부 코너의 3D 좌표 (cols*rows, 3) float32. 보드 평면을 z=0 으로 둔다.

    cols, rows: 가로·세로 '내부 코너' 개수 (칸 수가 아니다. 10x7 칸이면 9x6)
    square:     한 칸의 실제 길이. 여기서 정한 단위(mm)가 tvec 의 단위가 된다.

    순서가 cv2.findChessboardCorners 결과와 같아야 한다:
      (0,0,0), (s,0,0), (2s,0,0), ..., ((cols-1)s,0,0), (0,s,0), ...   → x 가 먼저 증가
    힌트: np.mgrid[0:cols, 0:rows] 또는 np.meshgrid. .T.reshape(-1, 2) 순서를 확인할 것
    """
    raise NotImplementedError("TODO: make_object_points")


def reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist) -> np.ndarray:
    """사진마다 재투영 오차(RMS, px)를 구한다. (N,) 배열.

    보정 결과(K, dist, 사진별 rvec/tvec)로 3D 코너를 다시 영상에 투영하고,
    실제로 검출한 코너와의 거리 제곱 평균의 제곱근을 사진별로 구한다.
    힌트: cv2.projectPoints(objp, rvec, tvec, K, dist) → (projected, _)
          projected 와 imgpoints[i] 는 모양이 (M, 1, 2) 다
    """
    raise NotImplementedError("TODO: reprojection_errors")


# ── 저장/불러오기 (완성) ─────────────────────────────────────────────


def save_intrinsics(path, K, dist, image_size, rms, pattern, square_mm, n_images):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("date", datetime.now().isoformat(timespec="seconds"))
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("rms_px", float(rms))
    fs.write("pattern", f"{pattern[0]}x{pattern[1]} inner corners, {square_mm} mm")
    fs.write("n_images", int(n_images))
    fs.release()


def load_intrinsics(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    fs.release()
    return K, dist
