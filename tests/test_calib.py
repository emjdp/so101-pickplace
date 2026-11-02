"""가짜 카메라로 체커보드를 '촬영'해서 보정 빈칸 함수를 확인한다."""

import cv2
import numpy as np

from so101.calib import make_object_points, reprojection_errors


def test_object_points_layout():
    objp = make_object_points(9, 6, 25.0)
    assert objp.shape == (54, 3) and objp.dtype == np.float32
    np.testing.assert_allclose(objp[0], [0, 0, 0])
    np.testing.assert_allclose(objp[1], [25, 0, 0])  # x 가 먼저 증가
    np.testing.assert_allclose(objp[9], [0, 25, 0])  # 한 줄(9개) 뒤에 y 증가
    np.testing.assert_allclose(objp[-1], [200, 125, 0])
    assert np.all(objp[:, 2] == 0)


def _fake_views(K, dist, objp, n=6):
    rng = np.random.default_rng(1)
    obj, img, rv, tv = [], [], [], []
    for _ in range(n):
        rvec = rng.uniform(-0.3, 0.3, (3, 1))
        tvec = np.array([[-100.0], [-60.0], [450.0]]) + rng.uniform(-30, 30, (3, 1))
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        obj.append(objp)
        img.append(proj.astype(np.float32))
        rv.append(rvec)
        tv.append(tvec)
    return obj, img, rv, tv


def test_reprojection_error_zero_for_perfect_data():
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    dist = np.zeros(5)
    obj, img, rv, tv = _fake_views(K, dist, make_object_points(9, 6, 25.0))
    err = reprojection_errors(obj, img, rv, tv, K, dist)
    assert np.asarray(err).shape == (6,)
    assert np.all(np.asarray(err) < 1e-3)


def test_reprojection_error_detects_shift():
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    dist = np.zeros(5)
    obj, img, rv, tv = _fake_views(K, dist, make_object_points(9, 6, 25.0))
    img[0] = img[0] + np.float32([3.0, 4.0])  # 첫 사진의 코너를 전부 (3, 4) px 밀기 → 오차 5 px
    err = np.asarray(reprojection_errors(obj, img, rv, tv, K, dist))
    assert np.isclose(err[0], 5.0, atol=1e-3)
    assert np.all(err[1:] < 1e-3)
