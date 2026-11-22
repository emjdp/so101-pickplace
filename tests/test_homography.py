"""호모그래피 및 카메라-로봇 좌표 변환(so101/homography.py) 테스트."""

import math

import numpy as np

import config
from so101.homography import (
    HomographyCalib,
    TopViewRayCalib,
    apply_homography,
    apply_homography_inverse,
    fit_homography,
)


def test_fit_and_apply_homography_roundtrip(tmp_path):
    """가상 기준점 6쌍으로 호모그래피를 구하고 정변환·역변환·저장·불러오기를 검증한다."""
    H_true = np.array(
        [
            [0.0, -0.0008, 0.45],
            [-0.0008, 0.0, 0.25],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    uv = np.array(
        [[150.0, 120.0], [450.0, 120.0], [450.0, 360.0], [150.0, 360.0], [300.0, 240.0], [220.0, 300.0]],
        dtype=np.float64,
    )
    xy = apply_homography(H_true, uv)
    H_est, errs = fit_homography(uv, xy)
    assert np.all(errs < 1e-6)

    calib = HomographyCalib(H=H_est, plane_z_m=0.025, rms_mm=0.0, max_mm=0.0)
    x, y = calib.pixel_to_xy(320.0, 200.0)
    u_back, v_back = calib.xy_to_pixel(x, y)
    assert math.isclose(u_back, 320.0, abs_tol=1e-4)
    assert math.isclose(v_back, 200.0, abs_tol=1e-4)

    p = tmp_path / "homography.json"
    calib.save(p)
    loaded = HomographyCalib.load(p)
    np.testing.assert_allclose(loaded.pixel_to_xy(320.0, 200.0), (x, y), atol=1e-6)


def test_ray_calib_to_homography_consistency():
    """실측 카메라 내부·외부 파라미터에서 유도한 평면 호모그래피가 3D 광선-평면 교차와 일치하는지 확인한다."""
    if not (config.CAMERA_INTRINSICS.exists() and config.CAMERA_EXTRINSICS.exists()):
        return
    ray = TopViewRayCalib.from_files(config.CAMERA_INTRINSICS, config.CAMERA_EXTRINSICS)
    z = 0.025
    h_calib = ray.to_homography(plane_z_m=z)
    for u, v in [(320.0, 240.0), (260.0, 180.0), (400.0, 300.0)]:
        p_ray = ray.to_plane(u, v, z)
        hx, hy = h_calib.pixel_to_xy(u, v)
        np.testing.assert_allclose([hx, hy], p_ray[:2], atol=1e-4)
