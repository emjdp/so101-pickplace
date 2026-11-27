"""검출기 Ablation 모듈(so101/detect_ablation.py) 단위 테스트."""

import cv2
import numpy as np

from so101.detect_ablation import (
    DETECTOR_ORDER,
    AblationParams,
    build_booth_roi_mask,
    calibrate_rgb_box_from_base,
    calibrate_rgb_diff_from_bg,
    run_detector,
)


def _make_scene(bg_bgr=(110, 114, 112), block_center=(250, 260), block_bgr=(65, 22, 12), size=32, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), bg_bgr, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-3, 4, size=img.shape), 0, 255).astype(np.uint8)
    if block_center is not None and block_bgr is not None:
        box = cv2.boxPoints((block_center, (size, size), 12.0)).astype(np.int32)
        cv2.fillPoly(img, [box], block_bgr)
    return img


def _default_params():
    return AblationParams(
        hsv_ranges=(((98, 80, 30), (130, 255, 255)),),
        rgb_lo=(0, 5, 45),
        rgb_hi=(30, 45, 95),
        rgb_diff_thresh=12.0,
        rgb_norm_thresh=0.45,
        rgb_norm_min_sum=20.0,
        bg_diff_thresh=35.0,
        bg_norm_diff_thresh=0.12,
    )


def test_all_detectors_succeed_on_base_blue():
    bg = _make_scene(block_center=None, seed=1)
    img = _make_scene(block_center=(255, 270), block_bgr=(65, 22, 12), seed=2)
    roi = build_booth_roi_mask(img.shape[:2])
    params = _default_params()

    for name in DETECTOR_ORDER:
        det, mask = run_detector(name, img, params, bg_bgr=bg, roi_mask=roi)
        assert det is not None, f"{name} failed on base blue"
        assert mask.shape == img.shape[:2] and mask.dtype == np.uint8
        assert np.hypot(det.u - 255, det.v - 270) < 3.0


def test_dim_light_exposes_hsv_vmin_and_rgb_box_while_norm_detectors_survive():
    bg_bright = _make_scene(bg_bgr=(110, 114, 112), block_center=None, seed=1)
    # 어두운 파란 블록: RGB=(2, 9, 24) -> BGR=(24, 9, 2)
    img_dim = _make_scene(bg_bgr=(46, 48, 48), block_center=(246, 270), block_bgr=(24, 9, 2), seed=3)
    roi = build_booth_roi_mask(img_dim.shape[:2])
    params = _default_params()

    det_hsv, _ = run_detector("hsv_orig", img_dim, params, roi_mask=roi)
    det_hsv_low, _ = run_detector("hsv_low_v", img_dim, params, roi_mask=roi)
    det_rgb_box, _ = run_detector("rgb_box", img_dim, params, roi_mask=roi)
    det_rgb_diff, _ = run_detector("rgb_diff", img_dim, params, roi_mask=roi)
    det_rgb_norm, _ = run_detector("rgb_norm", img_dim, params, roi_mask=roi)
    det_bg_raw, _ = run_detector("bg_diff", img_dim, params, bg_bgr=bg_bright, roi_mask=roi)
    det_bg_norm, _ = run_detector("bg_norm_diff", img_dim, params, bg_bgr=bg_bright, roi_mask=roi)

    assert det_hsv is None
    assert det_rgb_box is None
    assert det_hsv_low is not None and np.hypot(det_hsv_low.u - 246, det_hsv_low.v - 270) < 3.0
    assert det_rgb_diff is not None and np.hypot(det_rgb_diff.u - 246, det_rgb_diff.v - 270) < 3.0
    assert det_rgb_norm is not None and np.hypot(det_rgb_norm.u - 246, det_rgb_norm.v - 270) < 3.0
    assert det_bg_raw is None or np.hypot(det_bg_raw.u - 246, det_bg_raw.v - 270) > 20.0
    assert det_bg_norm is not None and np.hypot(det_bg_norm.u - 246, det_bg_norm.v - 270) < 3.0


def test_color_change_red_block_detected_by_bg_detectors():
    bg = _make_scene(bg_bgr=(160, 160, 160), block_center=None, seed=1)
    # 밝기가 드리프트된 배경 위 빨간 블록: BGR=(20, 25, 110)
    img_red = _make_scene(bg_bgr=(115, 115, 115), block_center=(250, 260), block_bgr=(20, 25, 110), seed=4)
    roi = build_booth_roi_mask(img_red.shape[:2])
    params = _default_params()

    for blue_only in ("hsv_orig", "hsv_low_v", "rgb_box", "rgb_diff", "rgb_norm"):
        det, _ = run_detector(blue_only, img_red, params, bg_bgr=bg, roi_mask=roi)
        assert det is None, f"{blue_only} should not detect red block"

    det_bg_norm, _ = run_detector("bg_norm_diff", img_red, params, bg_bgr=bg, roi_mask=roi)
    assert det_bg_norm is not None and np.hypot(det_bg_norm.u - 250, det_bg_norm.v - 260) < 3.0


def test_auto_calibrations():
    bg = _make_scene(bg_bgr=(195, 180, 178), block_center=None, seed=0)
    roi = build_booth_roi_mask(bg.shape[:2])
    t = calibrate_rgb_diff_from_bg(bg, roi_mask=roi)
    assert t >= 20.0

    imgs = [
        (_make_scene(block_center=(250, 260), block_bgr=(65, 22, 12), seed=i), 250.0, 260.0)
        for i in range(3)
    ]
    lo, hi = calibrate_rgb_box_from_base(imgs, patch_radius=5, margin=15)
    assert lo[2] >= 40 and hi[2] <= 90
    assert hi[0] <= 35 and hi[1] <= 45
