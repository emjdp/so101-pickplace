"""픽셀 단위 ROC 모듈(so101/detect_roc.py) 단위 테스트."""

import cv2
import numpy as np
import pytest

from so101.detect_ablation import DETECTOR_ORDER, AblationParams, build_booth_roi_mask
from so101.detect_roc import build_edges, image_hist, label_masks, roc_from_hists, score_map

# make_mask_* 의 형태학 처리 전 이진 판정을 그대로 재현한 기준 구현
from so101 import detect_ablation as da


def _scene(seed=0, block_bgr=(150, 60, 20), center=(300, 230), bg=(110, 114, 112), size=26):
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), bg, np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-25, 26, img.shape), 0, 255).astype(np.uint8)
    if center is not None:
        box = cv2.boxPoints((center, (size, size), 20.0)).astype(np.int32)
        cv2.fillPoly(img, [box], block_bgr)
    return img


def _params():
    return AblationParams(
        hsv_ranges=(((98, 80, 30), (130, 255, 255)),),
        rgb_lo=(0, 30, 120),
        rgb_hi=(60, 100, 200),
        rgb_diff_thresh=26.0,
    )


def _raw_mask(name, bgr, p, bg):
    """기존 검출기의 형태학 처리 전 판정 (ROI 없음)."""
    if name == "hsv_orig":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return np.any([cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0 for lo, hi in p.hsv_ranges], axis=0)
    if name == "hsv_low_v":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return np.any(
            [cv2.inRange(hsv, np.array((lo[0], lo[1], min(lo[2], p.hsv_low_v_min))), np.array(hi)) > 0 for lo, hi in p.hsv_ranges],
            axis=0,
        )
    if name == "rgb_box":
        lo = np.array(p.rgb_lo[::-1], np.uint8)
        hi = np.array(p.rgb_hi[::-1], np.uint8)
        return cv2.inRange(bgr, lo, hi) > 0
    if name == "rgb_diff":
        b, g, r = [bgr[:, :, i].astype(np.int16) for i in range(3)]
        return (b - np.maximum(r, g)) >= p.rgb_diff_thresh
    if name == "rgb_norm":
        f = bgr.astype(np.float32)
        t = f.sum(axis=2)
        return (t >= p.rgb_norm_min_sum) & (f[:, :, 0] / (t + 1e-6) >= p.rgb_norm_thresh)
    if name == "bg_diff":
        d = bgr.astype(np.float32) - bg.astype(np.float32)
        return np.sqrt((d * d).sum(axis=2)) >= p.bg_diff_thresh
    return None  # bg_norm_diff 는 ROI 중앙값 보정이 들어가 아래 테스트에서 따로 확인


@pytest.mark.parametrize("name", [n for n in DETECTOR_ORDER if n != "bg_norm_diff"])
def test_score_threshold_reproduces_original_decision(name):
    p = _params()
    bg = _scene(seed=1, center=None)
    img = _scene(seed=2)
    s, op = score_map(name, img, p, bg, None)
    ref = _raw_mask(name, img, p, bg)
    agree = np.mean((s >= op) == ref)
    assert agree > 0.9999, f"{name}: 일치율 {agree:.5f}"


def test_bg_norm_diff_matches_original_before_morphology(monkeypatch):
    monkeypatch.setattr(da, "_morph", lambda m: m)  # 형태학 처리를 끄고 픽셀 판정만 비교
    p = _params()
    bg = _scene(seed=1, center=None)
    img = _scene(seed=3)
    roi = build_booth_roi_mask(img.shape)
    s, op = score_map("bg_norm_diff", img, p, bg, roi)
    orig = da.make_mask_bg_norm_diff(img, p, bg, roi)
    inside = roi > 0
    assert np.mean(((s >= op) & inside) == (orig > 0)) > 0.9999


def test_label_masks_split_center_and_far():
    roi = build_booth_roi_mask((480, 640))
    pos, neg = label_masks((480, 640), 300, 230, roi, 6, 30)
    assert pos[230, 300] and not neg[230, 300]
    assert not pos[230, 320] and not neg[230, 320]  # 제외 고리
    assert neg[230, 400] and not pos[230, 400]
    assert not (pos & neg).any()


def test_roc_perfect_and_random():
    rng = np.random.default_rng(0)
    # 완벽 분리
    sp, sn = rng.uniform(2, 3, 500), rng.uniform(0, 1, 5000)
    s = np.concatenate([sp, sn])
    pos = np.zeros(s.size, bool)
    pos[:500] = True
    edges = build_edges(s, 1.5)
    ph, nh = image_hist(s, pos, ~pos, edges)
    r = roc_from_hists(ph[None], nh[None], edges, 1.5, n_boot=0)
    assert r.auc == pytest.approx(1.0, abs=1e-6)
    assert r.op_tpr == pytest.approx(1.0) and r.op_fpr == pytest.approx(0.0)
    # 같은 분포 -> AUC 0.5 근처
    s2 = rng.uniform(0, 1, 20000)
    pos2 = rng.random(s2.size) < 0.5
    edges2 = build_edges(s2, 0.5)
    ph2, nh2 = image_hist(s2, pos2, ~pos2, edges2)
    r2 = roc_from_hists(ph2[None], nh2[None], edges2, 0.5, n_boot=0)
    assert r2.auc == pytest.approx(0.5, abs=0.02)
    assert r2.op_tpr == pytest.approx(0.5, abs=0.02)
