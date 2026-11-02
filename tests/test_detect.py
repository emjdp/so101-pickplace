"""합성 영상으로 블록 검출을 확인한다. 카메라 없이 돌아간다."""

import cv2
import numpy as np
import pytest

from so101.detect import detect_block, make_mask

BLUE = [((100, 120, 60), (130, 255, 255))]
RED = [((0, 120, 60), (10, 255, 255)), ((170, 120, 60), (179, 255, 255))]


def synth(center, size, angle, color_bgr, seed=0):
    """회색 바탕 + 잡음 위에 회전한 사각형 블록 하나를 그린다."""
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), 110, np.uint8)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    box = cv2.boxPoints((center, (size, size), angle)).astype(np.int32)
    cv2.fillPoly(img, [box], color_bgr)
    return img


def angle_diff_mod90(a, b):
    d = (a - b) % 90
    return min(d, 90 - d)


def test_mask_is_binary():
    img = synth((320, 240), 40, 0, (200, 60, 30))
    m = make_mask(img, BLUE)
    assert m.shape == img.shape[:2] and m.dtype == np.uint8
    assert set(np.unique(m)) <= {0, 255}


@pytest.mark.parametrize("center,angle", [((320, 240), 0), ((150, 100), 20), ((500, 380), 37), ((260, 330), 71)])
def test_center_and_angle(center, angle):
    img = synth(center, 40, angle, (200, 60, 30))
    det = detect_block(img, BLUE)
    assert det is not None
    assert abs(det.u - center[0]) < 1.0 and abs(det.v - center[1]) < 1.0
    assert 0 <= det.angle_deg < 90
    assert angle_diff_mod90(det.angle_deg, angle) < 2.0


def test_red_needs_two_hue_ranges():
    img = synth((300, 200), 40, 15, (30, 30, 200))
    assert detect_block(img, RED) is not None


def test_small_blobs_are_rejected():
    img = synth((320, 240), 40, 0, (200, 60, 30))
    cv2.circle(img, (50, 50), 3, (200, 60, 30), -1)  # 작은 잡음 덩어리
    det = detect_block(img, BLUE)
    assert abs(det.u - 320) < 1.0, "가장 큰 덩어리를 골라야 한다"


def test_none_when_no_block():
    img = synth((320, 240), 40, 0, (110, 110, 110))
    assert detect_block(img, BLUE) is None
