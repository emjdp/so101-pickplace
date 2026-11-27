"""블록 검출기 비교(Ablation) 모듈: HSV vs RGB / RGB 차분 / 정규화 RGB / 배경 RGB 차분.

기존 시스템 A의 `so101.detect.find_block()` 후처리(5x5 OPEN->CLOSE, 최대 윤곽선, 모멘트 중심,
minAreaRect 각도)와 동일한 부스 내부 작업대 ROI를 모든 방식에 공통 적용해 순수 색/차분 분할
성능만 공정하게 비교한다.

비교 대상 7종:
  1. hsv_orig       : 기존 HSV 범위 (calib/hsv_blue.json, V_min=30)
  2. hsv_low_v      : 기존 HSV에서 V_min만 15로 완화 (조명 저하 실패 원인 격리용)
  3. rgb_box        : 단순 RGB 절대 범위 임계값 (base 조건 파란 블록 분포로 보정)
  4. rgb_diff       : RGB 채널 차분 (Excess-Blue: B - max(R, G) >= T, 빈 배경 잡음 상한 기준 보정)
  5. rgb_norm       : 정규화 RGB (Chromaticity: B / (R + G + B) >= T, 암부 노이즈 가드 포함)
  6. bg_diff        : 단순 배경 RGB 차분 (||I - I_bg||_2 >= T, 색 무관)
  7. bg_norm_diff   : 정규화 RGB 배경 차분 (||c(I) - shift - c(I_bg)||_2 >= T, 밝기 불변·색 무관)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from so101.detect import Detection, KERNEL, find_block, make_mask

# 부스 내부 작업대 바닥 픽셀 영역 (x0, y0, x1, y1): 부스 바깥 책상·로봇 베이스 제외
BOOTH_ROI_RECT: tuple[int, int, int, int] = (140, 95, 485, 375)

DETECTOR_ORDER = (
    "hsv_orig",
    "hsv_low_v",
    "rgb_box",
    "rgb_diff",
    "rgb_norm",
    "bg_diff",
    "bg_norm_diff",
)

DETECTOR_LABELS: dict[str, str] = {
    "hsv_orig": "HSV (기존, V≥30)",
    "hsv_low_v": "HSV (V≥15 완화)",
    "rgb_box": "단순 RGB 범위",
    "rgb_diff": "RGB 채널 차분 (B-max(R,G))",
    "rgb_norm": "정규화 RGB (B/(R+G+B))",
    "bg_diff": "단순 배경 차분 (||I-I_bg||)",
    "bg_norm_diff": "정규화 RGB 배경 차분",
}

DETECTOR_SHORT_LABELS: dict[str, str] = {
    "hsv_orig": "HSV\n(기존)",
    "hsv_low_v": "HSV\n(V≥15)",
    "rgb_box": "단순 RGB\n범위",
    "rgb_diff": "RGB 채널차분\nB-max(R,G)",
    "rgb_norm": "정규화 RGB\nB/(R+G+B)",
    "bg_diff": "단순 배경차분\n||I-I_bg||",
    "bg_norm_diff": "정규화 RGB\n배경 차분",
}


@dataclass(frozen=True)
class AblationParams:
    """각 검출기의 임계값 파라미터 (모두 base 조건 및 빈 배경에서만 결정)."""

    hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    hsv_low_v_min: int = 15
    # 직관적인 (R, G, B) 순서로 저장
    rgb_lo: tuple[int, int, int] = (0, 0, 40)
    rgb_hi: tuple[int, int, int] = (45, 65, 140)
    rgb_diff_thresh: float = 12.0
    rgb_norm_thresh: float = 0.45
    rgb_norm_min_sum: float = 20.0
    bg_diff_thresh: float = 35.0
    bg_norm_diff_thresh: float = 0.12
    min_area: float = 150.0


def _morph(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def _apply_roi(mask: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray:
    if roi_mask is None:
        return mask
    return cv2.bitwise_and(mask, roi_mask)


def build_workspace_mask(shape: tuple[int, int], poly_px: np.ndarray | None) -> np.ndarray | None:
    """작업영역 다각형 꼭짓점 (N, 2)으로부터 (H, W) uint8 마스크(내부 255)를 만든다."""
    if poly_px is None:
        return None
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(poly_px, dtype=np.int32)], 255)
    return mask


def build_booth_roi_mask(
    shape: tuple[int, int],
    rect: tuple[int, int, int, int] = BOOTH_ROI_RECT,
) -> np.ndarray:
    """부스 내부 작업대 바닥 직사각형 (x0, y0, x1, y1) 마스크 (H, W) uint8을 만든다."""
    x0, y0, x1, y1 = rect
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[max(0, y0) : min(shape[0], y1), max(0, x0) : min(shape[1], x1)] = 255
    return mask


def calibrate_rgb_box_from_base(
    base_samples: list[tuple[np.ndarray, float, float]],
    patch_radius: int = 5,
    margin: int = 15,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """base 조건 프레임의 블록 중심 패치(GT)만으로 단순 RGB 범위 (rgb_lo, rgb_hi)를 보정한다.

    다른 조건(light_change, color_change 등)을 보지 않고 base 분포만 사용해 공정성을 보장한다.
    반환값은 (R, G, B) 순서의 (lo, hi) 튜플이다.
    """
    patches = []
    for bgr, u, v in base_samples:
        h, w = bgr.shape[:2]
        cx, cy = int(round(u)), int(round(v))
        x0, x1 = max(0, cx - patch_radius), min(w, cx + patch_radius + 1)
        y0, y1 = max(0, cy - patch_radius), min(h, cy + patch_radius + 1)
        if x1 > x0 and y1 > y0:
            rgb_patch = bgr[y0:y1, x0:x1, ::-1].reshape(-1, 3)
            patches.append(rgb_patch)
    if not patches:
        return (0, 0, 40), (45, 65, 140)
    all_rgb = np.vstack(patches).astype(np.float32)
    p2 = np.percentile(all_rgb, 2, axis=0)
    p98 = np.percentile(all_rgb, 98, axis=0)
    lo = tuple(int(np.clip(np.floor(v - margin), 0, 255)) for v in p2)
    hi = tuple(int(np.clip(np.ceil(v + margin), 0, 255)) for v in p98)
    return lo, hi


def calibrate_rgb_diff_from_bg(
    bg_bgr: np.ndarray | None,
    roi_mask: np.ndarray | None = None,
    percentile: float = 99.5,
    margin: float = 6.0,
    min_thresh: float = 12.0,
) -> float:
    """빈 작업대 배경의 B - max(R, G) 잡음 상한으로부터 RGB 채널 차분 임계값을 자동 보정한다."""
    if bg_bgr is None:
        return min_thresh
    b = bg_bgr[:, :, 0].astype(np.int16)
    g = bg_bgr[:, :, 1].astype(np.int16)
    r = bg_bgr[:, :, 2].astype(np.int16)
    score = b - np.maximum(r, g)
    vals = score[roi_mask > 0] if roi_mask is not None else score.ravel()
    if vals.size == 0:
        return min_thresh
    return float(max(min_thresh, np.ceil(float(np.percentile(vals, percentile)) + margin)))


def make_mask_hsv_orig(bgr: np.ndarray, params: AblationParams, roi_mask: np.ndarray | None = None) -> np.ndarray:
    return _apply_roi(make_mask(bgr, list(params.hsv_ranges)), roi_mask)


def make_mask_hsv_low_v(bgr: np.ndarray, params: AblationParams, roi_mask: np.ndarray | None = None) -> np.ndarray:
    relaxed = [
        ((lo[0], lo[1], min(lo[2], params.hsv_low_v_min)), hi)
        for lo, hi in params.hsv_ranges
    ]
    return _apply_roi(make_mask(bgr, relaxed), roi_mask)


def make_mask_rgb_box(bgr: np.ndarray, params: AblationParams, roi_mask: np.ndarray | None = None) -> np.ndarray:
    # OpenCV 배열은 BGR이므로 (R, G, B) -> (B, G, R)로 뒤집어 inRange 적용
    lo_bgr = np.array([params.rgb_lo[2], params.rgb_lo[1], params.rgb_lo[0]], dtype=np.uint8)
    hi_bgr = np.array([params.rgb_hi[2], params.rgb_hi[1], params.rgb_hi[0]], dtype=np.uint8)
    mask = cv2.inRange(bgr, lo_bgr, hi_bgr)
    return _morph(_apply_roi(mask, roi_mask))


def make_mask_rgb_diff(bgr: np.ndarray, params: AblationParams, roi_mask: np.ndarray | None = None) -> np.ndarray:
    """RGB 채널 차분 (Excess-Blue): B - max(R, G) >= thresh."""
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    score = b - np.maximum(r, g)
    mask = np.where(score >= params.rgb_diff_thresh, 255, 0).astype(np.uint8)
    return _morph(_apply_roi(mask, roi_mask))


def make_mask_rgb_norm(bgr: np.ndarray, params: AblationParams, roi_mask: np.ndarray | None = None) -> np.ndarray:
    """정규화 RGB (Chromaticity): B / (R + G + B) >= thresh, R + G + B >= min_sum."""
    bgr_f = bgr.astype(np.float32)
    total = bgr_f[:, :, 0] + bgr_f[:, :, 1] + bgr_f[:, :, 2]
    b_norm = bgr_f[:, :, 0] / (total + 1e-6)
    mask = np.where((total >= params.rgb_norm_min_sum) & (b_norm >= params.rgb_norm_thresh), 255, 0).astype(np.uint8)
    return _morph(_apply_roi(mask, roi_mask))


def make_mask_bg_diff(
    bgr: np.ndarray,
    params: AblationParams,
    bg_bgr: np.ndarray | None,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """단순 배경 RGB 차분: ||I(x,y) - I_bg(x,y)||_2 >= thresh."""
    if bg_bgr is None:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    diff = bgr.astype(np.float32) - bg_bgr.astype(np.float32)
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.where(dist >= params.bg_diff_thresh, 255, 0).astype(np.uint8)
    return _morph(_apply_roi(mask, roi_mask))


def make_mask_bg_norm_diff(
    bgr: np.ndarray,
    params: AblationParams,
    bg_bgr: np.ndarray | None,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """정규화 RGB 배경 차분: 밝기 스케일·전역 색온도 이동을 상쇄한 색도 차분 ||c(I) - shift - c(I_bg)||_2 >= thresh."""
    if bg_bgr is None:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    cur_f = bgr.astype(np.float32)
    bg_f = bg_bgr.astype(np.float32)
    tot_cur = cur_f[:, :, 0] + cur_f[:, :, 1] + cur_f[:, :, 2]
    tot_bg = bg_f[:, :, 0] + bg_f[:, :, 1] + bg_f[:, :, 2]
    ch_cur = cur_f / (tot_cur[:, :, None] + 1e-6)
    ch_bg = bg_f / (tot_bg[:, :, None] + 1e-6)

    if roi_mask is not None and np.any(roi_mask > 0):
        sel = roi_mask > 0
        shift = np.median(ch_cur[sel], axis=0) - np.median(ch_bg[sel], axis=0)
    else:
        shift = np.median(ch_cur.reshape(-1, 3), axis=0) - np.median(ch_bg.reshape(-1, 3), axis=0)

    diff = (ch_cur - shift) - ch_bg
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.where((dist >= params.bg_norm_diff_thresh) & (tot_cur >= params.rgb_norm_min_sum), 255, 0).astype(np.uint8)
    return _morph(_apply_roi(mask, roi_mask))


def run_detector(
    name: str,
    bgr: np.ndarray,
    params: AblationParams,
    bg_bgr: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
) -> tuple[Detection | None, np.ndarray]:
    """지정한 검출기 1종을 실행해 (Detection | None, 이진 마스크)를 반환한다."""
    if name == "hsv_orig":
        mask = make_mask_hsv_orig(bgr, params, roi_mask=roi_mask)
    elif name == "hsv_low_v":
        mask = make_mask_hsv_low_v(bgr, params, roi_mask=roi_mask)
    elif name == "rgb_box":
        mask = make_mask_rgb_box(bgr, params, roi_mask=roi_mask)
    elif name == "rgb_diff":
        mask = make_mask_rgb_diff(bgr, params, roi_mask=roi_mask)
    elif name == "rgb_norm":
        mask = make_mask_rgb_norm(bgr, params, roi_mask=roi_mask)
    elif name == "bg_diff":
        mask = make_mask_bg_diff(bgr, params, bg_bgr=bg_bgr, roi_mask=roi_mask)
    elif name == "bg_norm_diff":
        mask = make_mask_bg_norm_diff(bgr, params, bg_bgr=bg_bgr, roi_mask=roi_mask)
    else:
        raise ValueError(f"알 수 없는 검출기: {name}")
    det = find_block(mask, min_area=params.min_area)
    return det, mask
