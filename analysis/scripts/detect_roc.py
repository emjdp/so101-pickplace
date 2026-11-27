"""검출기 7종의 픽셀 단위 ROC 분석 (근사 정답 마스크 방식).

`so101.detect_ablation`의 검출기는 임계값을 넘으면 255, 아니면 0인 마스크만 낸다.
ROC를 그리려면 픽셀마다 "블록다운 정도"를 나타내는 연속 점수가 필요하므로, 각 검출기를
같은 판정 규칙을 유지한 채 연속 점수로 바꾼다. 점수 >= 기존 임계값이면 기존 마스크(형태학
처리 전)와 같은 판정이 된다.

정답 마스크는 손으로 칠하지 않고, 클릭한 블록 중심(gt_u, gt_v)과 블록 크기로 근사한다.
  - 양성(블록) 픽셀: 중심에서 r_pos 이내 (회전과 클릭 오차가 있어도 블록 안쪽인 영역)
  - 음성(배경) 픽셀: 중심에서 r_neg 보다 먼 ROI 내부 픽셀
  - 그 사이 고리 영역은 블록 가장자리일 수 있어 평가에서 제외한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from so101.detect_ablation import AblationParams

# 연속 점수가 정의되지 않는 픽셀(너무 어두워 색을 믿을 수 없는 픽셀 등)에 주는 값
NO_SCORE = -1.0


def _box_margin(values: np.ndarray, lo: np.ndarray, hi: np.ndarray, full: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """채널별 [lo, hi] 상자 안쪽으로 얼마나 들어와 있는지 (정규화한 최소 여유).

    values: (H, W, C). lo/hi/full/scale: (C,). full 은 채널 최댓값(H=179, S/V/RGB=255).
    lo=0 또는 hi=full 인 경계는 실제 제약이 아니므로 무시한다.
    반환값 >= 0 이면 상자 안쪽, < 0 이면 바깥.
    """
    v = values.astype(np.float32)
    margin = np.full(v.shape[:2], np.inf, dtype=np.float32)
    for c in range(v.shape[2]):
        if lo[c] > 0:
            margin = np.minimum(margin, (v[:, :, c] - lo[c]) / scale[c])
        if hi[c] < full[c]:
            margin = np.minimum(margin, (hi[c] - v[:, :, c]) / scale[c])
    margin[np.isinf(margin)] = 1.0
    return margin


def _hsv_score(bgr: np.ndarray, ranges) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    full = np.array([179, 255, 255], np.float32)
    scale = np.array([180, 255, 255], np.float32)
    score = None
    for lo, hi in ranges:
        s = _box_margin(hsv, np.array(lo, np.float32), np.array(hi, np.float32), full, scale)
        score = s if score is None else np.maximum(score, s)
    return score


def score_map(
    name: str,
    bgr: np.ndarray,
    params: AblationParams,
    bg_bgr: np.ndarray | None,
    roi_mask: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    """검출기 이름 -> (픽셀별 연속 점수 (H, W) float32, 기존 동작 임계값).

    점수 >= 임계값 인 픽셀이 기존 검출기의 (형태학 처리 전) 양성 픽셀과 같다.
    """
    f = bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    total = b + g + r

    if name == "hsv_orig":
        return _hsv_score(bgr, params.hsv_ranges), 0.0
    if name == "hsv_low_v":
        relaxed = [((lo[0], lo[1], min(lo[2], params.hsv_low_v_min)), hi) for lo, hi in params.hsv_ranges]
        return _hsv_score(bgr, relaxed), 0.0
    if name == "rgb_box":
        lo = np.array([params.rgb_lo[2], params.rgb_lo[1], params.rgb_lo[0]], np.float32)  # BGR 순서
        hi = np.array([params.rgb_hi[2], params.rgb_hi[1], params.rgb_hi[0]], np.float32)
        full = np.full(3, 255, np.float32)
        return _box_margin(bgr, lo, hi, full, full), 0.0
    if name == "rgb_diff":
        return b - np.maximum(r, g), float(params.rgb_diff_thresh)
    if name == "rgb_norm":
        s = b / (total + 1e-6)
        s[total < params.rgb_norm_min_sum] = NO_SCORE
        return s, float(params.rgb_norm_thresh)
    if name in ("bg_diff", "bg_norm_diff"):
        if bg_bgr is None:
            return np.full(bgr.shape[:2], NO_SCORE, np.float32), 0.0
        bgf = bg_bgr.astype(np.float32)
        if name == "bg_diff":
            d = f - bgf
            return np.sqrt(np.sum(d * d, axis=2)), float(params.bg_diff_thresh)
        tot_bg = bgf.sum(axis=2)
        ch_cur = f / (total[:, :, None] + 1e-6)
        ch_bg = bgf / (tot_bg[:, :, None] + 1e-6)
        sel = roi_mask > 0 if roi_mask is not None and np.any(roi_mask > 0) else np.ones(total.shape, bool)
        shift = np.median(ch_cur[sel], axis=0) - np.median(ch_bg[sel], axis=0)
        d = (ch_cur - shift) - ch_bg
        s = np.sqrt(np.sum(d * d, axis=2))
        s[total < params.rgb_norm_min_sum] = NO_SCORE
        return s, float(params.bg_norm_diff_thresh)
    raise ValueError(f"알 수 없는 검출기: {name}")


def label_masks(
    shape: tuple[int, int],
    gt_u: float,
    gt_v: float,
    roi_mask: np.ndarray | None,
    r_pos: float,
    r_neg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """근사 정답: (양성 픽셀 bool, 음성 픽셀 bool). 둘 다 ROI 안쪽만."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - gt_u, yy - gt_v)
    roi = roi_mask > 0 if roi_mask is not None else np.ones((h, w), bool)
    return (dist <= r_pos) & roi, (dist > r_neg) & roi


@dataclass
class RocResult:
    fpr: np.ndarray
    tpr: np.ndarray
    precision: np.ndarray
    auc: float
    ap: float
    auc_ci: tuple[float, float]
    op_tpr: float
    op_fpr: float
    op_precision: float
    n_pos: int
    n_neg: int


def _curve_from_counts(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """구간(bin)별 양성/음성 개수 -> ROC·PR 곡선과 AUC·AP. 높은 점수 구간부터 임계값을 내린다."""
    tp = np.concatenate([[0.0], np.cumsum(pos[::-1])])
    fp = np.concatenate([[0.0], np.cumsum(neg[::-1])])
    P, N = max(tp[-1], 1.0), max(fp[-1], 1.0)
    tpr, fpr = tp / P, fp / N
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
    ap = float(np.sum(np.diff(tpr) * prec[1:]))
    return fpr, tpr, prec, auc, ap


def build_edges(score_samples: np.ndarray, op_thresh: float, n_bins: int = 2000) -> np.ndarray:
    """점수 분포의 분위수로 구간 경계를 만든다. 동작 임계값은 정확히 경계에 포함한다."""
    q = np.quantile(score_samples, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(np.concatenate([q, [op_thresh]]).astype(np.float64))
    if edges.size < 2:
        edges = np.array([edges[0] - 1.0, edges[0] + 1.0])
    # 맨 위 경계를 살짝 올려 최댓값도 마지막 구간에 들어가게 한다
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def image_hist(scores: np.ndarray, pos: np.ndarray, neg: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """한 장의 양성/음성 픽셀 점수를 구간별 개수로 센다 (범위 밖 값은 양 끝 구간으로)."""
    lo, hi = edges[0], np.nextafter(edges[-1], -np.inf)
    hp, _ = np.histogram(np.clip(scores[pos], lo, hi), bins=edges)
    hn, _ = np.histogram(np.clip(scores[neg], lo, hi), bins=edges)
    return hp.astype(np.float64), hn.astype(np.float64)


def roc_from_hists(
    pos_hists: np.ndarray,
    neg_hists: np.ndarray,
    edges: np.ndarray,
    op_thresh: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> RocResult:
    """이미지별 구간 개수 (n_img, n_bins) -> ROC 결과. AUC 신뢰구간은 이미지 단위 부트스트랩."""
    pos, neg = pos_hists.sum(axis=0), neg_hists.sum(axis=0)
    fpr, tpr, prec, auc, ap = _curve_from_counts(pos, neg)

    k = int(np.searchsorted(edges, op_thresh, side="left"))  # 점수 >= op_thresh 인 구간들
    op_tp, op_fp = pos[k:].sum(), neg[k:].sum()
    op_tpr = float(op_tp / max(pos.sum(), 1.0))
    op_fpr = float(op_fp / max(neg.sum(), 1.0))
    op_prec = float(op_tp / (op_tp + op_fp)) if op_tp + op_fp > 0 else float("nan")

    rng = np.random.default_rng(seed)
    n = pos_hists.shape[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(_curve_from_counts(pos_hists[idx].sum(axis=0), neg_hists[idx].sum(axis=0))[3])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (auc, auc)

    return RocResult(fpr, tpr, prec, auc, ap, ci, op_tpr, op_fpr, op_prec, int(pos.sum()), int(neg.sum()))
