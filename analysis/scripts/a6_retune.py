"""분석 6: 기본 조건 사진만으로 기준값을 다시 고르고, 나머지 조건에 그대로 적용한다.

선택 규칙 (기본 조건 20장 + 빈 배경만 사용):
  1) 후보 기준값마다 전체 파이프라인(점수 >= t -> ROI -> 5x5 OPEN/CLOSE -> 가장 큰 덩어리)을 돌린다.
  2) 기본 20장을 모두 맞게 찾고(중심 20 px 이내) 빈 배경에서는 아무것도 안 찾는 기준값만 남긴다.
  3) 남은 구간의 한가운데(중앙값)를 고른다 = 양쪽 실패 경계에서 가장 먼, 여유가 큰 기준값.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
REPO = Path("/home/claude/repo/fresh/so101-pickplace"); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from so101.detect import KERNEL, find_block
from so101.detect_ablation import BOOTH_ROI_RECT, DETECTOR_ORDER, DETECTOR_LABELS, build_booth_roi_mask
from so101.detect_roc import score_map
from eval_detector_roc import load_items, make_params
W = Path("/home/claude/work"); DATA = W / "real"

class A: hsv_low_v = 15; rgb_norm_thresh = 0.45; bg_diff_thresh = 35.0; bg_norm_diff_thresh = 0.12
items = load_items(DATA); bg = cv2.imread(str(DATA / "bg_empty.png")); roi = build_booth_roi_mask(bg.shape, BOOTH_ROI_RECT)
P = make_params(items, bg, roi, A)

def detect(score, t):
    m = np.where(score >= t, 255, 0).astype(np.uint8); m = cv2.bitwise_and(m, roi)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL), cv2.MORPH_CLOSE, KERNEL)
    return find_block(m, min_area=P.min_area)

def ok(d, it): return d is not None and np.hypot(d.u - it["gt_u"], d.v - it["gt_v"]) <= 20.0

conds = ["base", "new_pos", "color_change", "light_change"]
rows, chosen = [], {}
for name in DETECTOR_ORDER:
    scores = [score_map(name, it["img"], P, bg, roi)[0] for it in items]
    op = score_map(name, items[0]["img"], P, bg, roi)[1]
    s_bg = score_map(name, bg, P, bg, roi)[0]
    base_idx = [i for i, it in enumerate(items) if it["condition"] == "base"]
    pool = np.concatenate([scores[i][roi > 0] for i in base_idx])
    cands = np.unique(np.quantile(pool, np.linspace(0.5, 1.0, 400)))
    cands = np.unique(np.concatenate([cands, [op]]))
    valid = []
    for t in cands:
        if detect(s_bg, t) is not None and name not in ("bg_diff", "bg_norm_diff"):
            continue  # 빈 배경에서 무언가를 찾으면 탈락 (배경 차분은 자기 자신과 비교라 항상 통과)
        if all(ok(detect(scores[i], t), items[i]) for i in base_idx):
            valid.append(t)
    t_new = float(np.median(valid)) if valid else op
    chosen[name] = (op, t_new, len(valid), (min(valid), max(valid)) if valid else (np.nan, np.nan))
    for label, t in (("기존", op), ("재조정", t_new)):
        r = {"detector": name, "setting": label, "threshold": t}
        for c in conds:
            idx = [i for i, it in enumerate(items) if it["condition"] == c]
            r[c] = sum(ok(detect(scores[i], t), items[i]) for i in idx)
        r["total"] = sum(r[c] for c in conds); rows.append(r)
    print(f"{DETECTOR_LABELS[name]:<26} 기존 t={op:.4g}  유효 구간 [{chosen[name][3][0]:.4g}, {chosen[name][3][1]:.4g}] ({len(valid)}개)  -> t={t_new:.4g}")
df = pd.DataFrame(rows); df.to_csv(W / "analysis/out/retune.csv", index=False)
print(df.to_string(index=False))
