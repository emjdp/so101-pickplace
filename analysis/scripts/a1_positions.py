"""분석 1: 시연 첫 장면의 블록 위치 vs 시험 위치."""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
REPO = Path("/home/claude/repo/fresh/so101-pickplace"); sys.path.insert(0, str(REPO))
import config
from so101.detect_ablation import BOOTH_ROI_RECT, AblationParams, build_booth_roi_mask, run_detector
from so101.homography import HomographyCalib

W = Path("/home/claude/work"); OUT = W / "analysis/out"; OUT.mkdir(exist_ok=True)
calib = HomographyCalib.load(REPO / "calib/homography.json", intrinsics_path=REPO / "calib/camera_top.yaml")
bg = cv2.imread(str(W / "real/bg_empty.png"))
roi = build_booth_roi_mask(bg.shape, BOOTH_ROI_RECT)
hsv = json.loads((REPO / "calib/hsv_blue.json").read_text())["ranges"]
P = AblationParams(hsv_ranges=tuple((tuple(a), tuple(b)) for a, b in hsv))

rows = []
for f in sorted((W / "analysis/demo/first").glob("ep*.png")):
    img = cv2.imread(str(f)); i = int(f.stem[2:])
    d1, _ = run_detector("hsv_orig", img, P, bg, roi)
    d2, _ = run_detector("bg_norm_diff", img, P, bg, roi)
    r = {"episode": i}
    for tag, d in (("hsv", d1), ("bgn", d2)):
        r[f"{tag}_u"] = d.u if d else np.nan; r[f"{tag}_v"] = d.v if d else np.nan
        r[f"{tag}_angle"] = d.angle_deg if d else np.nan
    rows.append(r)
demo = pd.DataFrame(rows)
agree = np.hypot(demo.hsv_u - demo.bgn_u, demo.hsv_v - demo.bgn_v)
print("HSV 검출", demo.hsv_u.notna().sum(), "/ 50, 배경차분 검출", demo.bgn_u.notna().sum(), "/ 50")
print("두 검출기 위치 차이 중앙값 px", np.nanmedian(agree), "최대", np.nanmax(agree), "20px 넘는 것", int((agree > 20).sum()))
demo["u"], demo["v"] = demo.hsv_u, demo.hsv_v
xy = np.array([calib.pixel_to_xy(u, v) if np.isfinite(u) else (np.nan, np.nan) for u, v in zip(demo.u, demo.v)])
demo["x_mm"], demo["y_mm"] = xy[:, 0] * 1000, xy[:, 1] * 1000
demo.to_csv(OUT / "demo_positions.csv", index=False)

# 시험 위치: 시스템 A 실험 기록 (base, new_pos, light_change), 검출기 비교 사진 정답 (base, new_pos)
tr = pd.read_csv(REPO / "data/experiments/trials.csv")
A = tr[(tr.system == "A") & tr.u_px.notna()].copy()
A["x_mm"], A["y_mm"] = A.target_x_m * 1000, A.target_y_m * 1000
man = pd.read_csv(W / "real/manifest.csv")
man_xy = np.array([calib.pixel_to_xy(u, v) for u, v in zip(man.gt_u, man.gt_v)])
man["x_mm"], man["y_mm"] = man_xy[:, 0] * 1000, man_xy[:, 1] * 1000

D = demo[["x_mm", "y_mm"]].dropna().to_numpy()
def nearest(p): return float(np.min(np.hypot(D[:, 0] - p[0], D[:, 1] - p[1])))
from scipy.spatial import Delaunay
hull = Delaunay(D)
def stats(df, name):
    pts = df[["x_mm", "y_mm"]].to_numpy()
    dn = np.array([nearest(p) for p in pts]); inside = hull.find_simplex(pts) >= 0
    print(f"{name:<28} n={len(pts):2d}  시연 범위 안 {inside.sum():2d}  가장 가까운 시연까지 중앙값 {np.median(dn):5.1f} mm  최대 {dn.max():5.1f} mm")
    return dn, inside
res = {}
for cond in ["base", "new_pos", "light_change"]:
    res["A_" + cond] = stats(A[A.condition == cond], f"A 실험 {cond}")
for cond in ["base", "new_pos"]:
    res["M_" + cond] = stats(man[man.condition == cond], f"검출기 비교 사진 {cond}")
A["nearest_demo_mm"] = [nearest(p) for p in A[["x_mm", "y_mm"]].to_numpy()]
A["in_demo_hull"] = hull.find_simplex(A[["x_mm", "y_mm"]].to_numpy()) >= 0
A.to_csv(OUT / "A_positions.csv", index=False)
man.to_csv(OUT / "eval_positions.csv", index=False)
print("시연 위치 범위 x", np.nanmin(demo.x_mm).round(), "~", np.nanmax(demo.x_mm).round(), " y", np.nanmin(demo.y_mm).round(), "~", np.nanmax(demo.y_mm).round())
