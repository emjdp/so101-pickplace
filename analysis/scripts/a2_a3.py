"""분석 2 (밝기 분포), 분석 3 (그리퍼 개도, 동작 시간)."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
REPO = Path("/home/claude/repo/fresh/so101-pickplace"); sys.path.insert(0, str(REPO))
from so101.detect_ablation import BOOTH_ROI_RECT, build_booth_roi_mask
W = Path("/home/claude/work"); OUT = W / "analysis/out"
roi = build_booth_roi_mask((480, 640), BOOTH_ROI_RECT) > 0
def mean_v(p):
    img = cv2.imread(str(p)); return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2][roi].mean())
rows = [{"source": "demo_first", "file": p.name, "mean_v": mean_v(p)} for p in sorted((W / "analysis/demo/first").glob("*.png"))]
rows += [{"source": "demo_sample", "file": p.name, "mean_v": mean_v(p)} for p in sorted((W / "analysis/demo/sample").glob("*.png"))]
man = pd.read_csv(W / "real/manifest.csv")
for _, r in man.iterrows():
    rows.append({"source": "eval_" + r.condition, "file": r.filename, "mean_v": mean_v(W / "real" / r.filename), "mean_v_manifest": r.mean_v})
b = pd.DataFrame(rows); b.to_csv(OUT / "brightness.csv", index=False)
chk = b.dropna(subset=["mean_v_manifest"]); print("manifest 밝기와 재계산 차이 최대", (chk.mean_v - chk.mean_v_manifest).abs().max().round(2))
print(b.groupby("source").mean_v.describe()[["count", "min", "50%", "max"]].round(1))
lo = b[b.source.str.startswith("demo")].mean_v.min()
lc = b[b.source == "eval_light_change"]
print(f"시연 최저 밝기 {lo:.1f}; 조명 변경 사진 중 이보다 어두운 것 {int((lc.mean_v < lo).sum())}/20")

# 분석 3: 그리퍼
j = pd.read_csv(W / "analysis/demo/joints.csv")
hold = []
for e, g in j.groupby("episode"):
    gs = g.state_gripper.to_numpy(); ga = g.action_gripper.to_numpy()
    # 쥐고 있는 구간: 명령(리더)은 거의 닫혔는데(하위 구간) 실제 개도는 블록에 막혀 멈춘 구간
    closed_cmd = ga < (ga.min() + 0.25 * (ga.max() - ga.min()))
    seg = gs[closed_cmd]
    hold.append({"episode": e, "open_max": gs.max(), "cmd_min": ga.min(), "hold_median": float(np.median(seg)) if len(seg) else np.nan,
                 "hold_p05": float(np.percentile(seg, 5)) if len(seg) else np.nan, "n_hold": int(closed_cmd.sum()),
                 "duration_s": len(g) / 30.0})
h = pd.DataFrame(hold); h.to_csv(OUT / "demo_gripper.csv", index=False)
print(h[["open_max", "cmd_min", "hold_median", "hold_p05", "n_hold", "duration_s"]].describe().round(2))
tr = pd.read_csv(REPO / "data/experiments/trials.csv")
print("A 빈손 닫힘 개도:", sorted(set(tr[tr.note.str.contains("gripper=", na=False)].note.str.extract(r"gripper=([\d.]+)")[0])))
for s in "AB":
    t = tr[(tr.system == s) & (tr.success == 1)].elapsed_s
    print(s, "성공 시행 소요 시간 중앙값", t.median(), "범위", t.min(), t.max(), "n", len(t))
print("시연 길이 중앙값", h.duration_s.median().round(2), "범위", h.duration_s.min().round(2), h.duration_s.max().round(2))
