import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from scipy.spatial import ConvexHull
fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; fm.fontManager.addfont(fp)
plt.rcParams.update({"font.family": fm.FontProperties(fname=fp).get_name(), "axes.unicode_minus": False, "axes.edgecolor": "#BBBBBB",
                     "axes.spines.top": False, "axes.spines.right": False})
W = Path("/home/claude/work"); O = W / "analysis/out"; F = W / "final/figures"; F.mkdir(exist_ok=True)
REPO = Path("/home/claude/repo/fresh/so101-pickplace")
CL = {"base": "기본", "new_pos": "새 위치", "color_change": "색 변경", "light_change": "조명 변경"}
C_A, C_B, C_DEMO, C_BAD = "#2B5B84", "#E07B39", "#2E9E6B", "#C0392B"
bg = cv2.cvtColor(cv2.imread(str(W / "real/bg_empty.png")), cv2.COLOR_BGR2RGB)

# ---------- 분석 1: 시연 위치 vs 시험 위치 ----------
demo = pd.read_csv(O / "demo_positions.csv"); A = pd.read_csv(O / "A_positions.csv"); ev = pd.read_csv(O / "eval_positions.csv")
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
for a in ax:
    a.imshow((bg * 0.55 + 255 * 0.45).astype(np.uint8)); a.set_xlim(130, 500); a.set_ylim(380, 85); a.set_xticks([]); a.set_yticks([])
    pts = demo[["u", "v"]].to_numpy(); h = ConvexHull(pts); poly = pts[np.r_[h.vertices, h.vertices[0]]]
    a.fill(poly[:, 0], poly[:, 1], color=C_DEMO, alpha=0.15); a.plot(poly[:, 0], poly[:, 1], color=C_DEMO, lw=1.5)
    a.scatter(pts[:, 0], pts[:, 1], s=28, color=C_DEMO, edgecolor="white", lw=0.6, label="시연 50개 시작 위치", zorder=3)
nb = A[A.condition == "new_pos"]
ax[0].scatter(nb.u_px, nb.v_px, s=70, marker="X", c=np.where(nb.in_demo_hull, "#555555", C_BAD), edgecolor="white", lw=0.6, zorder=4)
ax[0].scatter([], [], marker="X", color=C_BAD, label="새 위치 시험 - 시연 범위 밖 (9곳)"); ax[0].scatter([], [], marker="X", color="#555555", label="새 위치 시험 - 시연 범위 안 (10곳)")
ab = A[A.condition == "base"]; ax[0].scatter(ab.u_px, ab.v_px, s=30, marker="s", color=C_A, label="기본 시험 위치 (20회)", zorder=4)
ax[0].set_title("시스템 A 실험에 쓴 시험 위치", fontsize=13); ax[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
en = ev[ev.condition == "new_pos"]; from scipy.spatial import Delaunay
ins = Delaunay(pts).find_simplex(en[["gt_u", "gt_v"]].to_numpy()) >= 0
ax[1].scatter(en.gt_u, en.gt_v, s=70, marker="X", c=np.where(ins, "#555555", C_BAD), edgecolor="white", lw=0.6, zorder=4)
ax[1].scatter([], [], marker="X", color=C_BAD, label=f"새 위치 사진 - 시연 범위 밖 ({(~ins).sum()}곳)"); ax[1].scatter([], [], marker="X", color="#555555", label=f"새 위치 사진 - 시연 범위 안 ({ins.sum()}곳)")
ax[1].set_title("검출기 비교 사진의 새 위치", fontsize=13); ax[1].legend(loc="lower left", fontsize=9, framealpha=0.9)
fig.suptitle("ACT 시연 시작 위치(초록 영역)와 '새 위치' 시험 위치", fontsize=15); fig.tight_layout(); fig.savefig(F / "fig3_demo_coverage.png", dpi=250); plt.close(fig)

# ---------- 분석 2: 밝기 ----------
b = pd.read_csv(O / "brightness.csv")
groups = [("demo_sample", "ACT 시연 영상\n(320 프레임)", C_DEMO), ("eval_base", "기본", "#8E9AAF"), ("eval_new_pos", "새 위치", "#8E9AAF"),
          ("eval_color_change", "색 변경", "#8E9AAF"), ("eval_light_change", "조명 변경", C_BAD)]
fig, ax = plt.subplots(figsize=(10, 4.8)); lo = b[b.source.str.startswith("demo")].mean_v.min()
ax.axvspan(lo, b[b.source.str.startswith("demo")].mean_v.max(), color=C_DEMO, alpha=0.1)
for i, (src, lab, col) in enumerate(groups):
    v = b[b.source == src].mean_v.to_numpy(); y = i + np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
    ax.scatter(v, y, s=14 if i == 0 else 30, color=col, alpha=0.6 if i == 0 else 0.85, edgecolor="none")
    ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="black", lw=2)
ax.axvline(lo, color=C_DEMO, ls="--", lw=1.2); ax.text(lo - 2, 4.45, f"시연 최저 밝기 {lo:.0f}", ha="right", fontsize=10, color=C_DEMO)
ax.set_yticks(range(len(groups)), [g[1] for g in groups]); ax.invert_yaxis(); ax.set_xlim(0, 240)
ax.set_xlabel("작업대 영역 평균 밝기 (HSV의 V, 0=검정 255=흰색) · 검은 막대 = 중앙값")
ax.set_title("시연 영상과 시험 사진의 밝기 비교: 조명 변경 사진 20장 모두 시연보다 어둡다", fontsize=13)
fig.tight_layout(); fig.savefig(F / "fig4_brightness.png", dpi=250); plt.close(fig)

# ---------- 분석 3: 그리퍼 + 속도 ----------
j = pd.read_csv(W / "analysis/demo/joints.csv"); gh = pd.read_csv(O / "demo_gripper.csv")
tr = pd.read_csv(REPO / "data/experiments/trials.csv")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.4, 1, 1]})
g = j[j.episode == 0]; ax[0].plot(g.t, g.state_gripper, color=C_A, lw=2, label="실제 개도 (팔로워)"); ax[0].plot(g.t, g.action_gripper, color="#999999", ls="--", label="명령 (리더암)")
ax[0].axhspan(gh.hold_min.min(), gh.hold_median.max(), color=C_DEMO, alpha=0.15); ax[0].text(0.1, 12.5, "블록을 쥔 구간", color=C_DEMO, fontsize=10, va="bottom")
ax[0].axhline(4, color=C_BAD, ls=":", lw=1.5); ax[0].text(6.3, 4.6, "A 판정 기준 4%", color=C_BAD, ha="right", fontsize=9)
ax[0].set_xlabel("시간 [s]"); ax[0].set_ylabel("그리퍼 개도 [%]"); ax[0].set_title("시연 1개(ep0)의 그리퍼 움직임", fontsize=12); ax[0].legend(fontsize=9, loc="upper right")
ax[1].hist(gh.hold_median, bins=np.arange(8, 16, 0.5), color=C_DEMO, alpha=0.8, label="시연 50개: 쥔 동안 개도(중앙값)")
ax[1].axvline(0.7, color="#555555", lw=2); ax[1].text(1.0, ax[1].get_ylim()[1] * 0.45, "A 빈손 닫힘\n0.7%", fontsize=9, color="#555555")
ax[1].axvline(4, color=C_BAD, ls=":", lw=1.5); ax[1].text(4.2, ax[1].get_ylim()[1] * 0.75, "A 기준 4%", color=C_BAD, fontsize=9)
ax[1].set_xlim(0, 16); ax[1].set_xlabel("그리퍼 개도 [%]"); ax[1].set_ylabel("시연 수"); ax[1].set_title("쥔 상태와 빈손 상태의 개도 차이", fontsize=12); ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0))
data = [gh.duration_s, tr[(tr.system == "B") & (tr.success == 1)].elapsed_s, tr[(tr.system == "A") & (tr.success == 1)].elapsed_s]
bp = ax[2].boxplot(data, tick_labels=["사람 시연", "시스템 B\n(ACT)", "시스템 A\n(기구학)"], patch_artist=True, widths=0.5)
for p, c in zip(bp["boxes"], [C_DEMO, C_B, C_A]): p.set_facecolor(c); p.set_alpha(0.6)
for i, d in enumerate(data): ax[2].text(i + 1, np.max(d) + 1.2, f"중앙값 {np.median(d):.1f}s", ha="center", va="bottom", fontsize=9.5)
ax[2].set_ylabel("한 번 옮기는 데 걸린 시간 [s]"); ax[2].set_ylim(0, 41); ax[2].set_title("동작 시간 (성공 시행)", fontsize=12)
fig.tight_layout(); fig.savefig(F / "fig5_gripper_speed.png", dpi=250); plt.close(fig)

# ---------- 분석 5: 통계 ----------
s = pd.read_csv(O / "ab_stats.csv"); conds = ["base", "new_pos", "color_change", "light_change"]
fig, ax = plt.subplots(figsize=(10, 4.8)); x = np.arange(4)
for k, (sy, col, off) in enumerate([("A", C_A, -0.18), ("B", C_B, 0.18)]):
    r = s.set_index("condition").loc[conds]
    ax.bar(x + off, r[f"{sy}_rate"] * 100, 0.34, color=col, alpha=0.85, label=f"시스템 {sy}")
    ax.errorbar(x + off, r[f"{sy}_rate"] * 100, yerr=[(r[f"{sy}_rate"] - r[f"{sy}_lo"]) * 100, (r[f"{sy}_hi"] - r[f"{sy}_rate"]) * 100], fmt="none", ecolor="black", capsize=4, lw=1)
for i, c in enumerate(conds):
    p = s.set_index("condition").loc[c, "p_fisher"]; txt = "p < 0.001 (유의)" if p < 0.001 else f"p = {p:.2f}"
    ax.text(i, 112, txt, ha="center", fontsize=10, color=C_BAD if p < 0.05 else "#555555")
ax.set_xticks(x, [CL[c] for c in conds]); ax.set_ylim(0, 120); ax.set_ylabel("성공률 [%] (막대: 95% 신뢰구간)")
ax.set_title("A vs B 성공률과 통계 검정 (조건당 20회, Fisher 정확 검정)", fontsize=13); ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
fig.tight_layout(); fig.savefig(F / "fig6_ab_significance.png", dpi=250); plt.close(fig)

# ---------- 분석 6: 기준값 재조정 ----------
rt = pd.read_csv(O / "retune.csv")
sys.path.insert(0, str(REPO)); from so101.detect_ablation import DETECTOR_ORDER, DETECTOR_LABELS
fig, axs = plt.subplots(1, 4, figsize=(16, 4.8), sharey=True)
for a, c in zip(axs, conds):
    y = np.arange(len(DETECTOR_ORDER))
    old = [rt[(rt.detector == d) & (rt.setting == "기존")][c].iloc[0] for d in DETECTOR_ORDER]
    new = [rt[(rt.detector == d) & (rt.setting == "재조정")][c].iloc[0] for d in DETECTOR_ORDER]
    a.barh(y - 0.2, np.array(old) * 5, 0.38, color="#BBBBBB", label="기존 기준값"); a.barh(y + 0.2, np.array(new) * 5, 0.38, color=C_DEMO, label="재조정 (기본 조건만 사용)")
    for i, (o, n) in enumerate(zip(old, new)):
        if n != o: a.text(n * 5 + 2, i + 0.2, f"{o*5:.0f}→{n*5:.0f}%", va="center", fontsize=8.5, color=C_DEMO)
    a.set_title(CL[c], fontsize=12); a.set_xlim(0, 125); a.set_xlabel("정검출률 [%]")
axs[0].set_yticks(np.arange(len(DETECTOR_ORDER)), [DETECTOR_LABELS[d] for d in DETECTOR_ORDER]); axs[0].invert_yaxis()
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=10)
fig.suptitle("기본 조건 사진 20장만으로 기준값을 다시 고른 뒤 다른 조건에 그대로 적용", fontsize=14); fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(F / "fig7_threshold_retune.png", dpi=250); plt.close(fig)

# ---------- 분석 7: A 실패 위치·순서 ----------
t = tr[tr.system == "A"].sort_values("timestamp")
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.5]})
ax[0].imshow((bg * 0.55 + 255 * 0.45).astype(np.uint8)); ax[0].set_xlim(130, 500); ax[0].set_ylim(380, 85); ax[0].set_xticks([]); ax[0].set_yticks([])
ok = A[A.success == 1]; gf = A[A.failure_cause == "grasp_fail"]
ax[0].scatter(ok.u_px, ok.v_px, s=40, color=C_DEMO, edgecolor="white", lw=0.5, label=f"성공 ({len(ok)})", zorder=3)
ax[0].scatter(gf.u_px, gf.v_px, s=70, marker="X", color=C_BAD, edgecolor="white", lw=0.5, label=f"파지 실패 ({len(gf)})", zorder=4)
ax[0].legend(loc="lower left", fontsize=9); ax[0].set_title("A 시행 위치 (검출된 53회)", fontsize=12)
mark = {"": ("o", C_DEMO), "grasp_fail": ("X", C_BAD), "detection_fail": ("s", "#999999")}
for i, r in enumerate(t[t.condition != "color_change"].itertuples()):
    m, c = mark.get(r.failure_cause if isinstance(r.failure_cause, str) else "", ("o", C_DEMO))
    ax[1].scatter(i, ["base", "new_pos", "light_change"].index(r.condition), marker=m, color=c, s=60, edgecolor="white", lw=0.5)
ax[1].set_yticks([0, 1, 2], ["기본", "새 위치", "조명 변경"]); ax[1].invert_yaxis(); ax[1].set_xlabel("실험 순서 (시행 번호, 색 변경 제외)")
for m, c, l in [("o", C_DEMO, "성공"), ("X", C_BAD, "파지 실패"), ("s", "#999999", "검출 실패")]: ax[1].scatter([], [], marker=m, color=c, label=l)
ax[1].legend(loc="upper right", fontsize=9, ncol=3); ax[1].set_title("시간 순서로 본 A 결과: 파지 실패가 연달아 나오는 경향", fontsize=12)
fig.tight_layout(); fig.savefig(F / "fig8_A_failure_map.png", dpi=250); plt.close(fig)

# ---------- 분석 8: 가상 분석 ----------
w = pd.read_csv(O / "whatif.csv"); sc = [("기존 HSV (실측)", "#8E9AAF"), ("HSV 기준값 재조정", "#5B9BD5"), ("정규화 RGB 배경 차분", C_DEMO), ("B (실측)", C_B)]
fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(4)
for k, (name, col) in enumerate(sc):
    r = w[w.scenario == name].set_index("condition").loc[conds]; off = (k - 1.5) * 0.2
    hatch = None if "실측" in name else "//"
    ax.bar(x + off, r.est * 100, 0.19, color=col, alpha=0.85, hatch=hatch, edgecolor="white", label=("시스템 A + " + name) if name.startswith(("HSV", "정규")) else ("시스템 A " + name.replace("기존 HSV ", "") if name.startswith("기존") else "시스템 B (실측)"))
    if r.lo.notna().any(): ax.errorbar(x + off, r.est * 100, yerr=[((r.est - r.lo) * 100).clip(lower=0), ((r.hi - r.est) * 100).clip(lower=0)], fmt="none", ecolor="black", capsize=3, lw=0.8)
ax.set_xticks(x, [CL[c] for c in conds]); ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("성공률 [%]")
ax.set_title("가상 분석: A의 검출기만 바꿨다면 (빗금 = 추정치, 검출률 × 검출 후 성공률)", fontsize=13); ax.legend(fontsize=9, loc="upper center", ncol=4, frameon=False)
fig.tight_layout(); fig.savefig(F / "fig9_whatif.png", dpi=250); plt.close(fig)
print("ok")
