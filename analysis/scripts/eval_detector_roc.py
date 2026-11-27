"""검출기 7종 픽셀 단위 ROC·PR 분석 (기존 검출기 비교 프레임 80장 재사용, 추가 촬영 없음).

사용법:
  python scripts/eval_detector_roc.py
  python scripts/eval_detector_roc.py --r-pos 6 --r-neg 30 --n-boot 1000

입력 (scripts/capture_detector_eval.py 가 만든 것):
  data/detector_ablation/manifest.csv, frames/*.png, bg_empty.png

출력:
  data/report_figures/fig1_detector_roc.png        조건별 ROC (가로축 FPR 로그 눈금) + 기존 임계값 위치
  data/report_figures/fig2_detector_auc_heatmap.png 검출기 x 조건 AUC
  data/detector_ablation/roc_auc.csv, roc_summary.md

방법 요약:
  - 각 검출기를 같은 판정 규칙의 연속 점수로 바꾼다 (so101/detect_roc.py).
  - 정답 중심에서 r_pos 이내 = 블록 픽셀, r_neg 보다 먼 ROI 픽셀 = 배경 픽셀, 사이는 제외.
  - 파라미터 보정은 scripts/eval_detectors.py 와 똑같이 base 프레임과 bg_empty 로만 한다.
  - 형태학 처리(OPEN/CLOSE)와 최대 윤곽선 선택 전의 "픽셀 분류" 성능만 본다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.detect_ablation import (  # noqa: E402
    BOOTH_ROI_RECT,
    DETECTOR_LABELS,
    DETECTOR_ORDER,
    AblationParams,
    build_booth_roi_mask,
    calibrate_rgb_box_from_base,
    calibrate_rgb_diff_from_bg,
)
from so101.detect_roc import build_edges, image_hist, label_masks, roc_from_hists, score_map  # noqa: E402

COND_ORDER = ["base", "new_pos", "color_change", "light_change"]
COND_LABELS = {
    "all": "전체 (80장)",
    "base": "기본",
    "new_pos": "새 위치",
    "color_change": "색 변경 (빨간 블록)",
    "light_change": "조명 변경",
}
COLORS = {
    "hsv_orig": "#2B5B84",
    "hsv_low_v": "#5B9BD5",
    "rgb_box": "#8E9AAF",
    "rgb_diff": "#F0AD4E",
    "rgb_norm": "#17A2B8",
    "bg_diff": "#C96A5B",
    "bg_norm_diff": "#2E9E6B",
}


def setup_korean_font() -> None:
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in candidates:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"


def load_items(data_dir: Path) -> list[dict]:
    with (data_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items = []
    for r in rows:
        img = cv2.imread(str(data_dir / r["filename"]))
        if img is None:
            print(f"[경고] 이미지 없음, 건너뜀: {data_dir / r['filename']}")
            continue
        items.append({"condition": r["condition"], "gt_u": float(r["gt_u"]), "gt_v": float(r["gt_v"]), "img": img})
    if not items:
        raise SystemExit(
            f"평가할 프레임이 없습니다: {data_dir}/frames\n"
            "프레임 PNG는 .gitignore 로 GitHub에 올라가지 않으니, 촬영한 컴퓨터의 data/detector_ablation/ 폴더를 복사해 오세요."
        )
    return items


def make_params(items: list[dict], bg_bgr, roi_mask, args) -> AblationParams:
    """scripts/eval_detectors.py 와 같은 공정 보정: base 프레임과 빈 배경만 사용."""
    hsv_raw = json.loads((config.ROOT / "calib" / "hsv_blue.json").read_text())["ranges"]
    base = [(it["img"], it["gt_u"], it["gt_v"]) for it in items if it["condition"] == "base"]
    rgb_lo, rgb_hi = calibrate_rgb_box_from_base(base, patch_radius=5, margin=15)
    rgb_diff_t = calibrate_rgb_diff_from_bg(bg_bgr, roi_mask=roi_mask, percentile=99.5, margin=6.0, min_thresh=12.0)
    return AblationParams(
        hsv_ranges=tuple((tuple(lo), tuple(hi)) for lo, hi in hsv_raw),
        hsv_low_v_min=args.hsv_low_v,
        rgb_lo=rgb_lo,
        rgb_hi=rgb_hi,
        rgb_diff_thresh=rgb_diff_t,
        rgb_norm_thresh=args.rgb_norm_thresh,
        bg_diff_thresh=args.bg_diff_thresh,
        bg_norm_diff_thresh=args.bg_norm_diff_thresh,
    )


def evaluate(items, params, bg_bgr, roi_mask, r_pos, r_neg, n_boot):
    """검출기별로 이미지마다 구간 개수를 모은 뒤 조건별 ROC를 계산한다."""
    labels = [label_masks(it["img"].shape, it["gt_u"], it["gt_v"], roi_mask, r_pos, r_neg) for it in items]
    conds = np.array([it["condition"] for it in items])
    results: dict[str, dict[str, object]] = {}
    for name in DETECTOR_ORDER:
        scores, op = [], 0.0
        for it in items:
            s, op = score_map(name, it["img"], params, bg_bgr, roi_mask)
            scores.append(s)
        rng = np.random.default_rng(0)
        sample = np.concatenate(
            [rng.choice(s[p | n], size=min(4000, int((p | n).sum())), replace=False) for s, (p, n) in zip(scores, labels)]
        )
        edges = build_edges(sample, op)
        ph, nh = zip(*[image_hist(s, p, n, edges) for s, (p, n) in zip(scores, labels)])
        ph, nh = np.stack(ph), np.stack(nh)
        results[name] = {"all": roc_from_hists(ph, nh, edges, op, n_boot)}
        for c in COND_ORDER:
            sel = conds == c
            if sel.any():
                results[name][c] = roc_from_hists(ph[sel], nh[sel], edges, op, n_boot)
        print(f"  {DETECTOR_LABELS[name]:<28} AUC(전체) = {results[name]['all'].auc:.3f}")
    return results


def plot_roc(results, out: Path) -> None:
    panels = ["all"] + [c for c in COND_ORDER if c in results[DETECTOR_ORDER[0]]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), panels):
        for name in DETECTOR_ORDER:
            r = results[name][c]
            x = np.clip(r.fpr, 1e-6, 1)
            ax.plot(x, r.tpr, color=COLORS[name], lw=1.6, label=f"{DETECTOR_LABELS[name]} ({r.auc:.3f})")
            ax.plot(max(r.op_fpr, 1.3e-5), r.op_tpr, "o", color=COLORS[name], ms=6, mec="white", mew=1.0)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.plot([1e-5, 1], [1e-5, 1], ls=":", color="#BBBBBB", lw=1)
        ax.set_title(COND_LABELS[c], fontsize=12)
        ax.set_xlabel("FPR (배경을 블록으로 잘못 판정, 로그)")
        ax.grid(True, which="major", ls=":", color="#DDDDDD")
    np.atleast_1d(axes)[0].set_ylabel("TPR (블록 픽셀을 맞게 판정)")
    handles, lbls = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, [l.rsplit(" (", 1)[0] for l in lbls], loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("검출기 7종 픽셀 단위 ROC  ·  점(●) = 현재 사용 중인 임계값", fontsize=14)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_auc_heatmap(results, out: Path) -> None:
    cols = ["all"] + [c for c in COND_ORDER if c in results[DETECTOR_ORDER[0]]]
    M = np.array([[results[n][c].auc for c in cols] for n in DETECTOR_ORDER])
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 0.55 * len(DETECTOR_ORDER) + 1.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols)), [COND_LABELS[c] for c in cols])
    ax.set_yticks(range(len(DETECTOR_ORDER)), [DETECTOR_LABELS[n] for n in DETECTOR_ORDER])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=10, color="black")
    fig.colorbar(im, ax=ax, label="AUC (0.5 = 무작위, 1 = 완벽)")
    ax.set_title("검출기 x 조건 픽셀 단위 AUC", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def write_tables(results, data_dir: Path, args, params: AblationParams) -> None:
    cols = ["all"] + [c for c in COND_ORDER if c in results[DETECTOR_ORDER[0]]]
    with (data_dir / "roc_auc.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["detector", "condition", "auc", "auc_ci_lo", "auc_ci_hi", "ap", "op_tpr", "op_fpr", "op_precision", "n_pos", "n_neg"])
        for n in DETECTOR_ORDER:
            for c in cols:
                r = results[n][c]
                w.writerow([n, c, f"{r.auc:.4f}", f"{r.auc_ci[0]:.4f}", f"{r.auc_ci[1]:.4f}", f"{r.ap:.4f}",
                            f"{r.op_tpr:.4f}", f"{r.op_fpr:.6f}", f"{r.op_precision:.4f}", r.n_pos, r.n_neg])

    lines = [
        "# 검출기 7종 픽셀 단위 ROC 분석",
        "",
        f"- 근사 정답: 블록 중심에서 {args.r_pos:g} px 이내 = 블록, {args.r_neg:g} px 밖 ROI `{BOOTH_ROI_RECT}` = 배경, 사이는 제외",
        f"- 파라미터는 `eval_detectors.py`와 동일 (base 프레임·빈 배경으로만 보정, RGB 채널 차분 임계값 {params.rgb_diff_thresh:.0f})",
        f"- AUC 95% 신뢰구간: 이미지 단위 부트스트랩 {args.n_boot}회",
        "- 형태학 처리와 최대 윤곽선 선택 전의 픽셀 분류 성능이므로, `summary.md`의 정검출률과 직접 같은 값은 아니다.",
        "- FPR이 0인 동작점은 그래프 왼쪽 끝(1e-5)에 표시한다.",
        "- 색 변경 조건의 파란색 전용 검출기 AUC는 '빨간 블록이 배경보다 파란색 기준에 더 가까운가'를 잰 값이라 해석에 주의한다. 실제로 쓰는 임계값에서의 TPR이 더 중요하다.",
        "",
        "## 1. 조건별 AUC",
        "",
        "| 검출 방식 | " + " | ".join(COND_LABELS[c] for c in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for n in DETECTOR_ORDER:
        cells = [f"{results[n][c].auc:.3f} [{results[n][c].auc_ci[0]:.3f}, {results[n][c].auc_ci[1]:.3f}]" for c in cols]
        lines.append(f"| {DETECTOR_LABELS[n]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 2. 현재 임계값의 위치 (전체 80장)",
        "",
        "| 검출 방식 | TPR | FPR | 정밀도 | AP (PR 곡선 넓이) |",
        "|---|---|---|---|---|",
    ]
    for n in DETECTOR_ORDER:
        r = results[n]["all"]
        lines.append(f"| {DETECTOR_LABELS[n]} | {r.op_tpr:.3f} | {r.op_fpr:.5f} | {r.op_precision:.3f} | {r.ap:.3f} |")
    (data_dir / "roc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR / "detector_ablation")
    ap.add_argument("--fig-dir", type=Path, default=config.DATA_DIR / "report_figures")
    ap.add_argument("--r-pos", type=float, default=6.0, help="블록으로 볼 중심 반경 [px] (블록 한 변 약 25 px)")
    ap.add_argument("--r-neg", type=float, default=30.0, help="이 반경 밖을 배경으로 본다 [px]")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--rgb-norm-thresh", type=float, default=0.45)
    ap.add_argument("--bg-diff-thresh", type=float, default=35.0)
    ap.add_argument("--bg-norm-diff-thresh", type=float, default=0.12)
    ap.add_argument("--hsv-low-v", type=int, default=15)
    args = ap.parse_args()

    setup_korean_font()
    items = load_items(args.data_dir)
    bg_bgr = cv2.imread(str(args.data_dir / "bg_empty.png"))
    if bg_bgr is None:
        print("[경고] bg_empty.png 없음: 배경 차분 2종은 점수를 계산할 수 없습니다.")
    roi_mask = build_booth_roi_mask(items[0]["img"].shape, BOOTH_ROI_RECT)
    params = make_params(items, bg_bgr, roi_mask, args)

    print(f"프레임 {len(items)}장으로 ROC 계산 중...")
    results = evaluate(items, params, bg_bgr, roi_mask, args.r_pos, args.r_neg, args.n_boot)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_roc(results, args.fig_dir / "fig1_detector_roc.png")
    plot_auc_heatmap(results, args.fig_dir / "fig2_detector_auc_heatmap.png")
    write_tables(results, args.data_dir, args, params)
    print(f"저장: {args.fig_dir}/fig1_detector_roc.png, fig2_detector_auc_heatmap.png")
    print(f"저장: {args.data_dir}/roc_auc.csv, roc_summary.md")


if __name__ == "__main__":
    main()
