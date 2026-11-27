"""검출기 비교(Ablation) 정량 평가 + Markdown 보고서 표 + 300 DPI 한글 그래프 생성기.

실행:
  python scripts/eval_detectors.py
  python scripts/eval_detectors.py --data-dir data/detector_ablation --tp-radius-px 20

생성 결과:
  - data/detector_ablation/results.csv                  (프레임별·검출기별 세부 기록)
  - data/detector_ablation/summary.md                   (보고서 바로 삽입용 Markdown 표)
  - data/report_figures/fig5_detector_ablation.png      (300 DPI 조건별 정검출률 비교 그래프)
  - data/report_figures/fig6_detector_masks.png         (300 DPI 조건별 대표 프레임 마스크 비교 그리드)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
    DETECTOR_SHORT_LABELS,
    AblationParams,
    build_booth_roi_mask,
    calibrate_rgb_box_from_base,
    calibrate_rgb_diff_from_bg,
    run_detector,
)
from so101.homography import HomographyCalib  # noqa: E402

COND_ORDER = ("base", "new_pos", "color_change", "light_change")
COND_LABELS = {
    "base": "기본\n(데모 위치)",
    "new_pos": "새 위치\n(작업대 전역)",
    "color_change": "색 변경\n(빨간 블록)",
    "light_change": "조명 변경\n(밝기 변화)",
}
COND_TABLE_LABELS = {
    "base": "기본 (데모 위치)",
    "new_pos": "새 위치",
    "color_change": "색 변경 (다른 색 블록)",
    "light_change": "조명 변경 (밝기 변화)",
}
DET_COLORS = {
    "hsv_orig": "#2B5B84",
    "hsv_low_v": "#5B9BD5",
    "rgb_box": "#8E9AAF",
    "rgb_diff": "#F0AD4E",
    "rgb_norm": "#17A2B8",
    "bg_diff": "#C96A5B",
    "bg_norm_diff": "#2E9E6B",
}


def setup_korean_font() -> None:
    for fp in (
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ):
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
    if Path("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf").exists():
        prop = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.9


def benchmark_detector_ms(
    name: str,
    bgr: np.ndarray,
    params: AblationParams,
    bg_bgr: np.ndarray | None,
    roi_mask: np.ndarray | None,
    repeats: int = 5,
) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        run_detector(name, bgr, params, bg_bgr=bg_bgr, roi_mask=roi_mask)
    return 1000.0 * (time.perf_counter() - t0) / repeats


def plot_ablation_rates(stats: dict, active_conds: list[str], out_path: Path) -> None:
    cats = list(active_conds) + ["overall"]
    xlabels = [COND_LABELS[c] for c in active_conds] + ["전체 합계\n(총 80장)"]

    n_det = len(DETECTOR_ORDER)
    x = np.arange(len(cats))
    total_w = 0.84
    bar_w = total_w / n_det

    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    for i, det_name in enumerate(DETECTOR_ORDER):
        offset = (i - (n_det - 1) / 2.0) * bar_w
        rates = [stats[(det_name, c)]["tp_rate"] for c in cats]
        bars = ax.bar(
            x + offset,
            rates,
            width=bar_w * 0.90,
            label=DETECTOR_LABELS[det_name],
            color=DET_COLORS[det_name],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for bar, r in zip(bars, rates):
            lbl = f"{r:.0f}" if abs(r - round(r)) < 0.05 else f"{r:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                r + 1.5,
                lbl,
                ha="center",
                va="bottom",
                fontsize=6.9,
                fontweight="bold",
                color="#333333",
            )

    ax.axvline(len(active_conds) - 0.5, color="#BBBBBB", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10.5)
    ax.set_ylabel("정검출률 (TP Rate, %)", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", linestyle=":", alpha=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("시스템 A 검출기 비교(Ablation): HSV vs RGB / RGB 차분 / 배경 차분 (부스 내부 공통 ROI)", fontsize=12.8, fontweight="bold", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=8.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mask_grid(
    samples_by_cond: dict[str, dict],
    params: AblationParams,
    bg_bgr: np.ndarray | None,
    roi_mask: np.ndarray | None,
    tp_radius_px: float,
    out_path: Path,
) -> None:
    show_dets = ("hsv_orig", "rgb_box", "rgb_diff", "rgb_norm", "bg_diff", "bg_norm_diff")
    conds = [c for c in COND_ORDER if c in samples_by_cond]
    if not conds:
        return

    nrows = len(conds)
    ncols = 1 + len(show_dets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.25 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["원본 + GT"] + [DETECTOR_SHORT_LABELS[d].replace("\n", " ") for d in show_dets]
    x0, y0, x1, y1 = BOOTH_ROI_RECT

    for r_idx, cond in enumerate(conds):
        item = samples_by_cond[cond]
        bgr = item["img"]
        gt_u, gt_v = item["gt_u"], item["gt_v"]

        rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        cv2.drawMarker(rgb_orig, (int(round(gt_u)), int(round(gt_v))), (255, 60, 220), cv2.MARKER_CROSS, 18, 2)
        ax0 = axes[r_idx, 0]
        ax0.imshow(rgb_orig[y0:y1, x0:x1])
        ax0.set_ylabel(COND_LABELS[cond], fontsize=9.5, fontweight="bold")
        ax0.set_xticks([])
        ax0.set_yticks([])
        if r_idx == 0:
            ax0.set_title(col_titles[0], fontsize=9.4, fontweight="bold", pad=6)

        for c_idx, det_name in enumerate(show_dets, start=1):
            det, mask = run_detector(det_name, bgr, params, bg_bgr=bg_bgr, roi_mask=roi_mask)
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            cv2.drawMarker(vis, (int(round(gt_u)), int(round(gt_v))), (255, 60, 220), cv2.MARKER_CROSS, 16, 2)
            if det is None:
                badge, bcol = "MISS", "#D9534F"
            else:
                dist = float(np.hypot(det.u - gt_u, det.v - gt_v))
                if dist <= tp_radius_px:
                    cv2.drawContours(vis, [det.box.astype(np.int32)], 0, (40, 225, 90), 2)
                    badge, bcol = f"TP ({dist:.1f}px)", "#2E9E6B"
                else:
                    cv2.drawContours(vis, [det.box.astype(np.int32)], 0, (235, 60, 60), 2)
                    badge, bcol = f"FP ({dist:.0f}px)", "#E06C3A"

            ax = axes[r_idx, c_idx]
            ax.imshow(vis[y0:y1, x0:x1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                6,
                20,
                badge,
                fontsize=7.6,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=bcol, edgecolor="none", alpha=0.88),
            )
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=8.8, fontweight="bold", pad=6)

    fig.suptitle("조건별 대표 프레임 검출 마스크 비교 (보라 십자: GT, 초록: 정검출, 빨강: 오검출)", fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_dataset(
    data_dir: Path,
    fig_dir: Path,
    tp_radius_px: float = 20.0,
    rgb_diff_thresh: float | None = None,
    rgb_norm_thresh: float = 0.45,
    bg_diff_thresh: float = 35.0,
    bg_norm_diff_thresh: float = 0.12,
    hsv_low_v_min: int = 15,
) -> Path:
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.csv 없음: {manifest_path}\n먼저 `python scripts/capture_detector_eval.py`로 프레임을 캡처하세요.")

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    if not manifest_rows:
        raise SystemExit(f"manifest.csv에 기록된 프레임이 없습니다: {manifest_path}")

    bg_path = data_dir / "bg_empty.png"
    bg_bgr = cv2.imread(str(bg_path)) if bg_path.exists() else None
    roi_mask = build_booth_roi_mask((config.CAM_HEIGHT, config.CAM_WIDTH), BOOTH_ROI_RECT)

    calib = (
        HomographyCalib.load(config.HOMOGRAPHY_CALIB, intrinsics_path=config.CAMERA_INTRINSICS)
        if config.HOMOGRAPHY_CALIB.exists()
        else None
    )

    hsv_raw = json.loads((config.ROOT / "calib" / "hsv_blue.json").read_text())["ranges"]
    hsv_ranges = tuple((tuple(lo), tuple(hi)) for lo, hi in hsv_raw)

    items: list[dict] = []
    for r in manifest_rows:
        fpath = data_dir / r["filename"]
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"[경고] 이미지 로드 실패, 건너뜀: {fpath}")
            continue
        items.append(
            {
                "filename": r["filename"],
                "condition": r["condition"],
                "trial": int(r["trial"]),
                "gt_u": float(r["gt_u"]),
                "gt_v": float(r["gt_v"]),
                "mean_v": float(r.get("mean_v") or 0.0),
                "img": img,
            }
        )

    # Fair calibration: fit rgb_box ONLY on base frames and rgb_diff ONLY on bg_empty!
    base_samples = [(it["img"], it["gt_u"], it["gt_v"]) for it in items if it["condition"] == "base"]
    rgb_lo, rgb_hi = calibrate_rgb_box_from_base(base_samples, patch_radius=5, margin=15)
    eff_rgb_diff_thresh = (
        rgb_diff_thresh
        if rgb_diff_thresh is not None
        else calibrate_rgb_diff_from_bg(bg_bgr, roi_mask=roi_mask, percentile=99.5, margin=6.0, min_thresh=12.0)
    )

    params = AblationParams(
        hsv_ranges=hsv_ranges,
        hsv_low_v_min=hsv_low_v_min,
        rgb_lo=rgb_lo,
        rgb_hi=rgb_hi,
        rgb_diff_thresh=eff_rgb_diff_thresh,
        rgb_norm_thresh=rgb_norm_thresh,
        bg_diff_thresh=bg_diff_thresh,
        bg_norm_diff_thresh=bg_norm_diff_thresh,
    )

    empty_fp: dict[str, bool] = {}
    if bg_bgr is not None:
        for d in DETECTOR_ORDER:
            det_e, _ = run_detector(d, bg_bgr, params, bg_bgr=bg_bgr, roi_mask=roi_mask)
            empty_fp[d] = det_e is not None

    if items:
        for d in DETECTOR_ORDER:
            run_detector(d, items[0]["img"], params, bg_bgr=bg_bgr, roi_mask=roi_mask)

    detail_rows: list[dict] = []
    for it in items:
        img = it["img"]
        gt_u, gt_v = it["gt_u"], it["gt_v"]
        gt_xy = calib.pixel_to_xy(gt_u, gt_v) if calib is not None else None
        for d in DETECTOR_ORDER:
            ms = benchmark_detector_ms(d, img, params, bg_bgr=bg_bgr, roi_mask=roi_mask, repeats=5)
            det, _ = run_detector(d, img, params, bg_bgr=bg_bgr, roi_mask=roi_mask)
            if det is None:
                outcome = "FN"
                err_px = ""
                err_mm = ""
                area = 0.0
                du, dv = "", ""
            else:
                dist_px = float(np.hypot(det.u - gt_u, det.v - gt_v))
                outcome = "TP" if dist_px <= tp_radius_px else "FP"
                err_px = round(dist_px, 2)
                if gt_xy is not None and calib is not None:
                    px_xy = calib.pixel_to_xy(det.u, det.v)
                    err_mm = round(float(np.hypot(px_xy[0] - gt_xy[0], px_xy[1] - gt_xy[1]) * 1000.0), 2)
                else:
                    err_mm = ""
                area = round(det.area, 1)
                du, dv = round(det.u, 1), round(det.v, 1)

            detail_rows.append(
                {
                    "filename": it["filename"],
                    "condition": it["condition"],
                    "trial": it["trial"],
                    "mean_v": it["mean_v"],
                    "detector": d,
                    "outcome": outcome,
                    "gt_u": gt_u,
                    "gt_v": gt_v,
                    "det_u": du,
                    "det_v": dv,
                    "err_px": err_px,
                    "err_mm": err_mm,
                    "area_px2": area,
                    "time_ms": round(ms, 3),
                }
            )

    results_csv = data_dir / "results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    active_conds = [c for c in COND_ORDER if any(it["condition"] == c for it in items)]
    stats: dict[tuple[str, str], dict] = {}
    for d in DETECTOR_ORDER:
        for c in active_conds + ["overall"]:
            sub = [r for r in detail_rows if r["detector"] == d and (c == "overall" or r["condition"] == c)]
            n = len(sub)
            tp = sum(1 for r in sub if r["outcome"] == "TP")
            fp = sum(1 for r in sub if r["outcome"] == "FP")
            fn = sum(1 for r in sub if r["outcome"] == "FN")
            tp_errs_px = [float(r["err_px"]) for r in sub if r["outcome"] == "TP"]
            tp_errs_mm = [float(r["err_mm"]) for r in sub if r["outcome"] == "TP" and r["err_mm"] != ""]
            times_ms = [float(r["time_ms"]) for r in sub]
            stats[(d, c)] = {
                "n": n,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tp_rate": 100.0 * tp / n if n else 0.0,
                "med_err_px": float(np.median(tp_errs_px)) if tp_errs_px else None,
                "med_err_mm": float(np.median(tp_errs_mm)) if tp_errs_mm else None,
                "mean_ms": float(np.mean(times_ms)) if times_ms else 0.0,
            }

    lines = [
        "# 시스템 A 검출기 비교(Ablation) 실험 결과",
        "",
        f"- 평가 데이터: 총 {len(items)}장 (조건: "
        + ", ".join(f"{c} {sum(1 for it in items if it['condition'] == c)}장" for c in active_conds)
        + ")",
        f"- 공통 조건: 부스 내부 작업대 ROI `{BOOTH_ROI_RECT}` (`x∈[140,485], y∈[95,375]`), 동일 후처리(`5×5 OPEN→CLOSE`, `min_area=150 px²`), 정검출(TP) 허용 반경 `{tp_radius_px:.0f} px`",
        "- 보정 공정성: 모든 파라미터는 `base` 조건 및 빈 작업대(`bg_empty.png`)에서만 결정하고 나머지 조건에 변경 없이 적용",
        f"  - `hsv_orig`: `H∈[98,130], S≥80, V≥30` (`calib/hsv_blue.json` 그대로)",
        f"  - `hsv_low_v`: `H∈[98,130], S≥80, V≥{params.hsv_low_v_min}`",
        f"  - `rgb_box` (`base` 자동 보정): `R∈[{rgb_lo[0]},{rgb_hi[0]}], G∈[{rgb_lo[1]},{rgb_hi[1]}], B∈[{rgb_lo[2]},{rgb_hi[2]}]`",
        f"  - `rgb_diff` (`bg_empty` p99.5+6 자동 보정): `B - max(R, G) ≥ {params.rgb_diff_thresh:.0f}`",
        f"  - `rgb_norm`: `B / (R + G + B) ≥ {params.rgb_norm_thresh:.2f}` (`R+G+B ≥ {params.rgb_norm_min_sum:.0f}`)",
        f"  - `bg_diff`: `||I - I_bg||₂ ≥ {params.bg_diff_thresh:.0f}`",
        f"  - `bg_norm_diff`: `||c(I) - shift - c(I_bg)||₂ ≥ {params.bg_norm_diff_thresh:.2f}` (`c = BGR/(B+G+R)`, `R+G+B ≥ {params.rgb_norm_min_sum:.0f}`)",
        "",
        "## 1. 조건별 정검출률 (TP / 시행)",
        "",
        "| 검출 방식 | " + " | ".join(COND_TABLE_LABELS[c] for c in active_conds) + " | 전체 합계 | 평균 연산 시간 [ms] |",
        "|---|" + "|".join(["---"] * (len(active_conds) + 2)) + "|",
    ]
    for d in DETECTOR_ORDER:
        cells = []
        for c in active_conds + ["overall"]:
            s = stats[(d, c)]
            cells.append(f"{s['tp_rate']:.1f}% ({s['tp']}/{s['n']})")
        ms = stats[(d, "overall")]["mean_ms"]
        lines.append(f"| {DETECTOR_LABELS[d]} | " + " | ".join(cells) + f" | {ms:.2f} |")

    lines.extend(
        [
            "",
            "## 2. 실패 유형(미검출 FN / 오검출 FP) 및 위치 정밀도",
            "",
            "| 검출 방식 | 전체 TP | 미검출 (FN) | 오검출/오위치 (FP) | 빈 배경 오검출 | TP 중심 오차 중앙값 [px / mm] |",
            "|---|---|---|---|---|---|",
        ]
    )
    for d in DETECTOR_ORDER:
        s = stats[(d, "overall")]
        err_str = "-"
        if s["med_err_px"] is not None:
            err_str = (
                f"{s['med_err_px']:.1f} px ({s['med_err_mm']:.1f} mm)"
                if s["med_err_mm"] is not None
                else f"{s['med_err_px']:.1f} px"
            )
        efp = "없음" if not empty_fp.get(d, False) else "발생"
        if bg_bgr is None:
            efp = "배경없음"
        lines.append(f"| {DETECTOR_LABELS[d]} | {s['tp']} / {s['n']} | {s['fn']} | {s['fp']} | {efp} | {err_str} |")

    summary_md = data_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Select representative frame per condition for fig6:
    # - new_pos: frame where rgb_box missed but hsv/rgb_norm succeeded (shows brightness gradient!)
    # - light_change: dim frame where hsv_orig missed but rgb_norm & bg_norm_diff succeeded!
    rep: dict[str, dict] = {}
    for c in active_conds:
        c_items = [it for it in items if it["condition"] == c]
        chosen = c_items[0]
        if c == "new_pos":
            for it in c_items:
                drb, _ = run_detector("rgb_box", it["img"], params, roi_mask=roi_mask)
                drn, _ = run_detector("rgb_norm", it["img"], params, roi_mask=roi_mask)
                if drb is None and drn is not None:
                    chosen = it
                    break
        elif c == "light_change":
            interesting = []
            for it in c_items:
                dh, _ = run_detector("hsv_orig", it["img"], params, roi_mask=roi_mask)
                drn, _ = run_detector("rgb_norm", it["img"], params, roi_mask=roi_mask)
                dbg, _ = run_detector("bg_norm_diff", it["img"], params, bg_bgr=bg_bgr, roi_mask=roi_mask)
                if dh is None and drn is not None and dbg is not None:
                    interesting.append(it)
            if interesting:
                chosen = min(interesting, key=lambda x: x["mean_v"])
            else:
                chosen = min(c_items, key=lambda x: x["mean_v"])
        rep[c] = chosen

    setup_korean_font()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig5_path = fig_dir / "fig5_detector_ablation.png"
    fig6_path = fig_dir / "fig6_detector_masks.png"
    plot_ablation_rates(stats, active_conds, fig5_path)
    plot_mask_grid(rep, params, bg_bgr, roi_mask, tp_radius_px, fig6_path)

    print(summary_md.read_text(encoding="utf-8"))
    print("✅ 생성 완료:")
    print(f"  - {results_csv}")
    print(f"  - {summary_md}")
    print(f"  - {fig5_path}")
    print(f"  - {fig6_path}")
    return summary_md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR / "detector_ablation")
    ap.add_argument("--fig-dir", type=Path, default=config.DATA_DIR / "report_figures")
    ap.add_argument("--tp-radius-px", type=float, default=20.0, help="GT 중심 기준 정검출 허용 반경 [px]")
    ap.add_argument("--rgb-diff-thresh", type=float, default=None, help="RGB 채널 차분 임계값 (미지정 시 bg_empty 자동 보정)")
    ap.add_argument("--rgb-norm-thresh", type=float, default=0.45, help="정규화 RGB 임계값 (B/(R+G+B))")
    ap.add_argument("--bg-diff-thresh", type=float, default=35.0, help="단순 배경 RGB 차분 임계값 (||I-I_bg||_2)")
    ap.add_argument("--bg-norm-diff-thresh", type=float, default=0.12, help="정규화 RGB 배경 차분 임계값")
    ap.add_argument("--hsv-low-v", type=int, default=15, help="HSV V_min 완화 대조군 임계값")
    args = ap.parse_args()

    evaluate_dataset(
        data_dir=args.data_dir,
        fig_dir=args.fig_dir,
        tp_radius_px=args.tp_radius_px,
        rgb_diff_thresh=args.rgb_diff_thresh,
        rgb_norm_thresh=args.rgb_norm_thresh,
        bg_diff_thresh=args.bg_diff_thresh,
        bg_norm_diff_thresh=args.bg_norm_diff_thresh,
        hsv_low_v_min=args.hsv_low_v,
    )


if __name__ == "__main__":
    main()
