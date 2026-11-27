"""시스템 A vs 시스템 B 비교 실험 및 ACT 학습 보고서용 300 DPI 한글 그래프 생성기.

실행:
  python tools/plot_report_figures.py
출력:
  data/report_figures/fig1_success_rate.png
  data/report_figures/fig2_elapsed_time.png
  data/report_figures/fig3_failure_causes.png
  data/report_figures/fig4_act_training_loss.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.experiment import load_trials  # noqa: E402

COND_ORDER = ["base", "new_pos", "color_change", "light_change"]
COND_LABELS = {
    "base": "기본\n(데모 위치)",
    "new_pos": "새 위치\n(미학습 위치)",
    "color_change": "색 변경\n(다른 색 블록)",
    "light_change": "조명 변경\n(밝기 변화)",
}
COLOR_A = "#2B5B84"  # 시스템 A (기구학)
COLOR_B = "#E06C3A"  # 시스템 B (ACT)


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


def plot_success_rate(records, out_path: Path) -> None:
    cats = COND_ORDER + ["overall"]
    labels = [COND_LABELS[c] for c in COND_ORDER] + ["전체 합계\n(총 80회)"]

    rates_a, rates_b = [], []
    counts_a, counts_b = [], []
    for c in cats:
        sub_a = [r for r in records if r.system == "A" and (c == "overall" or r.condition == c)]
        sub_b = [r for r in records if r.system == "B" and (c == "overall" or r.condition == c)]
        ok_a, n_a = sum(r.success for r in sub_a), len(sub_a)
        ok_b, n_b = sum(r.success for r in sub_b), len(sub_b)
        rates_a.append(100.0 * ok_a / n_a if n_a else 0.0)
        rates_b.append(100.0 * ok_b / n_b if n_b else 0.0)
        counts_a.append((ok_a, n_a))
        counts_b.append((ok_b, n_b))

    x = np.arange(len(cats))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars_a = ax.bar(x - w / 2, rates_a, w, label="시스템 A (기구학 + HSV)", color=COLOR_A, edgecolor="white", linewidth=1.0, zorder=3)
    bars_b = ax.bar(x + w / 2, rates_b, w, label="시스템 B (ACT 모방학습)", color=COLOR_B, edgecolor="white", linewidth=1.0, zorder=3)

    # Subtle separator before Overall
    ax.axvline(3.5, color="#BBBBBB", linestyle="--", linewidth=1.0, zorder=2)

    for bar, (ok, n), rate in zip(bars_a, counts_a, rates_a):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.8,
            f"{rate:.1f}%\n({ok}/{n})",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#1E3F5C",
        )
    for bar, (ok, n), rate in zip(bars_b, counts_b, rates_b):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.8,
            f"{rate:.1f}%\n({ok}/{n})",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#A84319",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("성공률 (%)", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", linestyle=":", alpha=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("조건별 픽앤플레이스 성공률 비교 (시스템 A vs 시스템 B, 총 160회)", fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_elapsed_time(records, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    positions_a = np.arange(len(COND_ORDER)) - 0.19
    positions_b = np.arange(len(COND_ORDER)) + 0.19
    rng = np.random.default_rng(42)

    for idx, cond in enumerate(COND_ORDER):
        ta = [r.elapsed_s for r in records if r.system == "A" and r.condition == cond and r.success == 1]
        tb = [r.elapsed_s for r in records if r.system == "B" and r.condition == cond and r.success == 1]

        for pos, times, col in ((positions_a[idx], ta, COLOR_A), (positions_b[idx], tb, COLOR_B)):
            if times:
                ax.boxplot(
                    [times],
                    positions=[pos],
                    widths=0.28,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=col, alpha=0.28, edgecolor=col, linewidth=1.4),
                    medianprops=dict(color=col, linewidth=2.2),
                    whiskerprops=dict(color=col, linewidth=1.2),
                    capprops=dict(color=col, linewidth=1.2),
                    zorder=3,
                )
                jitter = rng.uniform(-0.07, 0.07, size=len(times))
                ax.scatter(np.full(len(times), pos) + jitter, times, color=col, s=24, alpha=0.78, edgecolors="white", linewidth=0.4, zorder=4)
                med = float(np.median(times))
                ax.text(pos, max(times) + 1.1, f"{med:.1f}s", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=col)
            else:
                ax.text(pos, 1.5, "성공\n없음", ha="center", va="bottom", fontsize=8.5, color="#888888", style="italic")

    # Legend proxies
    ax.scatter([], [], color=COLOR_A, s=55, label="시스템 A (기구학 + HSV) — 중앙값 ~26.8초")
    ax.scatter([], [], color=COLOR_B, s=55, label="시스템 B (ACT 모방학습) — 중앙값 ~5.5초 (약 5배 빠름)")

    ax.set_xticks(np.arange(len(COND_ORDER)))
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER], fontsize=10.5)
    ax.set_ylabel("성공 시행 소요 시간 (초)", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, 42)
    ax.grid(axis="y", linestyle=":", alpha=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("성공 시행 소요 시간 분포 비교 (상자그림 + 개별 시행, 중앙값 표기)", fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=9.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_failure_causes(records, out_path: Path) -> None:
    causes = [
        ("detection_fail", "검출 실패", "#D9534F"),
        ("grasp_fail", "파지 실패", "#F0AD4E"),
        ("policy_drift", "정책 이탈/정지", "#7B68EE"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    x = np.arange(len(COND_ORDER))
    xlabels = [COND_LABELS[c] for c in COND_ORDER]

    for ax, sys_name, title in zip(
        axes,
        ("A", "B"),
        ("시스템 A (기구학 + HSV) — 총 실패 40건", "시스템 B (ACT 모방학습) — 총 실패 21건"),
    ):
        bottom = np.zeros(len(COND_ORDER))
        for key, label, col in causes:
            vals = np.array(
                [
                    sum(1 for r in records if r.system == sys_name and r.condition == c and r.success == 0 and r.failure_cause == key)
                    for c in COND_ORDER
                ],
                dtype=float,
            )
            ax.bar(x, vals, width=0.52, bottom=bottom, label=label, color=col, edgecolor="white", linewidth=1.0, zorder=3)
            for xi, v, b in zip(x, vals, bottom):
                if v >= 2:
                    ax.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=4)
            bottom += vals

        for xi, tot in zip(x, bottom):
            ax.text(xi, tot + 0.45, f"{int(tot)}건", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9.8)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle=":", alpha=0.65, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("실패 건수 (조건당 20회 중)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 23.5)
    axes[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=9.8)
    fig.suptitle("조건별 실패 원인 분석 (시스템 A vs 시스템 B)", fontsize=13.5, fontweight="bold", y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_act_loss(csv_path: Path, out_path: Path) -> None:
    steps, losses = [], []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = row.get("Step", "").strip()
            v = row.get("so101_pickplace_act - train/loss", "").strip()
            if s and v:
                steps.append(float(s))
                losses.append(float(v))
    steps_k = np.array(steps) / 1000.0
    losses = np.array(losses)

    # Exponential moving average for clean trend line
    ema = np.empty_like(losses)
    ema[0] = losses[0]
    alpha = 0.12
    for i in range(1, len(losses)):
        ema[i] = alpha * losses[i] + (1 - alpha) * ema[i - 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.5))

    for ax, is_log, title in (
        (ax1, False, "선형 스케일 (초기 급감 구간)"),
        (ax2, True, "로그 스케일 (후반 미세 수렴 구간)"),
    ):
        ax.plot(steps_k, losses, color=COLOR_B, alpha=0.25, linewidth=1.0, label="원본 (100스텝 로깅)")
        ax.plot(steps_k, ema, color="#C84B1C", linewidth=2.0, label="이동평균 (EMA)")
        if is_log:
            ax.set_yscale("log")
            ax.set_ylim(0.03, 12.0)
            ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
            ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10"])
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
        else:
            ax.set_ylim(0, 9.8)
        ax.set_xlim(0, 102)
        ax.set_xlabel("학습 스텝 (×1,000)", fontsize=10.8, fontweight="bold")
        ax.set_ylabel("학습 손실 (train/loss)", fontsize=10.8, fontweight="bold")
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate start & end points
        ax.annotate(
            f"초기 {losses[0]:.2f}",
            xy=(steps_k[0], losses[0]),
            xytext=(14, losses[0] * (0.65 if is_log else 0.88)),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
            fontsize=9.2,
            fontweight="bold",
            color="#333333",
        )
        ax.annotate(
            f"최종 {losses[-1]:.4f}\n(100k 스텝)",
            xy=(steps_k[-1], losses[-1]),
            xytext=(62, 0.18 if is_log else 1.8),
            arrowprops=dict(arrowstyle="->", color="#C84B1C", lw=1.2),
            fontsize=9.2,
            fontweight="bold",
            color="#A84319",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F0", edgecolor="#E06C3A", alpha=0.9),
        )

    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=9.2)
    fig.suptitle("시스템 B (ACT) 학습 손실 수렴 곡선 (50 에피소드, RTX 3090 · 3시간 32분)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    setup_korean_font()
    out_dir = config.DATA_DIR / "report_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_trials(config.EXPERIMENT_CSV)
    plot_success_rate(records, out_dir / "fig1_success_rate.png")
    plot_elapsed_time(records, out_dir / "fig2_elapsed_time.png")
    plot_failure_causes(records, out_dir / "fig3_failure_causes.png")

    loss_csv = config.DATA_DIR / "report_screenshots" / "wandb_export_loss.csv"
    if loss_csv.exists():
        plot_act_loss(loss_csv, out_dir / "fig4_act_training_loss.png")

    print("✅ 보고서용 300 DPI 그래프 생성 완료:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  - {p.relative_to(config.ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
