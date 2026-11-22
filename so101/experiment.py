"""비교 실험(계획서 3.4절) 기록 및 정량 집계 모듈.

계획서 3.4절 비교 실험 조건 (조건마다 시스템별로 20회씩, 총 160회):
- 기본 (`base`): ACT 데모 수집 때 사용한 위치 범위
- 새 위치 (`new_pos`): 데모에 없던 위치
- 색 변경 (`color_change`): 다른 색 블록, 설정 변경 없음
- 조명 변경 (`light_change`): 실내 조명 밝기 변경

측정 항목:
- 성공률 (성공 횟수 / 시행 횟수, %)
- 1회 소요 시간 (성공 시행의 평균·중앙값·범위, 실패 시행 경과 시간)
- 준비 시간 (시스템 A: 보정 시간, 시스템 B: 데모 수집 + 학습 시간)
- 실패 원인 분류 (`detection_fail`, `coord_error`, `grasp_fail`, `out_of_range`, `drop_fail`, `policy_drift`, `timeout`)
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

SYSTEMS = ("A", "B")
CONDITIONS: dict[str, str] = {
    "base": "기본 (데모 위치 범위)",
    "new_pos": "새 위치 (데모 밖 위치)",
    "color_change": "색 변경 (다른 색 블록)",
    "light_change": "조명 변경 (조명 밝기 변경)",
}
FAILURE_CAUSES: dict[str, str] = {
    "": "성공",
    "detection_fail": "검출 실패",
    "coord_error": "위치/좌표 오차",
    "grasp_fail": "파지 실패",
    "out_of_range": "작업 범위 밖 (IK 실패 포함)",
    "drop_fail": "이송 중 낙하",
    "policy_drift": "정책 이탈/정지",
    "timeout": "제한 시간 초과",
}


@dataclass
class TrialRecord:
    """단일 시행 기록."""

    timestamp: str
    system: str  # "A" | "B"
    condition: str  # "base" | "new_pos" | "color_change" | "light_change"
    trial: int  # 1..20
    success: int  # 1 | 0
    elapsed_s: float
    failure_cause: str = ""
    block_color: str = "blue"
    u_px: float | str = ""
    v_px: float | str = ""
    target_x_m: float | str = ""
    target_y_m: float | str = ""
    note: str = ""


FIELDNAMES = [
    "timestamp",
    "system",
    "condition",
    "trial",
    "success",
    "elapsed_s",
    "failure_cause",
    "block_color",
    "u_px",
    "v_px",
    "target_x_m",
    "target_y_m",
    "note",
]


def load_trials(csv_path: str | Path) -> list[TrialRecord]:
    path = Path(csv_path)
    if not path.exists():
        return []
    out: list[TrialRecord] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("system"):
                continue
            out.append(
                TrialRecord(
                    timestamp=row.get("timestamp", ""),
                    system=row["system"].strip().upper(),
                    condition=row["condition"].strip(),
                    trial=int(row.get("trial") or 0),
                    success=int(row.get("success") or 0),
                    elapsed_s=float(row.get("elapsed_s") or 0.0),
                    failure_cause=row.get("failure_cause", "").strip(),
                    block_color=row.get("block_color", "blue").strip(),
                    u_px=row.get("u_px", ""),
                    v_px=row.get("v_px", ""),
                    target_x_m=row.get("target_x_m", ""),
                    target_y_m=row.get("target_y_m", ""),
                    note=row.get("note", ""),
                )
            )
    return out


def append_trial(csv_path: str | Path, record: TrialRecord) -> TrialRecord:
    """CSV 파일에 시행 1건을 추가한다. trial 번호가 0이면 해당 (system, condition)의 다음 번호를 자동 부여한다."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_trials(path)
    if record.trial <= 0:
        same = [r.trial for r in existing if r.system == record.system and r.condition == record.condition]
        record.trial = (max(same) + 1) if same else 1
    if not record.timestamp:
        record.timestamp = datetime.now().isoformat(timespec="seconds")
    if record.success:
        record.failure_cause = ""
    elif not record.failure_cause:
        record.failure_cause = "grasp_fail"

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(asdict(record))
    return record


def summarize_trials(records: list[TrialRecord]) -> list[dict]:
    """(system, condition) 조합별 성공률, 소요 시간 통계, 실패 원인 분포를 집계한다."""
    summary = []
    for cond in CONDITIONS:
        for sys_name in SYSTEMS:
            sub = [r for r in records if r.system == sys_name and r.condition == cond]
            if not sub:
                continue
            n = len(sub)
            succ = [r for r in sub if r.success == 1]
            fail = [r for r in sub if r.success == 0]
            n_succ = len(succ)
            rate_pct = 100.0 * n_succ / n

            if succ:
                times = np.array([r.elapsed_s for r in succ], dtype=float)
                mean_s = float(np.mean(times))
                median_s = float(np.median(times))
                min_s = float(np.min(times))
                max_s = float(np.max(times))
            else:
                mean_s = median_s = min_s = max_s = 0.0

            causes: dict[str, int] = {}
            for r in fail:
                k = r.failure_cause or "unknown"
                causes[k] = causes.get(k, 0) + 1

            summary.append(
                {
                    "system": sys_name,
                    "condition": cond,
                    "condition_label": CONDITIONS.get(cond, cond),
                    "n_trials": n,
                    "n_success": n_succ,
                    "success_rate_pct": round(rate_pct, 1),
                    "mean_s": round(mean_s, 2),
                    "median_s": round(median_s, 2),
                    "min_s": round(min_s, 2),
                    "max_s": round(max_s, 2),
                    "failures": causes,
                }
            )
    return summary


def format_markdown_report(records: list[TrialRecord]) -> str:
    """실험 기록을 최종 보고서에 바로 붙여 넣을 수 있는 Markdown 표로 만든다."""
    rows = summarize_trials(records)
    if not rows:
        return "기록된 실험 시행이 없습니다.\n"

    lines = [
        "# 시스템 A(기구학) vs 시스템 B(ACT) 비교 실험 결과",
        "",
        "## 1. 조건별 성공률 및 소요 시간",
        "",
        "| 조건 | 시스템 | 성공 / 시행 | 성공률 | 성공 소요 시간 중앙값 [s] | 평균 [s] | 범위 [s] | 주요 실패 원인 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["n_success"] > 0:
            t_med = f"{r['median_s']:.1f}"
            t_mean = f"{r['mean_s']:.1f}"
            t_rng = f"{r['min_s']:.1f}~{r['max_s']:.1f}"
        else:
            t_med = t_mean = t_rng = "-"
        if r["failures"]:
            f_str = ", ".join(
                f"{FAILURE_CAUSES.get(k, k)} {v}건"
                for k, v in sorted(r["failures"].items(), key=lambda x: -x[1])
            )
        else:
            f_str = "-"
        lines.append(
            f"| {r['condition_label']} | 시스템 {r['system']} | {r['n_success']} / {r['n_trials']} "
            f"| {r['success_rate_pct']:.1f}% | {t_med} | {t_mean} | {t_rng} | {f_str} |"
        )

    lines.extend(
        [
            "",
            "## 2. 실패 원인별 집계",
            "",
            "| 시스템 | 조건 | 검출 실패 | 위치/좌표 오차 | 파지 실패 | 작업 범위 밖 | 이송 중 낙하 | 정책 이탈 | 시간 초과 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    cause_keys = ["detection_fail", "coord_error", "grasp_fail", "out_of_range", "drop_fail", "policy_drift", "timeout"]
    for r in rows:
        f = r["failures"]
        counts = " | ".join(str(f.get(k, 0)) for k in cause_keys)
        lines.append(f"| 시스템 {r['system']} | {r['condition_label']} | {counts} |")

    return "\n".join(lines) + "\n"
