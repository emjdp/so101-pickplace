"""비교 실험 기록 및 결과표 집계 스크립트 (계획서 3.4절, 총 160회 실험 관리).

사용법:
  # 1) 현재까지 쌓인 실험 기록(data/experiments/trials.csv)으로 성공률·소요시간·실패원인 표 출력
  python scripts/eval_compare.py summary

  # 2) 결과 보고서 Markdown 파일로 저장 (data/experiments/summary.md)
  python scripts/eval_compare.py summary --out data/experiments/summary.md

  # 3) 수동으로 시행 1건 기록 추가
  python scripts/eval_compare.py add --system A --condition base --success 1 --time 11.4
  python scripts/eval_compare.py add --system B --condition new_pos --success 0 --time 25.0 --cause policy_drift

  # 4) 연속 실험 진행 보조 (대화형으로 타이머 측정 및 결과 연속 기록)
  python scripts/eval_compare.py interactive --system A --condition base --count 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from so101.experiment import (  # noqa: E402
    CONDITIONS,
    FAILURE_CAUSES,
    SYSTEMS,
    TrialRecord,
    append_trial,
    format_markdown_report,
    load_trials,
)


def cmd_summary(args) -> None:
    records = load_trials(args.csv)
    md = format_markdown_report(records)
    print(md)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"보고서 표 저장 완료: {out_path}")


def cmd_add(args) -> None:
    rec = append_trial(
        args.csv,
        TrialRecord(
            timestamp="",
            system=args.system,
            condition=args.condition,
            trial=args.trial,
            success=args.success,
            elapsed_s=args.time,
            failure_cause="" if args.success else args.cause,
            block_color=args.color,
            note=args.note,
        ),
    )
    print(
        f"추가 완료: 시스템 {rec.system} | {CONDITIONS[rec.condition]} | #{rec.trial} | "
        f"{'성공' if rec.success else '실패(' + rec.failure_cause + ')'} | {rec.elapsed_s:.2f} s"
    )


def cmd_interactive(args) -> None:
    print(f"=== 연속 실험 기록 모드: 시스템 {args.system} | 조건: {CONDITIONS[args.condition]} ===")
    print("각 시행마다 Enter를 누르면 타이머가 시작되고, 끝날 때 다시 Enter를 누르면 경과 시간이 측정됩니다.")
    print("중단하려면 'q' + Enter를 누르세요.\n")

    for i in range(args.count):
        existing = [r for r in load_trials(args.csv) if r.system == args.system and r.condition == args.condition]
        next_no = (max(r.trial for r in existing) + 1) if existing else 1
        cmd = input(f"[시행 #{next_no}] 블록을 놓고 Enter로 시작 (q: 종료): ").strip().lower()
        if cmd == "q":
            break
        t0 = time.perf_counter()
        input("  ... 동작 중 ... 완료되면 Enter: ")
        elapsed = round(time.perf_counter() - t0, 2)
        ans = input(f"  소요 시간 {elapsed:.2f} s | 성공했습니까? [Y/n]: ").strip().lower()
        success = 0 if ans in ("n", "no", "0") else 1
        cause = ""
        if not success:
            print("  실패 원인:", ", ".join(f"{k}({v})" for k, v in FAILURE_CAUSES.items() if k))
            default_cause = "coord_error" if args.system == "A" else "policy_drift"
            cause = input(f"  원인 코드 [{default_cause}]: ").strip() or default_cause
        note = input("  메모 (선택): ").strip()
        rec = append_trial(
            args.csv,
            TrialRecord(
                timestamp="",
                system=args.system,
                condition=args.condition,
                trial=next_no,
                success=success,
                elapsed_s=elapsed,
                failure_cause=cause,
                block_color=args.color,
                note=note,
            ),
        )
        print(f"  -> 저장됨 (#{rec.trial}, 성공={rec.success}, {rec.elapsed_s:.2f}s)\n")

    print("\n" + format_markdown_report(load_trials(args.csv)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=config.EXPERIMENT_CSV, help="실험 기록 CSV 파일")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summary", help="실험 결과 집계표 출력")
    p_sum.add_argument("--out", type=Path, help="집계표를 저장할 Markdown 파일 경로")
    p_sum.set_defaults(func=cmd_summary)

    p_add = sub.add_parser("add", help="단일 시행 결과 추가")
    p_add.add_argument("--system", choices=list(SYSTEMS), required=True)
    p_add.add_argument("--condition", choices=list(CONDITIONS), required=True)
    p_add.add_argument("--trial", type=int, default=0, help="0이면 다음 번호 자동 부여")
    p_add.add_argument("--success", type=int, choices=[0, 1], required=True)
    p_add.add_argument("--time", type=float, required=True, help="소요 시간 [s]")
    p_add.add_argument("--cause", choices=[k for k in FAILURE_CAUSES if k], default="coord_error")
    p_add.add_argument("--color", default="blue")
    p_add.add_argument("--note", default="")
    p_add.set_defaults(func=cmd_add)

    p_int = sub.add_parser("interactive", help="타이머 포함 연속 실험 기록")
    p_int.add_argument("--system", choices=list(SYSTEMS), required=True)
    p_int.add_argument("--condition", choices=list(CONDITIONS), default="base")
    p_int.add_argument("--count", type=int, default=20)
    p_int.add_argument("--color", default="blue")
    p_int.set_defaults(func=cmd_interactive)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
