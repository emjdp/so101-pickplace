#!/usr/bin/env bash
# ACT 텔레옵 연속 세션 도우미 (10개씩 끊어서 50개까지 자동 이어찍기).
# gnome-terminal 등 실제 TTY에서 실행해야 →/←/Esc (또는 n/r/q) 키가 동작한다.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# ROS 2 환경 변수 제거 (실기 안전)
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH

# conda lerobot 활성화
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate lerobot
fi

REPO_ID="${ACT_REPO_ID:-emjdp/so101_pickplace}"
DATASET_ROOT="${ACT_DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/$REPO_ID}"
TARGET="${1:-50}"
EXTRA_ARGS=("${@:2}")

episode_count() {
    if [ -f "$DATASET_ROOT/meta/info.json" ]; then
        grep -o '"total_episodes": [0-9]*' "$DATASET_ROOT/meta/info.json" | grep -o '[0-9]*' || echo 0
    else
        echo 0
    fi
}

while true; do
    HAVE=$(episode_count)
    REM=$((TARGET - HAVE))
    [ "$REM" -lt 0 ] && REM=0
    DEF=10
    [ "$REM" -gt 0 ] && [ "$REM" -lt 10 ] && DEF="$REM"

    echo
    echo "================================================================"
    echo "  ACT 텔레옵 시연 녹화  |  현재 수집: ${HAVE} / ${TARGET} 에피소드"
    echo "  카메라 옵션: ${EXTRA_ARGS[*]:-top + wrist (2카메라)}"
    echo "================================================================"
    echo "  [녹화 중 조작 — 이 터미널 창에 포커스를 둔 상태에서 입력]"
    echo "    →  또는  n : 현재 에피소드 저장 (상자에 넣자마자)"
    echo "    ←  또는  r : 실패 시 폐기 후 재녹화"
    echo "    Esc 또는 q : 이번 배치 조기 종료 (다음 에피소드 시작 직후 권장)"
    echo "----------------------------------------------------------------"
    if [ "$HAVE" -ge "$TARGET" ]; then
        echo "  🎉 목표 ${TARGET}개 달성! 'p' 입력 시 Seraph 업로드용 tar.gz 압축"
    fi
    read -r -p "리더암에 손을 올리고 Enter [${DEF}개 시작 / 숫자 입력 / p:압축 / q:종료]: " ans
    case "$ans" in
        q|Q)
            echo "종료합니다. (현재 총 $(episode_count)개 저장됨)"
            break
            ;;
        p|P)
            tools/record_act.sh --pack
            continue
            ;;
        "")
            N="$DEF"
            ;;
        *[!0-9]*)
            echo "숫자, Enter, p, 또는 q를 입력하세요."
            continue
            ;;
        *)
            N="$ans"
            ;;
    esac

    tools/lock_camera.sh >/dev/null 2>&1 || true
    tools/record_act.sh "$N" "${EXTRA_ARGS[@]}" || echo "⚠ 세션이 중단되었습니다. 위 로그를 확인하세요."
done

read -r -p "창을 닫으려면 Enter..." _
