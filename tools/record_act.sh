#!/usr/bin/env bash
# 시스템 B (ACT 모방학습) 시연 데이터 녹화 스크립트.
#
# - calib/lerobot/{follower,leader} 의 tp_follower / tp_leader 캘리브레이션을 그대로 사용한다.
# - 기존 데이터셋($HOME/.cache/huggingface/lerobot/emjdp/so101_pickplace)이 있으면 자동으로 --resume=true 를 붙인다.
# - 손목 카메라(/dev/soarm_cam_wrist)가 연결되어 있으면 top + wrist 두 대로 녹화하고,
#   탑뷰만 있거나 --top-only 를 주면 top 카메라 1대로 녹화한다.
#
# 사용법:
#   tools/record_act.sh                 # 이어서 10개 녹화 (첫 실행이면 새 데이터셋 생성)
#   tools/record_act.sh 15              # 이번 세션에 15개 녹화
#   tools/record_act.sh 10 --top-only   # 탑뷰 카메라 1대만 사용하여 녹화
#   tools/record_act.sh 10 --push       # 마지막 세션 후 Hugging Face Hub에 비공개 업로드
#   tools/record_act.sh --pack          # 수집된 데이터셋을 Seraph 업로드용 tar.gz 로 압축
#
# 녹화 중 키 조작 (터미널 창에 포커스가 있어야 함):
#   → (오른쪽 화살표): 현재 에피소드 조기 종료 및 저장 (상자에 넣자마자 누름)
#   ← (왼쪽 화살표)  : 현재 에피소드 폐기 후 재녹화 (실패 시 반드시 누름)
#   Esc              : 세션 종료 (주의: 다음 에피소드 시작 직후에 누를 것)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

REPO_ID="${ACT_REPO_ID:-emjdp/so101_pickplace}"
DATASET_ROOT="${ACT_DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/$REPO_ID}"
TASK="${ACT_TASK:-Pick up the block and put it in the box}"

if [ "${1:-}" = "--pack" ]; then
    if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
        echo "❌ 압축할 데이터셋이 없습니다: $DATASET_ROOT" >&2
        exit 1
    fi
    HAVE=$(grep -o '"total_episodes": [0-9]*' "$DATASET_ROOT/meta/info.json" | grep -o '[0-9]*')
    ARCHIVE="$ROOT_DIR/data/so101_pickplace_dataset.tar.gz"
    mkdir -p "$ROOT_DIR/data"
    echo "📦 데이터셋 압축 중 (${HAVE} 에피소드) -> $ARCHIVE"
    tar -czf "$ARCHIVE" -C "$(dirname "$DATASET_ROOT")" "$(basename "$DATASET_ROOT")"
    ls -lh "$ARCHIVE"
    echo "✅ 완료. Seraph의 /data/\$USER/datasets/ 로 SFTP 업로드한 뒤 seraph/train_act.sbatch 를 제출하세요."
    exit 0
fi

NUM_EPISODES="10"
PUSH=false
TOP_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --push) PUSH=true ;;
        --top-only) TOP_ONLY=true ;;
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                NUM_EPISODES="$arg"
            else
                echo "❌ 알 수 없는 인자: $arg (사용법: tools/record_act.sh [개수] [--top-only] [--push] | --pack)" >&2
                exit 1
            fi
            ;;
    esac
done

RESUME_ARGS=()
if [ -f "$DATASET_ROOT/meta/info.json" ]; then
    HAVE=$(grep -o '"total_episodes": [0-9]*' "$DATASET_ROOT/meta/info.json" | grep -o '[0-9]*')
    echo "📦 기존 데이터셋 발견: ${HAVE}개 -> ${NUM_EPISODES}개 추가 (총 $((HAVE + NUM_EPISODES))개 예정)"
    RESUME_ARGS=(--resume=true)
else
    echo "📦 새 데이터셋 생성: $DATASET_ROOT"
    echo "   ${NUM_EPISODES}개 녹화 예정"
fi

if [ "$TOP_ONLY" = false ] && [ -e /dev/soarm_cam_wrist ]; then
    echo "📷 카메라 구성: top (/dev/soarm_cam_top) + wrist (/dev/soarm_cam_wrist) [MJPG]"
    CAMERAS='{ wrist: {type: opencv, index_or_path: /dev/soarm_cam_wrist, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: opencv, index_or_path: /dev/soarm_cam_top, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}'
else
    echo "📷 카메라 구성: top 단독 (/dev/soarm_cam_top) [YUYV]"
    CAMERAS='{ top: {type: opencv, index_or_path: /dev/soarm_cam_top, width: 640, height: 480, fps: 30, fourcc: "YUYV"}}'
fi

echo "ℹ  녹화 전 다른 터미널에서 tools/lock_camera_live.sh 를 띄워 노출·화이트밸런스를 고정하세요."
echo "   -> 시작하면 곧바로 첫 에피소드가 녹화됩니다. 리더암에 손을 올려두고 대기하세요."
echo

exec lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/soarm_follower \
    --robot.id=tp_follower \
    --robot.calibration_dir="$ROOT_DIR/calib/lerobot/follower" \
    --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/soarm_leader \
    --teleop.id=tp_leader \
    --teleop.calibration_dir="$ROOT_DIR/calib/lerobot/leader" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.no_stamp=true \
    --dataset.single_task="$TASK" \
    --dataset.fps=30 \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub="$PUSH" \
    --dataset.private=true \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --display_data=true \
    "${RESUME_ARGS[@]}"
