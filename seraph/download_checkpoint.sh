#!/usr/bin/env bash
# Seraph(Aurora)에서 학습 완료된 ACT 체크포인트(pretrained_model) 내려받기 + 완료 로그 캡처 + dry-run 검증.
# SSH ControlMaster로 비밀번호는 딱 1번만 입력한다.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${SERAPH_HOST:-aurora.khu.ac.kr}"
PORT="${SERAPH_PORT:-30080}"
USER_NAME="${SERAPH_USER:-seongho06}"
JOB_ID="${1:-127615}"
REMOTE_OUT="/data/${USER_NAME}/outputs/act-so101-pickplace"
REMOTE_CKPT="${REMOTE_OUT}/checkpoints/last/pretrained_model"
LOCAL_PARENT="$ROOT_DIR/outputs/act_pickplace"
SHOT_DIR="$ROOT_DIR/data/report_screenshots"
mkdir -p "$LOCAL_PARENT" "$SHOT_DIR"

CTL_SOCK="/tmp/seraph-dl-${USER_NAME}-$$"
SSH_OPTS=(-p "$PORT" -o ControlPath="$CTL_SOCK" -o ControlMaster=auto -o ControlPersist=10m -o ServerAliveInterval=30)
SCP_OPTS=(-P "$PORT" -o ControlPath="$CTL_SOCK" -o ControlMaster=auto)

cleanup() {
    ssh -p "$PORT" -o ControlPath="$CTL_SOCK" -O exit "${USER_NAME}@${HOST}" >/dev/null 2>&1 || true
    rm -f "$CTL_SOCK"
}
trap cleanup EXIT

echo "================================================================"
echo "  KHU Seraph ACT 체크포인트 다운로드 (Job ${JOB_ID})"
echo "  원격: ${REMOTE_CKPT}"
echo "  로컬: ${LOCAL_PARENT}/pretrained_model"
echo "================================================================"
echo
echo "🔐 Seraph 비밀번호를 1번만 입력해 주세요..."
ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "echo 'Connected to '$(hostname)"

# 1) 완료 로그 및 체크포인트 목록 확인 + 보고서용 캡처
TMP4=$(mktemp)
{
    echo "seongho06@aurora-master:~$ ls -la ${REMOTE_OUT}/checkpoints/"
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "ls -la '${REMOTE_OUT}/checkpoints/'"
    echo
    echo "seongho06@aurora-master:~$ tail -n 25 /data/${USER_NAME}/logs/act-${JOB_ID}.out"
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "tail -n 25 '/data/${USER_NAME}/logs/act-${JOB_ID}.out'"
} | tee "$TMP4"
"$HOME/miniconda3/envs/lerobot/bin/python" "$ROOT_DIR/tools/render_terminal_shot.py" \
    --title "Seraph Aurora - ACT Training Completed (Job ${JOB_ID})" \
    --in-file "$TMP4" \
    --out "$SHOT_DIR/04_seraph_job_${JOB_ID}_complete.png" || true
rm -f "$TMP4"

# 2) 기존 로컬 pretrained_model 있으면 백업 후 깔끔하게 내려받기
if [ -d "${LOCAL_PARENT}/pretrained_model" ]; then
    mv "${LOCAL_PARENT}/pretrained_model" "${LOCAL_PARENT}/pretrained_model_bak_$(date +%Y%m%d_%H%M%S)"
fi

echo
echo "📥 pretrained_model 내려받는 중..."
scp -r "${SCP_OPTS[@]}" "${USER_NAME}@${HOST}:${REMOTE_CKPT}" "${LOCAL_PARENT}/pretrained_model"
echo
ls -lh "${LOCAL_PARENT}/pretrained_model"

# 3) 로컬 dry-run 추론 검증
echo
echo "🧪 로컬 dry-run 추론 검증 중..."
"$HOME/miniconda3/envs/lerobot/bin/python" "$ROOT_DIR/scripts/run_system_b.py" --dry-run

echo
echo "================================================================"
echo "  ✅ 다운로드 및 dry-run 추론 검증 완료!"
echo "  경로: ${LOCAL_PARENT}/pretrained_model"
echo "================================================================"
read -r -p "창을 닫으려면 Enter..." _
