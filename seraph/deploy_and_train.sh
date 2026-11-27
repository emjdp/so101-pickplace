#!/usr/bin/env bash
# Seraph(Aurora) 데이터셋 업로드 + ACT 학습 제출 + 보고서용 터미널 캡처 자동화.
# - SERAPH_GPU_GUIDE.md 규정 준수: master에서 python/압축해제/재귀탐색 금지, Slurm만 사용.
# - SSH ControlMaster를 써서 비밀번호는 딱 1번만 입력한다.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${SERAPH_HOST:-aurora.khu.ac.kr}"
PORT="${SERAPH_PORT:-30080}"
USER_NAME="${SERAPH_USER:-seongho06}"
ARCHIVE="$ROOT_DIR/data/so101_pickplace_dataset.tar.gz"
SBATCH_LOCAL="$ROOT_DIR/seraph/train_act.sbatch"
SHOT_DIR="$ROOT_DIR/data/report_screenshots"
mkdir -p "$SHOT_DIR"

if [ ! -f "$ARCHIVE" ]; then
    echo "❌ 압축 데이터셋이 없습니다: $ARCHIVE" >&2
    exit 1
fi

CTL_SOCK="/tmp/seraph-cm-${USER_NAME}-$$"
SSH_OPTS=(-p "$PORT" -o ControlPath="$CTL_SOCK" -o ControlMaster=auto -o ControlPersist=10m -o ServerAliveInterval=30)
SCP_OPTS=(-P "$PORT" -o ControlPath="$CTL_SOCK" -o ControlMaster=auto)

cleanup() {
    ssh -p "$PORT" -o ControlPath="$CTL_SOCK" -O exit "${USER_NAME}@${HOST}" >/dev/null 2>&1 || true
    rm -f "$CTL_SOCK"
}
trap cleanup EXIT

render_shot() {
    local title="$1" txt="$2" out="$3"
    "$HOME/miniconda3/envs/lerobot/bin/python" "$ROOT_DIR/tools/render_terminal_shot.py" \
        --title "$title" --in-file "$txt" --out "$out" || true
}

echo "================================================================"
echo "  KHU Seraph (Aurora) ACT 학습 배포 + 보고서 캡처"
echo "  접속: ${USER_NAME}@${HOST}:${PORT}"
echo "  데이터셋: $(basename "$ARCHIVE") ($(du -h "$ARCHIVE" | awk '{print $1}'))"
echo "================================================================"
echo
read -r -s -p "W&B API Key (없거나 오프라인 학습이면 그냥 Enter): " WANDB_KEY
echo
if [ -n "$WANDB_KEY" ]; then
    WANDB_MODE="online"
    echo "-> W&B online 모드로 제출합니다."
else
    WANDB_MODE="offline"
    echo "-> W&B offline 모드로 제출합니다."
fi
echo

echo "🔐 Seraph 비밀번호를 1번만 입력해 주세요..."
ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "echo 'Connected to '$(hostname)"

# ── 1) 접속 & GPU 자원 확인 (캡처 1) ──
TMP1=$(mktemp)
{
    echo "$ ssh -p ${PORT} ${USER_NAME}@${HOST}"
    echo "seongho06@aurora-master:~$ hostname && show-qos | head -n 12"
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "hostname && (show-qos 2>/dev/null | head -n 12 || true)"
    echo
    echo "seongho06@aurora-master:~$ slurm-gres-viz -i"
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "slurm-gres-viz -i 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' || sinfo -p batch_ugrad"
} | tee "$TMP1"
render_shot "Seraph Aurora - SSH Login & GPU Status (slurm-gres-viz)" "$TMP1" "$SHOT_DIR/01_seraph_gpu_status.png"
rm -f "$TMP1"

# ── 2) 디렉터리 준비 + 데이터셋/배치 업로드 + sbatch 제출 (캡처 2) ──
TMP2=$(mktemp)
{
    echo "$ ssh -p ${PORT} ${USER_NAME}@${HOST} 'mkdir -p /data/${USER_NAME}/{datasets,logs,seraph}'"
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "mkdir -p /data/${USER_NAME}/datasets /data/${USER_NAME}/logs /data/${USER_NAME}/seraph"

    echo "$ scp -P ${PORT} data/so101_pickplace_dataset.tar.gz ${USER_NAME}@${HOST}:/data/${USER_NAME}/datasets/"
    scp "${SCP_OPTS[@]}" "$ARCHIVE" "${USER_NAME}@${HOST}:/data/${USER_NAME}/datasets/so101_pickplace_dataset.tar.gz"
    echo "Uploaded so101_pickplace_dataset.tar.gz ($(du -h "$ARCHIVE" | awk '{print $1}'))"

    echo "$ scp -P ${PORT} seraph/train_act.sbatch ${USER_NAME}@${HOST}:/data/${USER_NAME}/seraph/train_act_pickplace.sbatch"
    scp "${SCP_OPTS[@]}" "$SBATCH_LOCAL" "${USER_NAME}@${HOST}:/data/${USER_NAME}/seraph/train_act_pickplace.sbatch"

    echo
    if [ "$WANDB_MODE" = "online" ]; then
        echo "seongho06@aurora-master:~$ REPO_ID=emjdp/so101_pickplace WANDB_MODE=online sbatch /data/${USER_NAME}/seraph/train_act_pickplace.sbatch"
        SUBMIT_OUT=$(ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" \
            "REPO_ID=emjdp/so101_pickplace WANDB_MODE=online WANDB_API_KEY='${WANDB_KEY}' sbatch /data/${USER_NAME}/seraph/train_act_pickplace.sbatch")
    else
        echo "seongho06@aurora-master:~$ REPO_ID=emjdp/so101_pickplace WANDB_MODE=offline sbatch /data/${USER_NAME}/seraph/train_act_pickplace.sbatch"
        SUBMIT_OUT=$(ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" \
            "REPO_ID=emjdp/so101_pickplace WANDB_MODE=offline sbatch /data/${USER_NAME}/seraph/train_act_pickplace.sbatch")
    fi
    echo "$SUBMIT_OUT"
    JOB_ID=$(echo "$SUBMIT_OUT" | grep -o '[0-9]\+' | tail -n 1 || true)

    sleep 3
    echo
    echo "seongho06@aurora-master:~$ squeue -u ${USER_NAME}"
    ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "squeue -u ${USER_NAME}"
} | tee "$TMP2"
render_shot "Seraph Aurora - Dataset Upload & Slurm sbatch Submit" "$TMP2" "$SHOT_DIR/02_seraph_upload_sbatch.png"
JOB_ID=$(grep -o 'Submitted batch job [0-9]\+' "$TMP2" | awk '{print $4}' | tail -n 1 || true)
rm -f "$TMP2"

# ── 3) 할당 및 초기 학습 로그 대기 후 캡처 (최대 60초 대기) ──
if [ -n "${JOB_ID:-}" ]; then
    echo
    echo "⏳ Job ${JOB_ID} 시작 로그 대기 중 (최대 60초)..."
    LOG_PATH="/data/${USER_NAME}/logs/act-${JOB_ID}.out"
    for _ in $(seq 1 12); do
        if ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "test -s '${LOG_PATH}'" 2>/dev/null; then
            sleep 5
            break
        fi
        sleep 5
    done

    TMP3=$(mktemp)
    {
        echo "seongho06@aurora-master:~$ squeue -u ${USER_NAME}"
        ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "squeue -u ${USER_NAME}"
        echo
        echo "seongho06@aurora-master:~$ head -n 35 ${LOG_PATH}"
        ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "head -n 35 '${LOG_PATH}' 2>/dev/null || echo '(대기열 PD 상태 - 노드 할당 대기 중)'"
    } | tee "$TMP3"
    render_shot "Seraph Aurora - Slurm Job ${JOB_ID} Running Log" "$TMP3" "$SHOT_DIR/03_seraph_job_${JOB_ID}_log.png"
    rm -f "$TMP3"
fi

echo
echo "================================================================"
echo "  ✅ 완료! Job ID: ${JOB_ID:-확인필요}"
echo "  📸 보고서 캡처 저장 위치: $SHOT_DIR"
ls -lh "$SHOT_DIR"
echo "================================================================"
read -r -p "창을 닫으려면 Enter..." _
