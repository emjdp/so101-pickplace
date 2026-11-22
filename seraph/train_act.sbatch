#!/usr/bin/bash
#SBATCH --job-name=so101-act
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=batch_ugrad
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/%u/logs/act-%j.out

# 경희대학교 Seraph (Aurora 클러스터) ACT 정책 학습 스크립트.
#
# 사용 절차:
#   1. 노트북에서 데모 수집 후 압축:
#        tools/record_act.sh --pack
#      (결과: data/so101_pickplace_dataset.tar.gz)
#   2. SFTP로 Seraph의 /data/$USER/datasets/so101_pickplace_dataset.tar.gz 에 업로드
#   3. Seraph 마스터 노드에서:
#        mkdir -p /data/$USER/logs
#        export WANDB_API_KEY=<본인_키>   # 또는 WANDB_MODE=offline
#        sbatch train_act.sbatch
#   4. 학습 완료 후 /data/$USER/outputs/act-so101-pickplace/checkpoints/last/pretrained_model
#      폴더를 노트북의 ~/dev/so101-pickplace/outputs/act_pickplace/pretrained_model 로 내려받아
#      python scripts/run_system_b.py 로 실행한다.

set -euo pipefail

USER_NAME=${USER:-seongho06}
DATASET_NAME=${DATASET_NAME:-so101_pickplace}
REPO_ID=${REPO_ID:-${USER_NAME}/${DATASET_NAME}}
ARCHIVE_PATH=${ARCHIVE_PATH:-/data/${USER_NAME}/datasets/${DATASET_NAME}_dataset.tar.gz}
LOCAL_JOB_ROOT=/local_datasets/${USER_NAME}/act-${SLURM_JOB_ID}
DATASET_ROOT=${LOCAL_JOB_ROOT}/${DATASET_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-/data/${USER_NAME}/outputs/act-so101-pickplace}
RESUME_CONFIG=${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json
CONDA_ROOT=/data/${USER_NAME}/anaconda3
WANDB_MODE=${WANDB_MODE:-online}
STEPS=${STEPS:-100000}

cleanup_local_data() {
    rm -rf -- "${LOCAL_JOB_ROOT:?}"
}
trap cleanup_local_data EXIT

if [[ ! -r "${ARCHIVE_PATH}" ]]; then
    echo "Dataset archive is missing or unreadable: ${ARCHIVE_PATH}" >&2
    exit 1
fi

if [[ "${WANDB_MODE}" == "online" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "WANDB_API_KEY is not set (or set WANDB_MODE=offline)." >&2
        exit 1
    fi
fi

if [[ -d "${OUTPUT_DIR}" && ! -r "${RESUME_CONFIG}" ]]; then
    echo "Output exists but no resumable checkpoint was found: ${OUTPUT_DIR}" >&2
    exit 1
fi

mkdir -p \
    "${LOCAL_JOB_ROOT}" \
    "/data/${USER_NAME}/outputs" \
    "/data/${USER_NAME}/logs" \
    "/data/${USER_NAME}/cache/torch" \
    "/data/${USER_NAME}/cache/huggingface"

export TORCH_HOME=/data/${USER_NAME}/cache/torch
export HF_HOME=/data/${USER_NAME}/cache/huggingface
export TMPDIR=${LOCAL_JOB_ROOT}/tmp
mkdir -p "${TMPDIR}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate lerobot

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "dataset_archive=${ARCHIVE_PATH}"
echo "dataset_root=${DATASET_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# NAS 과부하 방지를 위해 압축 파일 하나만 노드 로컬 디스크(/local_datasets)로 복사해 해제한다.
cp "${ARCHIVE_PATH}" "${LOCAL_JOB_ROOT}/"
tar -xzf "${LOCAL_JOB_ROOT}/$(basename "${ARCHIVE_PATH}")" -C "${LOCAL_JOB_ROOT}"

if [[ ! -r "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Extracted dataset is invalid: ${DATASET_ROOT}" >&2
    exit 1
fi

if [[ -r "${RESUME_CONFIG}" ]]; then
    echo "Resuming from: ${RESUME_CONFIG}"
    lerobot-train \
        --config_path="${RESUME_CONFIG}" \
        --resume=true \
        --dataset.root="${DATASET_ROOT}" \
        --output_dir="${OUTPUT_DIR}" \
        --steps="${STEPS}" \
        --wandb.mode="${WANDB_MODE}"
else
    lerobot-train \
        --policy.type=act \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --dataset.repo_id="${REPO_ID}" \
        --dataset.root="${DATASET_ROOT}" \
        --dataset.video_backend=pyav \
        --output_dir="${OUTPUT_DIR}" \
        --job_name=so101_pickplace_act \
        --steps="${STEPS}" \
        --batch_size=8 \
        --num_workers=4 \
        --log_freq=100 \
        --save_checkpoint=true \
        --save_freq=10000 \
        --wandb.enable=true \
        --wandb.project=so101-pickplace-act \
        --wandb.mode="${WANDB_MODE}" \
        --wandb.disable_artifact=true
fi

echo "Training completed: ${OUTPUT_DIR}"
