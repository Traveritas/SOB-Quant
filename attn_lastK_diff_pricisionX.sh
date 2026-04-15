#!/bin/bash
#SBATCH --job-name=sob_llama_lastk_xcodebook
#SBATCH --account=rrg-yymao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/zxl/scratch/soblogs/sob_llama_lastk_xcodebook_%A_%a.out
#SBATCH --error=/home/zxl/scratch/soblogs/sob_llama_lastk_xcodebook_%A_%a.err
#SBATCH --array=0-41
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=2450827@tongji.edu.cn

set -euo pipefail

echo "=== 作业开始 (SOB Llama2-7B last-k QKVO quant with different X precision) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"

mkdir -p /home/zxl/scratch/logs

module load StdEnv gcc arrow cuda python scipy-stack faiss cudnn
source /home/zxl/env/sobq/bin/activate

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3

export HF_HOME=/home/zxl/scratch/hf_cache
export HF_DATASETS_CACHE=/home/zxl/scratch/hf_cache/datasets
export TRANSFORMERS_CACHE=/home/zxl/scratch/hf_cache

MODEL_PATH="/home/zxl/scratch/hf_cache/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
PY_SCRIPT="${SLURM_SUBMIT_DIR}/basis_diff_pricisionX.py"
OUTPUT_ROOT="/home/zxl/scratch/SOBquant_results/lastk_diff_pricisionX_4.14"

K_VALUES=(8 12 16 20 24 28 31)
X_CODEBOOK_MODES=(int4 int5 int6 int7 int8 none)

NUM_K=${#K_VALUES[@]}
NUM_X_MODE=${#X_CODEBOOK_MODES[@]}

ARRAY_ID=${SLURM_ARRAY_TASK_ID}
K_ID=$((ARRAY_ID / NUM_X_MODE))
X_MODE_ID=$((ARRAY_ID % NUM_X_MODE))

if [ "${K_ID}" -ge "${NUM_K}" ] || [ "${X_MODE_ID}" -ge "${NUM_X_MODE}" ]; then
    echo "Invalid array index: ARRAY_ID=${ARRAY_ID}, K_ID=${K_ID}, X_MODE_ID=${X_MODE_ID}"
    echo "Expected ARRAY_ID in [0, $((NUM_K * NUM_X_MODE - 1))]"
    exit 1
fi

K=${K_VALUES[$K_ID]}
X_CODEBOOK_MODE=${X_CODEBOOK_MODES[$X_MODE_ID]}

BETA=1
GAMMA=1e3
LAMBDA_INNER_ITERS=3
MAX_ITERS=80
CALIB_NUM_TOKENS=4096
STRIDE=512
RUN_MODE=discrete
MODEL_DTYPE=float16
INIT_MODE=random
ERROR_MODE=relative
TARGET_MODE=qkvo
FIT_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"

START_BLOCK=$((32 - K))
BLOCK_INDICES=""
for ((i=START_BLOCK; i<32; i++)); do
    if [ -z "${BLOCK_INDICES}" ]; then
        BLOCK_INDICES="${i}"
    else
        BLOCK_INDICES="${BLOCK_INDICES},${i}"
    fi
done

EXP_NAME="last${K}_qkvo_x${X_CODEBOOK_MODE}"
OUTDIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTDIR}"

echo "=== 当前配置 ==="
echo "ARRAY_ID=${ARRAY_ID}"
echo "EXP_NAME=${EXP_NAME}"
echo "TARGET_MODE=${TARGET_MODE}"
echo "K=${K}"
echo "BLOCK_INDICES=${BLOCK_INDICES}"
echo "X_CODEBOOK_MODE=${X_CODEBOOK_MODE}"
echo "OUTDIR=${OUTDIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PY_SCRIPT=${PY_SCRIPT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FIT_DEVICES=${FIT_DEVICES}"
echo "BETA=${BETA} GAMMA=${GAMMA} LAMBDA_INNER_ITERS=${LAMBDA_INNER_ITERS} MAX_ITERS=${MAX_ITERS}"
echo "CALIB_NUM_TOKENS=${CALIB_NUM_TOKENS} STRIDE=${STRIDE} RUN_MODE=${RUN_MODE}"

python "${PY_SCRIPT}" \
    --target_mode "${TARGET_MODE}" \
    --block_indices "${BLOCK_INDICES}" \
    --model_name "${MODEL_PATH}" \
    --model_dtype "${MODEL_DTYPE}" \
    --output_dir "${OUTDIR}" \
    --device cuda:0 \
    --fit_devices "${FIT_DEVICES}" \
    --run_mode "${RUN_MODE}" \
    --calib_num_tokens "${CALIB_NUM_TOKENS}" \
    --stride "${STRIDE}" \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --lambda_inner_iters "${LAMBDA_INNER_ITERS}" \
    --max_iters "${MAX_ITERS}" \
    --init_mode "${INIT_MODE}" \
    --error_mode "${ERROR_MODE}" \
    --x_codebook_mode "${X_CODEBOOK_MODE}"

echo "End time: $(date)"
echo "=== 作业结束 ==="
