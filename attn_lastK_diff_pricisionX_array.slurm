#!/bin/bash
#SBATCH --job-name=basis_lastk_xprec
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1
#
#SBATCH --time=0-12:00:00
#SBATCH --mem=48g
#SBATCH --array=0-20%1

# 用法示例：
#   sbatch attn_lastK_diff_pricisionX_array.slurm
#   sbatch --array=0-20%4 attn_lastK_diff_pricisionX_array.slurm
#   sbatch --export=ALL,PROJECT_ROOT=/ssdfs/datahome/u24170/sobq-exp attn_lastK_diff_pricisionX_array.slurm
#   sbatch --export=ALL,MODEL_NAME_OR_PATH=/path/to/local/model attn_lastK_diff_pricisionX_array.slurm
#
# sweep 维度说明（与 basis_diff_pricisionX.py 参数保持一致）：
#   默认 K 候选值为 (8 12 16 20 24 28 31)
#   脚本会根据模型层数 TOTAL_BLOCKS 自动过滤非法 K
#   X_CODEBOOK_MODES=(int4 int6 none)
#   这里的 --array=0-20 是 32 层模型的安全上界；层数更小的模型会自动跳过多余 task

echo "Time is $(date)"
echo "Directory is $PWD"
echo "This job runs on the following nodes:"
echo "$SLURM_JOB_NODELIST"
echo "This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores."
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<none>}"

module purge
module load anaconda3/24.1.2
module load cuda
source ~/.bashrc
conda activate quant

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdfs/datahome/u24170/sobq-exp}"
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/basis_diff_pricisionX.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs_basis_diff_pricisionX}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-2-7b-hf}"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
mkdir -p logs "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$OUTPUT_ROOT"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Python script not found: $SCRIPT_PATH"
    exit 1
fi

TOTAL_BLOCKS="${TOTAL_BLOCKS:-}"
if [[ -z "$TOTAL_BLOCKS" ]]; then
    export MODEL_NAME_OR_PATH
    TOTAL_BLOCKS=$(python - <<'PY'
import os
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
for attr_name in ("num_hidden_layers", "n_layer", "num_layers", "num_hidden_layers_decoder"):
    value = getattr(cfg, attr_name, None)
    if value is not None:
        print(int(value))
        break
else:
    raise SystemExit("Could not infer TOTAL_BLOCKS from model config")
PY
)
fi

if ! [[ "$TOTAL_BLOCKS" =~ ^[0-9]+$ ]]; then
    echo "TOTAL_BLOCKS must be an integer, got: $TOTAL_BLOCKS"
    exit 1
fi

if [[ -n "${K_VALUES_CSV:-}" ]]; then
    IFS=',' read -r -a CANDIDATE_K_VALUES <<< "$K_VALUES_CSV"
else
    CANDIDATE_K_VALUES=(8 12 16 20 24 28 31)
fi

K_VALUES=()
for k in "${CANDIDATE_K_VALUES[@]}"; do
    k="${k// /}"
    [[ -z "$k" ]] && continue
    if (( k <= TOTAL_BLOCKS )); then
        K_VALUES+=("$k")
    fi
done

if (( ${#K_VALUES[@]} == 0 )); then
    K_VALUES=("$TOTAL_BLOCKS")
fi

if [[ -n "${X_CODEBOOK_MODES_CSV:-}" ]]; then
    IFS=',' read -r -a X_CODEBOOK_MODES <<< "$X_CODEBOOK_MODES_CSV"
else
    X_CODEBOOK_MODES=(int4 int6 none)
fi

NUM_K=${#K_VALUES[@]}
NUM_X_MODE=${#X_CODEBOOK_MODES[@]}
MAX_ARRAY_ID=$((NUM_K * NUM_X_MODE - 1))
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( ARRAY_ID < 0 )); then
    echo "Invalid array index: ARRAY_ID=${ARRAY_ID}"
    exit 1
fi

if (( ARRAY_ID > MAX_ARRAY_ID )); then
    echo "ARRAY_ID=${ARRAY_ID} exceeds supported combinations for TOTAL_BLOCKS=${TOTAL_BLOCKS}."
    echo "Active K_VALUES=(${K_VALUES[*]})"
    echo "Supported ARRAY_ID range is [0, ${MAX_ARRAY_ID}]. Skipping this task."
    exit 0
fi

K_ID=$((ARRAY_ID / NUM_X_MODE))
X_MODE_ID=$((ARRAY_ID % NUM_X_MODE))
K=${K_VALUES[$K_ID]}
X_CODEBOOK_MODE=${X_CODEBOOK_MODES[$X_MODE_ID]}

BETA="${BETA:-1}"
GAMMA="${GAMMA:-1e3}"
LAMBDA_INNER_ITERS="${LAMBDA_INNER_ITERS:-3}"
MAX_ITERS="${MAX_ITERS:-80}"
CALIB_NUM_TOKENS="${CALIB_NUM_TOKENS:-4096}"
STRIDE="${STRIDE:-512}"
RUN_MODE="${RUN_MODE:-discrete}"
MODEL_DTYPE="${MODEL_DTYPE:-float16}"
INIT_MODE="${INIT_MODE:-random}"
ERROR_MODE="${ERROR_MODE:-relative}"
TARGET_MODE="${TARGET_MODE:-qkvo}"
DEVICE="${DEVICE:-cuda:0}"
FIT_DEVICES="${FIT_DEVICES:-cuda:0}"
RUN_BASELINE_PPL="${RUN_BASELINE_PPL:-false}"
RUN_SQ_BASELINE="${RUN_SQ_BASELINE:-false}"

START_BLOCK=$((TOTAL_BLOCKS - K))
BLOCK_INDICES=$(seq -s, "$START_BLOCK" $((TOTAL_BLOCKS - 1)))

EXP_NAME="last${K}_${TARGET_MODE}_x${X_CODEBOOK_MODE}"
OUTDIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$OUTDIR"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "SCRIPT_PATH=$SCRIPT_PATH"
echo "MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "HF_HOME=$HF_HOME"
echo "TOTAL_BLOCKS=$TOTAL_BLOCKS"
echo "K_VALUES=(${K_VALUES[*]})"
echo "ARRAY_ID=$ARRAY_ID"
echo "EXP_NAME=$EXP_NAME"
echo "TARGET_MODE=$TARGET_MODE"
echo "K=$K"
echo "BLOCK_INDICES=$BLOCK_INDICES"
echo "X_CODEBOOK_MODE=$X_CODEBOOK_MODE"
echo "DEVICE=$DEVICE"
echo "FIT_DEVICES=$FIT_DEVICES"
echo "BETA=$BETA GAMMA=$GAMMA LAMBDA_INNER_ITERS=$LAMBDA_INNER_ITERS MAX_ITERS=$MAX_ITERS"
echo "CALIB_NUM_TOKENS=$CALIB_NUM_TOKENS STRIDE=$STRIDE RUN_MODE=$RUN_MODE"
echo "RUN_BASELINE_PPL=$RUN_BASELINE_PPL RUN_SQ_BASELINE=$RUN_SQ_BASELINE"
echo "OUTDIR=$OUTDIR"

python "$SCRIPT_PATH" \
    --target_mode "$TARGET_MODE" \
    --block_indices "$BLOCK_INDICES" \
    --model_name "$MODEL_NAME_OR_PATH" \
    --model_dtype "$MODEL_DTYPE" \
    --output_dir "$OUTDIR" \
    --device "$DEVICE" \
    --fit_devices "$FIT_DEVICES" \
    --run_mode "$RUN_MODE" \
    --calib_num_tokens "$CALIB_NUM_TOKENS" \
    --stride "$STRIDE" \
    --beta "$BETA" \
    --gamma "$GAMMA" \
    --lambda_inner_iters "$LAMBDA_INNER_ITERS" \
    --max_iters "$MAX_ITERS" \
    --init_mode "$INIT_MODE" \
    --error_mode "$ERROR_MODE" \
    --run_baseline_ppl "$RUN_BASELINE_PPL" \
    --run_sq_baseline "$RUN_SQ_BASELINE" \
    --x_codebook_mode "$X_CODEBOOK_MODE"

echo "End time: $(date)"
