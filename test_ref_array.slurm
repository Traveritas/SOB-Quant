#!/bin/bash
#SBATCH --job-name=qkvo_ref_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1

#SBATCH --time=0-01:00:00
#SBATCH --mem=4g
#SBATCH --array=0-0%1

# 用法示例：
#   NUM=$(python opt_qkvo_refactored_array.py --print-num-combos)
#   sbatch --array=0-$((NUM-1))%4 test_ref_array.slurm
#
# 若所有 array task 结束后想重建一次总汇总：
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> --export=ALL,REBUILD_SUMMARY_ONLY=1 test_ref_array.slurm

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
export HF_HOME=/ssdfs/datahome/u24170/sobq-exp/.cache/huggingface
export HF_HUB_CACHE=/ssdfs/datahome/u24170/sobq-exp/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/ssdfs/datahome/u24170/sobq-exp/.cache/huggingface/hub
export HF_DATASETS_CACHE=/ssdfs/datahome/u24170/sobq-exp/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /ssdfs/datahome/u24170/sobq-exp/

SCRIPT=opt_qkvo_refactored_array.py
OUT_ROOT=./out_qkvo_ref

echo "HF_HOME=$HF_HOME"
echo "SCRIPT=$SCRIPT"
echo "OUT_ROOT=$OUT_ROOT"

if [[ "${REBUILD_SUMMARY_ONLY:-0}" == "1" ]]; then
    echo "Rebuilding sweep summary only..."
    python "$SCRIPT" --output-root "$OUT_ROOT" --rebuild-summary
    exit 0
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    python "$SCRIPT" --output-root "$OUT_ROOT" --array-task-id "$SLURM_ARRAY_TASK_ID"
else
    echo "No SLURM_ARRAY_TASK_ID detected; running the full sweep sequentially."
    python "$SCRIPT" --output-root "$OUT_ROOT"
fi
