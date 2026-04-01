#!/bin/bash
#SBATCH --job-name=fpl-rl-v2
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fpl_rl_%j.out
#SBATCH --error=logs/fpl_rl_%j.err

set -e

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/fpl-rl
PYTHON=$WORKDIR/envs/fpl-rl/bin/python
cd $PROJECT

echo "============================================"
echo "FPL RL Training Job v2 (formation+bench)"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "============================================"

$PYTHON scripts/train_rl_cluster.py \
    --data-dir data/raw \
    --predictor-dir models/point_predictor \
    --run-dir runs/fpl_ppo \
    --total-steps 15000000 \
    --n-envs 30 \
    --batch-size 512 \
    --eval-freq 50000 \
    --checkpoint-freq 500000 \
    --n-eval-episodes 5

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
