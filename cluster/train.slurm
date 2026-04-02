#!/bin/bash
#SBATCH --job-name=fpl-hybrid
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fpl_rl_%j.out
#SBATCH --error=logs/fpl_rl_%j.err

set -e

WORKDIR=/gpfs/workdir/dalbanal
PROJECT=$WORKDIR/fpl-rl
PYTHON=$WORKDIR/envs/fpl-rl/bin/python
cd $PROJECT

echo "============================================"
echo "FPL RL Training — HYBRID (RL+MILP)"
echo "  - 2-dim action space (transfers + chips)"
echo "  - MILP optimizer handles player selection"
echo "  - No invalid actions possible"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "============================================"

rm -rf runs/fpl_ppo

$PYTHON scripts/train_rl_cluster.py \
    --hybrid \
    --data-dir data/raw \
    --predictor-dir models/point_predictor \
    --run-dir runs/fpl_ppo \
    --total-steps 20000000 \
    --n-envs 20 \
    --batch-size 256 \
    --n-steps 1024 \
    --eval-freq 20000 \
    --checkpoint-freq 200000 \
    --n-eval-episodes 5

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
