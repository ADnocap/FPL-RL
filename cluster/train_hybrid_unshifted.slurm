#!/bin/bash
#SBATCH --job-name=rl-unshift
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rl_unshifted_%j.out
#SBATCH --error=logs/rl_unshifted_%j.err

set -e
cd /gpfs/workdir/dalbanal/fpl-rl
PYTHON=/gpfs/workdir/dalbanal/envs/fpl-rl/bin/python

echo "============================================"
echo "Hybrid RL Training — UNSHIFTED xP"
echo "Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "============================================"

rm -rf runs/hybrid_unshifted

$PYTHON scripts/train_rl_cluster.py \
    --hybrid \
    --data-dir data/raw \
    --predictor-dir models/ab_test_unshifted \
    --run-dir runs/hybrid_unshifted \
    --total-steps 5000000 \
    --n-envs 12 \
    --batch-size 256 \
    --n-steps 1024 \
    --eval-freq 20000 \
    --checkpoint-freq 500000 \
    --n-eval-episodes 5

echo "Finished: $(date)"
