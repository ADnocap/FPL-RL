#!/bin/bash
#SBATCH --job-name=fpl-rl-v3
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
echo "FPL RL Training Job v3 (rules audit fixes)"
echo "  - No observation lookahead"
echo "  - 5 transfer pairs (was 2)"
echo "  - GW1 squad reshaping (15 FTs)"
echo "  - Captain mask to lineup only"
echo "  - DGW/BGW flags populated"
echo "  - FH GW19/20 restriction"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "============================================"

# Delete old runs from previous (lookahead-contaminated) training
rm -rf runs/fpl_ppo

$PYTHON scripts/train_rl_cluster.py \
    --data-dir data/raw \
    --predictor-dir models/point_predictor \
    --run-dir runs/fpl_ppo \
    --total-steps 20000000 \
    --n-envs 38 \
    --batch-size 512 \
    --eval-freq 50000 \
    --checkpoint-freq 500000 \
    --n-eval-episodes 5

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
