#!/bin/bash
#SBATCH --job-name=fpl-ab-xp
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ab_test_%j.out
#SBATCH --error=logs/ab_test_%j.err

set -e
cd /gpfs/workdir/dalbanal/fpl-rl

echo "============================================"
echo "A/B Test: Shifted vs Unshifted xP"
echo "Job ID: $SLURM_JOB_ID"
echo "Start:  $(date)"
echo "============================================"

/gpfs/workdir/dalbanal/envs/fpl-rl/bin/python scripts/ab_test_xp.py

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
