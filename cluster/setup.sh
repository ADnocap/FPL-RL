#!/bin/bash
# Setup script for ruche cluster — run once after uploading the project.
#
# Usage:
#   ssh ruche
#   cd /gpfs/workdir/dalbanal/fpl-rl
#   bash cluster/setup.sh

set -e

echo "Setting up FPL-RL environment on ruche..."

# Load Anaconda (has Python 3.11)
module load anaconda3/2023.09-0/none-none
echo "Python: $(python3 --version)"

# Create conda env with Python 3.11
if [ ! -d "$WORKDIR/envs/fpl-rl" ]; then
    conda create -y -p $WORKDIR/envs/fpl-rl python=3.11
    echo "Created conda environment"
fi

# Activate
source activate $WORKDIR/envs/fpl-rl

# Install the project and dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Verify key imports
python -c "
from sb3_contrib import MaskablePPO
from fpl_rl.env.fpl_env import FPLEnv
from fpl_rl.prediction.model import PointPredictor
print('All imports OK')
"

# Create logs directory
mkdir -p logs

echo ""
echo "Setup complete! Submit with:"
echo "  sbatch cluster/train.slurm"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/fpl_rl_*.out"
