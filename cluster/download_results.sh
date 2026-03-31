#!/bin/bash
# Download training results from ruche cluster.
# Run from the project root on your local machine.
#
# Usage:
#   bash cluster/download_results.sh           # download everything
#   bash cluster/download_results.sh best      # just best model
#   bash cluster/download_results.sh latest     # just latest checkpoint

set -e

REMOTE="dalbanal@ruche.mesocentre.universite-paris-saclay.fr"
REMOTE_DIR="/gpfs/workdir/dalbanal/fpl-rl"
MODE=${1:-all}

echo "Downloading results from ruche ($MODE)..."

case $MODE in
    best)
        echo "Downloading best model..."
        rsync -avz $REMOTE:$REMOTE_DIR/runs/fpl_ppo/best_model/ runs/fpl_ppo/best_model/
        ;;
    latest)
        echo "Downloading latest checkpoint..."
        # Get the most recent checkpoint file
        LATEST=$(ssh $REMOTE "ls -t $REMOTE_DIR/runs/fpl_ppo/checkpoints/*.zip 2>/dev/null | head -1")
        if [ -n "$LATEST" ]; then
            mkdir -p runs/fpl_ppo/checkpoints
            scp $REMOTE:$LATEST runs/fpl_ppo/checkpoints/
            echo "Downloaded: $(basename $LATEST)"
        else
            echo "No checkpoints found"
        fi
        ;;
    all|*)
        echo "Downloading all results..."
        mkdir -p runs/fpl_ppo
        rsync -avz --progress \
            $REMOTE:$REMOTE_DIR/runs/fpl_ppo/ \
            runs/fpl_ppo/
        ;;
esac

# Always download logs
mkdir -p logs
rsync -avz $REMOTE:$REMOTE_DIR/logs/ logs/ 2>/dev/null || true

echo ""
echo "Done! Check:"
echo "  runs/fpl_ppo/best_model/     - best eval model"
echo "  runs/fpl_ppo/checkpoints/    - periodic checkpoints"
echo "  runs/fpl_ppo/final_model/    - final model"
echo "  runs/fpl_ppo/tb_logs/        - TensorBoard metrics"
echo "  logs/                        - SLURM output"
