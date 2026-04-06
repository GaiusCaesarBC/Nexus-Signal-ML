#!/bin/bash
# retrain_models.sh — Monthly model retraining script
# Run via cron: 0 2 1 * * /path/to/retrain_models.sh
# (2 AM on the 1st of every month)

echo "========================================="
echo "Nexus Signal AI — Monthly Model Retrain"
echo "Date: $(date)"
echo "========================================="

cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
fi

echo ""
echo "=== Training 7-day models (0.3% threshold) ==="
python scripts/train_multi_horizon.py --horizons 7 --threshold 0.3 --no-tuning

echo ""
echo "=== Training 30-day models (2.0% threshold) ==="
python scripts/train_multi_horizon.py --horizons 30 --threshold 2.0 --no-tuning

echo ""
echo "=== Training 90-day models (5.0% threshold) ==="
python scripts/train_multi_horizon.py --horizons 90 --threshold 5.0 --no-tuning

echo ""
echo "========================================="
echo "Retraining complete: $(date)"
echo "========================================="

# Auto-commit and push if git is available
if command -v git &> /dev/null; then
    git add -f trained_models/
    git commit -m "chore: Monthly model retrain $(date +%Y-%m-%d)"
    git push
    echo "Models pushed to GitHub"
fi
