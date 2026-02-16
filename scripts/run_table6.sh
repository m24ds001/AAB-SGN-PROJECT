#!/bin/bash
# scripts/run_table6.sh
# =====================
# Reproduce Table 6 (main paper): CIFAR-10, 40% symmetric noise, 5 seeds.
# Expected: Standalone AAB 93.85 ± 0.52%,  AAB-SGN 94.23 ± 0.45%
#
# Usage: chmod +x scripts/run_table6.sh && ./scripts/run_table6.sh
# Time : ~4h on single NVIDIA RTX 4090 (200 epochs/seed × 5 seeds)

set -e
mkdir -p results checkpoints

NOISE_RATE=0.40
SEEDS=(0 42 123 456 789)
RESULTS_FILE="results/table6_cifar10_40pct.txt"
echo "seed,best_acc,mode,kl_estimate" > "$RESULTS_FILE"

echo "========================================"
echo " Reproducing Table 6 — AAB-SGN CIFAR-10"
echo " noise_rate=${NOISE_RATE}, 5 seeds"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed ${SEED} ---"
    python experiments/train_cifar10.py \
        --noise_rate  ${NOISE_RATE} \
        --seed        ${SEED} \
        --epochs      200 \
        --batch_size  128 \
        --lr          0.05 \
        --pi_min      0.10 \
        --mode        auto \
        2>&1 | tee results/cifar10_seed${SEED}.log

    ACC=$(grep "Best Test Accuracy" results/cifar10_seed${SEED}.log | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    MODE=$(grep "Mode" results/cifar10_seed${SEED}.log | tail -1 | awk '{print $NF}')
    KL=$(grep "KL Estimate" results/cifar10_seed${SEED}.log | tail -1 | awk '{print $NF}')
    echo "${SEED},${ACC},${MODE},${KL}" >> "$RESULTS_FILE"
done

echo ""
echo "========================================"
echo " SUMMARY"
cat "$RESULTS_FILE"
python3 - << 'EOF'
import csv, statistics
with open("results/table6_cifar10_40pct.txt") as f:
    rows = list(csv.DictReader(f))
accs = [float(r["best_acc"]) for r in rows if r["best_acc"]]
if accs:
    print(f" Mean ± Std: {statistics.mean(accs):.2f} ± {statistics.stdev(accs):.2f}%")
    print(f" Expected  : 93.85 ± 0.52% (SA) / 94.23 ± 0.45% (two-stage)")
EOF
echo "========================================"
