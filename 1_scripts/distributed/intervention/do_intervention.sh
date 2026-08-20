#!/usr/bin/env bash
# One-shot intervention: kill old fleet queue processes, launch resume queue.
# Run this AFTER initr_16/lambda_10 finishes (watch for PASS line in fahqlora log).
#
# Usage:
#   bash 1_scripts/distributed/intervention/do_intervention.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEDLORA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "[$(date '+%H:%M:%S')] === Starting intervention ==="

# Step 1: Kill the old fleet processes
echo "[$(date '+%H:%M:%S')] Killing fleet_master_queue.sh (PID 1769776)..."
kill 1769776 2>/dev/null && echo "  killed master queue" || echo "  master queue already dead"

echo "[$(date '+%H:%M:%S')] Killing fleet_queue_fahqlora.sh (PID 2932665)..."
kill 2932665 2>/dev/null && echo "  killed fahqlora queue" || echo "  fahqlora queue already dead"

# Kill any leftover sleep from the cooldown
pkill -f "sleep 30" 2>/dev/null || true

# Verify no server or full_fl_run still running
if pgrep -f "distributed.server.main" > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] WARNING: FL server still running. Waiting for it to finish..."
    echo "  If this hangs, the experiment hasn't finished yet. Ctrl+C and retry later."
    while pgrep -f "distributed.server.main" > /dev/null 2>&1; do
        sleep 5
    done
fi

echo "[$(date '+%H:%M:%S')] Old fleet processes killed."

# Step 2: Launch the resume queue
echo "[$(date '+%H:%M:%S')] Launching fleet_resume_after_intervention.sh..."
cd "$FEDLORA_ROOT"
nohup bash 1_scripts/distributed/intervention/fleet_resume_after_intervention.sh \
    2>&1 | tee -a exp2/fleet_master_queue.log &
RESUME_PID=$!

echo "[$(date '+%H:%M:%S')] Resume queue launched (PID ${RESUME_PID})"
echo "[$(date '+%H:%M:%S')] Order: HetLoRA retry → FAH-QLoRA (initr_32,64) → v2 → v3"
echo "[$(date '+%H:%M:%S')] === Intervention complete ==="
