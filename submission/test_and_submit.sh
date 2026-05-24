#!/bin/bash
#
# Submit a quick sanity test job, wait for it to complete, check the log
# for a healthy training environment, then submit the full throughput suite.
#
# Usage:
#   ./test_and_submit.sh            # submits a fresh 5-step test job
#   ./test_and_submit.sh <jobid>    # reuse an already-submitted test job

set -euo pipefail

source "$(dirname "$0")/config.sh"

TIMEOUT_SECS=86400  # 24h max wait — covers overnight queue delays
POLL_PENDING=300    # check every 5 min while job is still pending
POLL_RUNNING=30     # check every 30 s once job is actually running

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    TEST_JOBID=$1
    echo "Reusing existing test job: $TEST_JOBID"
else
    echo "Submitting 5-step 125m sanity test..."
    output=$(./launch.sh throughput 125m 5 1 2>&1)
    TEST_JOBID=$(echo "$output" | grep "Submitted batch job" | awk '{print $NF}')
    if [[ -z "$TEST_JOBID" ]]; then
        echo "ERROR: failed to submit test job"
        echo "$output"
        exit 1
    fi
    scontrol update JobId="$TEST_JOBID" TimeLimit=20:00 2>/dev/null \
        && echo "Time limit set to 20 min" \
        || echo "Note: could not update time limit (job may have already started)"
    echo "Submitted test job: $TEST_JOBID"
fi

echo ""
echo "Waiting for job $TEST_JOBID (timeout: 24h)..."
echo "Polling every 5 min while pending, every 30 s once running."
echo ""

elapsed=0
final_state=""

while true; do
    state=$(sacct -j "$TEST_JOBID" --noheader --format=State --parsable2 2>/dev/null | head -1)

    case "${state:-}" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
            final_state="$state"
            echo "[$(date '+%H:%M:%S')] Job finished with state: $final_state"
            break
            ;;
        RUNNING)
            echo "[$(date '+%H:%M:%S')] Running..."
            sleep "$POLL_RUNNING"
            elapsed=$((elapsed + POLL_RUNNING))
            ;;
        PENDING)
            # Show estimated start time when available
            start=$(squeue -j "$TEST_JOBID" --noheader --format="%S" 2>/dev/null | head -1)
            echo "[$(date '+%H:%M:%S')] Pending (est. start: ${start:-unknown})..."
            sleep "$POLL_PENDING"
            elapsed=$((elapsed + POLL_PENDING))
            ;;
        *)
            echo "[$(date '+%H:%M:%S')] State: ${state:-unknown}"
            sleep "$POLL_PENDING"
            elapsed=$((elapsed + POLL_PENDING))
            ;;
    esac

    if [[ $elapsed -ge $TIMEOUT_SECS ]]; then
        echo ""
        echo "ERROR: timed out after 24h. Job $TEST_JOBID is still ${state:-unknown}."
        echo "Check manually: squeue --me"
        exit 1
    fi
done

echo ""
logfile=$(ls logs/*-"${TEST_JOBID}".log 2>/dev/null | head -1)

if [[ -z "$logfile" ]]; then
    echo "ERROR: no log file found for job $TEST_JOBID in logs/"
    exit 1
fi

echo "Log file: $logfile"
echo ""

if grep -q "ImportError" "$logfile"; then
    echo "FAIL: ImportError in log — conda PATH fix did not work."
    echo ""
    grep "ImportError" "$logfile"
    echo ""
    echo "Do NOT submit throughput jobs. Investigate the error above."
    exit 1
fi

if grep -q "undefined symbol" "$logfile"; then
    echo "FAIL: symbol mismatch in log — Python/library version conflict."
    echo ""
    grep "undefined symbol" "$logfile"
    exit 1
fi

if ! grep -q "tokens/sec/GPU" "$logfile"; then
    echo "FAIL: no 'tokens/sec/GPU' found — training did not produce output."
    echo ""
    echo "Last 20 lines of log:"
    tail -20 "$logfile"
    echo ""
    echo "Do NOT submit throughput jobs."
    exit 1
fi

echo "════════════════════════════════════════════════════"
echo " SANITY TEST PASSED"
echo " Environment is healthy. Submitting throughput suite."
echo "════════════════════════════════════════════════════"
echo ""

echo "Sample output from test log:"
grep "tokens/sec/GPU" "$logfile" | head -3
echo ""

./run_experiments.sh throughput
