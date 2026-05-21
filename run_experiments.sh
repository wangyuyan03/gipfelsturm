#!/bin/bash
#
# Master experiment runner for Gipfelsturm project.
# Submits SLURM jobs in named groups and records all job IDs.
#
# Usage:
#   ./run_experiments.sh throughput          — model size + batch size sweep
#   ./run_experiments.sh lr <steps>          — LR schedule ablation (cosine vs WSD)
#   ./run_experiments.sh final <steps> <schedule> [wsd_pct]  — final 30-min run
#   ./run_experiments.sh status              — show all submitted jobs
#
# Examples:
#   ./run_experiments.sh throughput
#   ./run_experiments.sh lr 1500
#   ./run_experiments.sh final 3200 cosine
#   ./run_experiments.sh final 3200 WSD 30

set -euo pipefail

source "$(dirname "$0")/config.sh"

RESULTS_DIR="$WORKDIR/results"
JOB_LOG="$RESULTS_DIR/job_ids.log"
DRY_RUN=false

# Parse top-level --dry-run flag (must come before the subcommand)
_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        _ARGS+=("$arg")
    fi
done
set -- "${_ARGS[@]}"

mkdir -p "$RESULTS_DIR"

# Submit a job via launch.sh and record the job ID.
# In dry-run mode, generate the sbatch script but do not submit.
submit() {
    local group=$1
    local name=$2
    shift 2
    local output
    if [[ "$DRY_RUN" == "true" ]]; then
        output=$(./launch.sh "$@" --dry-run 2>&1)
        echo "$output"
        return
    fi
    output=$(./launch.sh "$@" 2>&1)
    local jobid
    jobid=$(echo "$output" | grep "Submitted batch job" | awk '{print $NF}')
    if [[ -z "$jobid" ]]; then
        echo "ERROR: submission failed for $name"
        echo "$output"
        exit 1
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $group | $name | $jobid" | tee -a "$JOB_LOG"
    echo "$jobid"
}

# ─────────────────────────────────────────────────────────────────────────────
# THROUGHPUT — measures tokens/sec/GPU for each model/batch combo.
# These results determine how many steps fit in 30 minutes AND serve as the
# batch size trade-off experiment for the report.
# ─────────────────────────────────────────────────────────────────────────────
run_throughput() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " Submitting THROUGHPUT experiments"
    echo " Purpose: measure tokens/sec/GPU, pick model + GBS,"
    echo "          and produce all throughput tables for report."
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # ── (A) Model size sweep at GBS=256, 1 node ──────────────────────────────
    # Report table: tokens/sec/GPU vs model size
    # Also tells us which model size maximises tokens in 30 min
    for model in 125m 350m 760m 1.5b 3b; do
        submit throughput "${model}-gbs256-1n" throughput "$model" 50 1
    done

    # ── (B) Batch size sweep for 760m, 1 node ────────────────────────────────
    # Report table: tokens/sec/GPU + convergence trade-off vs GBS
    for gbs in 128 512; do
        # Throughput only (50 steps)
        submit throughput "760m-gbs${gbs}-1n" throughput 760m 50 1 --gbs "$gbs"
    done

    # ── (C) Scaling: 760m across 1 / 2 / 4 nodes ─────────────────────────────
    # Report plot: scaling efficiency (tokens/sec/GPU vs node count)
    for nodes in 2 4; do
        submit throughput "760m-gbs256-${nodes}n" throughput 760m 50 "$nodes"
    done

    echo ""
    echo "All throughput jobs submitted. Job log: $JOB_LOG"
    echo ""
    echo "Next: wait for jobs to complete, then run:"
    echo "  python3 analyze.py --group throughput"
    echo "to get tokens/sec/GPU numbers, plots, and N_steps for 30 min."
}

# ─────────────────────────────────────────────────────────────────────────────
# BATCH SIZE CONVERGENCE — short train runs at each GBS to compare
# validation loss per token (not just throughput). Complements the throughput
# batch size sweep with actual convergence data.
# Run AFTER throughput so you know the right step count.
# ─────────────────────────────────────────────────────────────────────────────
run_batchsize() {
    local steps=${1:?Usage: ./run_experiments.sh batchsize <steps>}

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " Submitting BATCH SIZE CONVERGENCE experiments"
    echo " Steps: $steps  |  Model: 760m  |  Nodes: 1"
    echo " Compares val loss at same compute budget across GBS"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # GBS=256 cosine is already submitted as part of the LR ablation (cosine baseline).
    # Only submit GBS=128 and GBS=512 here — analyze.py uses the cosine lr run for GBS=256.
    for gbs in 128 512; do
        submit batchsize "760m-gbs${gbs}-${steps}s" \
            batchsize 760m "$steps" 1 --gbs "$gbs" --lr-schedule cosine
    done

    echo ""
    echo "Batch size convergence jobs submitted. Job log: $JOB_LOG"
    echo ""
    echo "Note: GBS=256 is shared with the LR ablation cosine run — no duplicate needed."
    echo "After jobs complete, run:"
    echo "  python3 analyze.py --group batchsize"
}

# ─────────────────────────────────────────────────────────────────────────────
# LR ABLATION — proxy training runs comparing Cosine vs WSD schedules.
# Run AFTER throughput so you know the right step count.
# ─────────────────────────────────────────────────────────────────────────────
run_lr_ablation() {
    local steps=${1:?Usage: ./run_experiments.sh lr <steps>}

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " Submitting LR ABLATION experiments"
    echo " Steps: $steps  |  Model: 760m  |  Nodes: 1"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # Cosine baseline
    submit lr "760m-cosine-${steps}s" train 760m "$steps" 1 --lr-schedule cosine

    # WSD variants — 20%, 30%, 40% of steps used for the decay phase
    for pct in 20 30 40; do
        submit lr "760m-wsd${pct}-${steps}s" \
            train 760m "$steps" 1 --lr-schedule WSD --wsd-decay-pct "$pct"
    done

    echo ""
    echo "All LR ablation jobs submitted. Job log: $JOB_LOG"
    echo ""
    echo "Next: wait for jobs to complete, then run:"
    echo "  python3 analyze.py --group lr"
    echo "to compare validation loss curves and generate plots."
}

# ─────────────────────────────────────────────────────────────────────────────
# FINAL RUN — the real 30-minute training job with the best config.
# Run AFTER LR ablation results are in.
# ─────────────────────────────────────────────────────────────────────────────
run_final() {
    local steps=${1:?Usage: ./run_experiments.sh final <steps> <cosine|WSD> [wsd_pct]}
    local schedule=${2:?Usage: ./run_experiments.sh final <steps> <cosine|WSD> [wsd_pct]}
    local wsd_pct=${3:-30}

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " Submitting FINAL 30-min run"
    echo " Steps: $steps | Schedule: $schedule | Model: 760m | Nodes: 1"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    if [[ "$schedule" == "WSD" ]]; then
        submit final "FINAL-760m-${schedule}${wsd_pct}-${steps}s" \
            train 760m "$steps" 1 --lr-schedule WSD --wsd-decay-pct "$wsd_pct"
    else
        submit final "FINAL-760m-${schedule}-${steps}s" \
            train 760m "$steps" 1 --lr-schedule "$schedule"
    fi

    echo ""
    echo "Final job submitted. Job log: $JOB_LOG"
}

# ─────────────────────────────────────────────────────────────────────────────
# STATUS — show current state of all submitted jobs
# ─────────────────────────────────────────────────────────────────────────────
show_status() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " All submitted jobs (from $JOB_LOG)"
    echo "═══════════════════════════════════════════════════════"
    if [[ ! -f "$JOB_LOG" ]]; then
        echo "No jobs submitted yet."
        return
    fi

    # Cross-reference job_ids.log with SLURM state and log files
    printf "%-12s %-30s %-10s %s\n" "JOBID" "NAME" "STATE" "LOG"
    echo "────────────────────────────────────────────────────────────────────"
    while IFS='|' read -r ts group name jobid; do
        jobid=$(echo "$jobid" | tr -d ' ')
        name=$(echo "$name" | tr -d ' ')
        state=$(sacct -j "$jobid" --noheader --format=State --parsable2 2>/dev/null | head -1)
        [[ -z "$state" ]] && state="UNKNOWN"
        logfile=$(ls logs/*-${jobid}.log 2>/dev/null | head -1)
        [[ -z "$logfile" ]] && logfile="(not yet)"
        printf "%-12s %-30s %-10s %s\n" "$jobid" "$name" "$state" "$logfile"
    done < "$JOB_LOG"

    echo ""
    echo "Current queue:"
    squeue --me
}

# ─────────────────────────────────────────────────────────────────────────────
# RESUBMIT — detect failed/cancelled jobs from job_ids.log and resubmit them
# ─────────────────────────────────────────────────────────────────────────────
resubmit_failed() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " Checking for failed jobs and resubmitting..."
    echo "═══════════════════════════════════════════════════════"
    echo ""

    if [[ ! -f "$JOB_LOG" ]]; then
        echo "No job log found. Nothing to resubmit."
        return
    fi

    local resubmitted=0
    while IFS='|' read -r ts group name jobid; do
        group=$(echo "$group" | tr -d ' ')
        name=$(echo "$name" | tr -d ' ')
        jobid=$(echo "$jobid" | tr -d ' ')

        # Check if this job ended in failure
        state=$(sacct -j "$jobid" --noheader --format=State --parsable2 2>/dev/null | head -1)

        case "$state" in
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                echo "FAILED ($state): $name (job $jobid) — resubmitting from sbatch script..."
                # Find the generated sbatch script by name pattern
                sbatch_file=$(ls logs/gipfel-*${jobid}*.sbatch 2>/dev/null | head -1)
                if [[ -z "$sbatch_file" ]]; then
                    # Try matching by name (without jobid)
                    sbatch_file=$(ls logs/*.sbatch 2>/dev/null | grep -i "$(echo $name | sed 's/-[0-9]*s/-/')" | head -1)
                fi
                if [[ -n "$sbatch_file" ]]; then
                    new_jobid=$(sbatch "$sbatch_file" | awk '{print $NF}')
                    echo "$(date '+%Y-%m-%d %H:%M:%S') | ${group}-RETRY | $name | $new_jobid" | tee -a "$JOB_LOG"
                    ((resubmitted++))
                else
                    echo "  WARNING: could not find sbatch script for $name — rerun manually"
                fi
                ;;
            COMPLETED)
                echo "OK: $name (job $jobid)"
                ;;
            RUNNING|PENDING)
                echo "ACTIVE: $name (job $jobid) — $state"
                ;;
            *)
                echo "UNKNOWN ($state): $name (job $jobid)"
                ;;
        esac
    done < "$JOB_LOG"

    echo ""
    echo "Resubmitted: $resubmitted jobs"
}

# ─────────────────────────────────────────────────────────────────────────────
case ${1:?Usage: ./run_experiments.sh <throughput|batchsize|lr|final|status|resubmit> [args]} in
    throughput) run_throughput ;;
    batchsize)  run_batchsize "${2:-}" ;;
    lr)         run_lr_ablation "${2:-}" ;;
    final)      run_final "${2:-}" "${3:-}" "${4:-}" ;;
    status)     show_status ;;
    resubmit)   resubmit_failed ;;
    *)
        echo "Unknown group: $1. Choose: throughput, batchsize, lr, final, status, resubmit"
        exit 1
        ;;
esac
