#!/usr/bin/env python3
"""
Parse SLURM logs and generate all report figures for the Gipfelsturm project.

Run this after an experiment group completes. It does everything in one pass:
  1. Parses every .log file in logs/ that matches the requested group
  2. Writes results/summary.csv and results/<job>.json (for reproducibility)
  3. Prints tables to stdout (throughput, LR ablation, batch size)
  4. Saves all plots to results/plots/

Usage:
  python3 analyze.py                     # parse + plot all groups
  python3 analyze.py --group throughput  # only throughput
  python3 analyze.py --group lr          # only LR ablation
  python3 analyze.py --group batchsize   # only batch size comparison
  python3 analyze.py --group final       # only final run
  python3 analyze.py --list              # list all parsed runs and exit
  python3 analyze.py --no-plots          # tables/CSV only, skip matplotlib

Workflow per phase:
  After throughput jobs finish:
    python3 analyze.py --group throughput
    -> prints tok/s/GPU table + how many steps fit in 30 min
    -> saves results/plots/throughput_model_size.png
    -> saves results/plots/throughput_batch_size.png
    -> saves results/plots/scaling_efficiency.png

  After LR ablation jobs finish:
    python3 analyze.py --group lr
    -> prints final val loss table
    -> saves results/plots/lr_val_loss.png   (key figure for report)
    -> saves results/plots/lr_train_loss.png

  After batch size jobs finish:
    python3 analyze.py --group batchsize
    -> saves results/plots/batchsize_val_loss.png

  After final run finishes:
    python3 analyze.py --group final
    -> saves results/plots/final_<jobname>.png
"""

from __future__ import print_function

import re
import json
import csv
import sys
import argparse
from pathlib import Path


# ── Log parsing ───────────────────────────────────────────────────────────────
#
# Megatron-LM log line format (with tokens/sec patch applied):
#   0: [default3]: [2026-05-08 12:13:47] iteration       10/      50 |
#     elapsed time per iteration (ms): 5368.0 |
#     tokens/sec/GPU: 48834 | lm loss: 8.995049E+00 | ...
#
# Validation line format:
#   validation loss at iteration 50 | lm loss value: 8.123456E+00

ITER_RE = re.compile(
    r'iteration\s+(\d+)/\s*\d+\s*\|'
    r'.*?elapsed time per iteration \(ms\):\s*([\d.]+)'
    r'.*?tokens/sec/GPU:\s*([\d.]+)'
    r'.*?lm loss:\s*([\d.E+\-]+)'
)

VAL_RE = re.compile(
    r'validation loss at iteration\s+(\d+)\s*\|'
    r'.*?lm loss value:\s*([\d.E+\-]+)'
)


class RunMetrics(object):
    def __init__(self, job_name, log_file, group):
        self.job_name = job_name
        self.log_file = log_file
        self.group = group
        self.iterations = []
        self.elapsed_ms = []
        self.tokens_per_sec_gpu = []
        self.train_loss = []
        self.val_loss = []               # list of (iter, loss) tuples
        self.avg_tokens_per_sec_gpu = None
        self.final_train_loss = None
        self.final_val_loss = None
        self.total_steps = 0

    def to_dict(self):
        return {
            "job_name": self.job_name,
            "log_file": self.log_file,
            "group": self.group,
            "iterations": self.iterations,
            "elapsed_ms": self.elapsed_ms,
            "tokens_per_sec_gpu": self.tokens_per_sec_gpu,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "avg_tokens_per_sec_gpu": self.avg_tokens_per_sec_gpu,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "total_steps": self.total_steps,
        }


def classify_group(job_name):
    name = job_name.lower()
    if "throughput" in name:
        return "throughput"
    if "final" in name:
        return "final"
    if "batchsize" in name:
        return "batchsize"
    if any(s in name for s in ["cosine", "wsd", "train"]):
        return "lr"
    return "unknown"


def parse_log(log_path):
    text = log_path.read_text(errors="replace")
    job_name = log_path.stem
    m = RunMetrics(job_name=job_name, log_file=str(log_path),
                   group=classify_group(job_name))

    for match in ITER_RE.finditer(text):
        m.iterations.append(int(match.group(1)))
        m.elapsed_ms.append(float(match.group(2)))
        m.tokens_per_sec_gpu.append(float(match.group(3)))
        m.train_loss.append(float(match.group(4)))

    for match in VAL_RE.finditer(text):
        m.val_loss.append((int(match.group(1)), float(match.group(2))))

    m.total_steps = max(m.iterations) if m.iterations else 0

    # Skip first 5 iterations — JIT warmup inflates times
    stable = m.tokens_per_sec_gpu[5:]
    if stable:
        m.avg_tokens_per_sec_gpu = sum(stable) / len(stable)

    if m.train_loss:
        m.final_train_loss = m.train_loss[-1]
    if m.val_loss:
        m.final_val_loss = m.val_loss[-1][1]

    return m


def steps_in_budget(tok_per_sec_gpu, n_gpus, wall_seconds, gbs, seq_len):
    return int(tok_per_sec_gpu * n_gpus * wall_seconds / (gbs * seq_len))


# ── Label / colour helpers ────────────────────────────────────────────────────

def lr_label(job_name):
    name = job_name.lower()
    if "wsd40" in name:
        return "WSD 40%"
    if "wsd30" in name:
        return "WSD 30%"
    if "wsd20" in name:
        return "WSD 20%"
    if "cosine" in name:
        return "Cosine"
    return job_name


COLORS_LR  = {"cosine": "#2196F3", "wsd20": "#FF9800",
               "wsd30": "#4CAF50", "wsd40": "#F44336"}
COLORS_GBS = {"128": "#9C27B0",   "256": "#2196F3", "512": "#FF9800"}


def lr_color(job_name):
    name = job_name.lower()
    for key, color in COLORS_LR.items():
        if key in name:
            return color
    return "#607D8B"


def gbs_color(job_name):
    m = re.search(r"gbs(\d+)", job_name)
    return COLORS_GBS.get(m.group(1), "#607D8B") if m else "#607D8B"


def lr_sort_key(r):
    name = r.job_name.lower()
    if "cosine" in name:
        return 0
    m = re.search(r"wsd(\d+)", name)
    return int(m.group(1)) if m else 99


def _smooth(values, n_points=50):
    w = max(1, len(values) // n_points)
    return [sum(values[max(0, i - w):i + 1]) / len(values[max(0, i - w):i + 1])
            for i in range(len(values))]


# ── Tables ────────────────────────────────────────────────────────────────────

def print_throughput_table(runs):
    thr = [r for r in runs if r.group == "throughput" and r.avg_tokens_per_sec_gpu]
    if not thr:
        return
    print()
    print("THROUGHPUT RESULTS")
    print("{:<42} {:>10}  {:>13}  {:>10}".format(
        "Run", "Tok/s/GPU", "Steps/30 min", "Steps/1 h"))
    print("─" * 82)
    for r in thr:
        tps = r.avg_tokens_per_sec_gpu
        s30 = steps_in_budget(tps, 4, 1800, 256, 4096)
        s60 = steps_in_budget(tps, 4, 3600, 256, 4096)
        print("{:<42} {:>10,.0f}  {:>13,}  {:>10,}".format(r.job_name, tps, s30, s60))
    print("─" * 82)
    print("  * assumes 4 GPUs, GBS=256, seq_len=4096")


def print_lr_table(runs):
    lr = [r for r in runs if r.group == "lr"]
    if not lr:
        return
    print()
    print("LR ABLATION RESULTS")
    print("{:<14} {:>6}  {:>16}  {:>14}".format(
        "Schedule", "Steps", "Final Train Loss", "Final Val Loss"))
    print("─" * 58)
    for r in sorted(lr, key=lr_sort_key):
        tl = "{:.4f}".format(r.final_train_loss) if r.final_train_loss else "—"
        vl = "{:.4f}".format(r.final_val_loss)   if r.final_val_loss   else "—"
        print("{:<14} {:>6}  {:>16}  {:>14}".format(
            lr_label(r.job_name), r.total_steps, tl, vl))
    print("─" * 58)


def print_batchsize_table(runs):
    bs = [r for r in runs if r.group == "batchsize"]
    if not bs:
        return

    def _gbs(r):
        m = re.search(r"gbs(\d+)", r.job_name)
        return int(m.group(1)) if m else 0

    print()
    print("BATCH SIZE RESULTS")
    print("{:<8} {:>6}  {:>16}  {:>14}".format(
        "GBS", "Steps", "Final Train Loss", "Final Val Loss"))
    print("─" * 52)
    for r in sorted(bs, key=_gbs):
        tl = "{:.4f}".format(r.final_train_loss) if r.final_train_loss else "—"
        vl = "{:.4f}".format(r.final_val_loss)   if r.final_val_loss   else "—"
        m  = re.search(r"gbs(\d+)", r.job_name)
        print("{:<8} {:>6}  {:>16}  {:>14}".format(
            m.group(1) if m else r.job_name, r.total_steps, tl, vl))
    print("─" * 52)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    })
    return plt


def _save(plt, fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


def plot_throughput(runs, out, plt):
    thr = [r for r in runs if r.group == "throughput" and r.avg_tokens_per_sec_gpu]

    # Model size bar chart (GBS=256, 1 node)
    order = ["125m", "350m", "760m", "1.5b", "3b"]
    size_data = {}
    for r in thr:
        if "1n" not in r.job_name or "gbs256" not in r.job_name:
            continue
        for size in order:
            if "-{}-".format(size) in r.job_name:
                size_data[size] = r.avg_tokens_per_sec_gpu

    if size_data:
        sizes = [s for s in order if s in size_data]
        vals  = [size_data[s] / 1000 for s in sizes]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(sizes, vals, color="#2196F3", width=0.55, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    "{:.1f}k".format(v), ha="center", va="bottom", fontsize=10)
        ax.set_xlabel("Model size")
        ax.set_ylabel("Tokens / sec / GPU (x1000)")
        ax.set_title("Throughput vs Model Size  (GBS=256, 1 node, seq=4096)")
        _save(plt, fig, out / "throughput_model_size.png")

    # Batch size bar chart (760m, 1 node)
    gbs_data = {}
    for r in thr:
        if "760m" not in r.job_name or "1n" not in r.job_name:
            continue
        m = re.search(r"gbs(\d+)", r.job_name)
        if m:
            gbs_data[int(m.group(1))] = r.avg_tokens_per_sec_gpu

    if gbs_data:
        keys = sorted(gbs_data)
        vals = [gbs_data[g] / 1000 for g in keys]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar([str(g) for g in keys], vals, color="#4CAF50", width=0.5,
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    "{:.1f}k".format(v), ha="center", va="bottom", fontsize=10)
        ax.set_xlabel("Global Batch Size")
        ax.set_ylabel("Tokens / sec / GPU (x1000)")
        ax.set_title("Throughput vs Batch Size  (760m, 1 node, seq=4096)")
        _save(plt, fig, out / "throughput_batch_size.png")

    # Scaling efficiency (760m, GBS=256, varying nodes)
    node_data = {}
    for r in thr:
        if "760m" not in r.job_name or "gbs256" not in r.job_name:
            continue
        m = re.search(r"(\d+)n", r.job_name)
        if m:
            node_data[int(m.group(1))] = r.avg_tokens_per_sec_gpu

    if len(node_data) >= 2:
        nodes    = sorted(node_data)
        baseline = node_data[nodes[0]]
        eff  = [node_data[n] / baseline * 100 for n in nodes]
        raw  = [node_data[n] / 1000 for n in nodes]
        ideal = [100.0] * len(nodes)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(nodes, raw, "o-", color="#2196F3", linewidth=2, markersize=7)
        for n, v in zip(nodes, raw):
            ax1.annotate("{:.1f}k".format(v), (n, v),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
        ax1.set_xlabel("Nodes (4 GPUs each)")
        ax1.set_ylabel("Tokens / sec / GPU (x1000)")
        ax1.set_title("Raw Throughput vs Nodes")
        ax1.set_xticks(nodes)

        ax2.plot(nodes, ideal, "--", color="#9E9E9E", linewidth=1.5,
                 label="Ideal (100%)")
        ax2.plot(nodes, eff, "o-", color="#F44336", linewidth=2,
                 markersize=7, label="Actual")
        for n, v in zip(nodes, eff):
            ax2.annotate("{:.0f}%".format(v), (n, v),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
        ax2.set_xlabel("Nodes (4 GPUs each)")
        ax2.set_ylabel("Scaling efficiency (%)")
        ax2.set_title("Scaling Efficiency (normalized to 1 node)")
        ax2.set_xticks(nodes)
        ax2.set_ylim(0, 115)
        ax2.legend()

        fig.suptitle("760m Multi-Node Scaling  (GBS=256, seq=4096)", y=1.02)
        fig.tight_layout()
        _save(plt, fig, out / "scaling_efficiency.png")


def plot_lr(runs, out, plt):
    lr_runs = sorted([r for r in runs if r.group == "lr"], key=lr_sort_key)

    # Val loss curves
    with_val = [r for r in lr_runs if r.val_loss]
    if with_val:
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in with_val:
            iters  = [v[0] for v in r.val_loss]
            losses = [v[1] for v in r.val_loss]
            ax.plot(iters, losses, "o-", label=lr_label(r.job_name),
                    color=lr_color(r.job_name), linewidth=2, markersize=5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Validation Loss")
        ax.set_title("LR Schedule Ablation - Validation Loss\n(760m, GBS=256, 1 node)")
        ax.legend(loc="upper right")
        _save(plt, fig, out / "lr_val_loss.png")

    # Training loss curves (smoothed)
    with_train = [r for r in lr_runs if r.train_loss]
    if with_train:
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in with_train:
            ax.plot(r.iterations, _smooth(r.train_loss),
                    label=lr_label(r.job_name),
                    color=lr_color(r.job_name), linewidth=1.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training Loss (smoothed)")
        ax.set_title("LR Schedule Ablation - Training Loss\n(760m, GBS=256, 1 node)")
        ax.legend(loc="upper right")
        _save(plt, fig, out / "lr_train_loss.png")


def plot_batchsize(runs, out, plt):
    def _gbs(r):
        m = re.search(r"gbs(\d+)", r.job_name)
        return int(m.group(1)) if m else 0

    bs_runs = sorted([r for r in runs if r.group == "batchsize" and r.val_loss],
                     key=_gbs)
    if not bs_runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in bs_runs:
        iters  = [v[0] for v in r.val_loss]
        losses = [v[1] for v in r.val_loss]
        m = re.search(r"gbs(\d+)", r.job_name)
        label = "GBS {}".format(m.group(1)) if m else r.job_name
        ax.plot(iters, losses, "o-", label=label,
                color=gbs_color(r.job_name), linewidth=2, markersize=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Batch Size Ablation - Validation Loss\n(760m, same step count, 1 node)")
    ax.legend(loc="upper right")
    _save(plt, fig, out / "batchsize_val_loss.png")


def plot_final(runs, out, plt):
    for r in [r for r in runs if r.group == "final"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        if r.train_loss:
            ax1.plot(r.iterations, _smooth(r.train_loss),
                     color="#2196F3", linewidth=1.5)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Training Loss (smoothed)")
        ax1.set_title("Training Loss")

        if r.val_loss:
            iters  = [v[0] for v in r.val_loss]
            losses = [v[1] for v in r.val_loss]
            ax2.plot(iters, losses, "o-", color="#4CAF50", linewidth=2, markersize=5)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Validation Loss")
            ax2.set_title("Validation Loss  (final = {:.4f})".format(losses[-1]))

        fig.suptitle("Final 30-min Run - 760m, {}".format(lr_label(r.job_name)),
                     y=1.02)
        fig.tight_layout()
        _save(plt, fig, out / "final_{}.png".format(r.job_name))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse Gipfelsturm training logs and generate report figures")
    parser.add_argument("--group",
                        choices=["throughput", "lr", "batchsize", "final", "all"],
                        default="all")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--list", action="store_true",
                        help="List all parsed runs and exit")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation (tables + CSV only)")
    args = parser.parse_args()

    logs_dir    = Path(args.logs_dir)
    results_dir = Path(args.results_dir)
    plots_dir   = results_dir / "plots"
    results_dir.mkdir(exist_ok=True)

    log_files = sorted(logs_dir.glob("*.log"))
    if not log_files:
        print("No .log files found in {}/".format(logs_dir))
        sys.exit(1)

    all_metrics = []
    for lf in log_files:
        m = parse_log(lf)
        if args.group != "all" and m.group != args.group:
            continue
        if not m.iterations:
            continue   # skip infra-test and other non-training logs
        all_metrics.append(m)

    if args.list:
        print("{:<55} {:<12} {:>6}  {:>10}".format(
            "Job name", "Group", "Steps", "Val loss"))
        print("─" * 90)
        for r in all_metrics:
            vl = "{:.4f}".format(r.final_val_loss) if r.final_val_loss else "—"
            print("{:<55} {:<12} {:>6}  {:>10}".format(
                r.job_name, r.group, r.total_steps, vl))
        return

    if not all_metrics:
        print("No logs found for group '{}' (or none have started yet).".format(args.group))
        sys.exit(0)

    # Write per-run JSON (full iteration data)
    for m in all_metrics:
        out_json = results_dir / "{}.json".format(m.job_name)
        out_json.write_text(json.dumps(m.to_dict(), indent=2))

    # Write summary CSV
    summary_csv = results_dir / "summary.csv"
    with open(str(summary_csv), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_name", "group", "total_steps",
                    "avg_tokens_per_sec_gpu", "final_train_loss", "final_val_loss",
                    "n_val_evals"])
        for m in all_metrics:
            w.writerow([
                m.job_name, m.group, m.total_steps,
                "{:.1f}".format(m.avg_tokens_per_sec_gpu) if m.avg_tokens_per_sec_gpu else "",
                "{:.6f}".format(m.final_train_loss)       if m.final_train_loss       else "",
                "{:.6f}".format(m.final_val_loss)         if m.final_val_loss         else "",
                len(m.val_loss),
            ])
    print("Summary CSV: {}".format(summary_csv))

    g = args.group
    if g in ("throughput", "all"):
        print_throughput_table(all_metrics)
    if g in ("lr", "all"):
        print_lr_table(all_metrics)
    if g in ("batchsize", "all"):
        print_batchsize_table(all_metrics)
    if g in ("final", "all"):
        final = [r for r in all_metrics if r.group == "final"]
        if final:
            print("\nFINAL RUN")
            for r in final:
                print("  {}: val_loss={}, steps={}".format(
                    r.job_name, r.final_val_loss, r.total_steps))

    if args.no_plots:
        return

    try:
        plt = _setup_mpl()
    except ImportError:
        print("matplotlib not available — skipping plots "
              "(install with: pip install matplotlib)")
        return

    if g in ("throughput", "all"):
        plot_throughput(all_metrics, plots_dir, plt)
    if g in ("lr", "all"):
        plot_lr(all_metrics, plots_dir, plt)
    if g in ("batchsize", "all"):
        plot_batchsize(all_metrics, plots_dir, plt)
    if g in ("final", "all"):
        plot_final(all_metrics, plots_dir, plt)

    if not args.no_plots:
        print("\nAll plots saved to: {}/".format(plots_dir))


if __name__ == "__main__":
    main()
