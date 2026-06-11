"""실험 결과 시각화 (headless 환경: Agg 백엔드, savefig만 사용).

run_experiments.py에서 호출되며, 저장된 JSON만으로 단독 재실행도 가능:
    python plotting.py --part 1
    python plotting.py --part 2
"""
import argparse
import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import load_json

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "results"))
FIGURES_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "figures"))


def load_runs(results_subdir):
    """{설정 이름: [run dict (seed별)]} 형태로 결과 JSON을 모은다."""
    runs_by_name = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_subdir, "*_seed*.json"))):
        run = load_json(path)
        name = run["name"]
        runs_by_name[name].append(run)
    return dict(runs_by_name)


def plot_metric(runs_by_name, metric, ylabel, title, out_path, log_scale=False):
    plt.figure(figsize=(7.5, 5))
    for name, runs in runs_by_name.items():
        epochs = [h["epoch"] for h in runs[0]["history"]]
        values = np.mean([[h[metric] for h in r["history"]] for r in runs], axis=0)
        plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.5, label=name)
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_lr_schedules(runs_by_name, out_path):
    """seed 0 run의 lr_samples를 이어붙여 실제 LR 궤적을 그린다."""
    plt.figure(figsize=(7.5, 5))
    for name, runs in runs_by_name.items():
        run = runs[0]
        xs, ys = [], []
        for h in run["history"]:
            samples = h["lr_samples"]
            for i, lr in enumerate(samples):
                xs.append(h["epoch"] - 1 + i / len(samples))
                ys.append(lr)
        plt.plot(xs, ys, linewidth=1.5, label=name)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedules")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_part(part: int, results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR):
    subdir = os.path.join(results_dir, f"part{part}")
    runs_by_name = load_runs(subdir)
    if not runs_by_name:
        print(f"[warn] no results in {subdir}")
        return
    label = "Optimizer" if part == 1 else "Scheduler"
    prefix = f"part{part}"
    plot_metric(runs_by_name, "train_loss", "Train Loss",
                f"{label} Comparison: Train Loss", os.path.join(figures_dir, f"{prefix}_train_loss.png"),
                log_scale=True)
    plot_metric(runs_by_name, "val_loss", "Validation Loss",
                f"{label} Comparison: Validation Loss", os.path.join(figures_dir, f"{prefix}_val_loss.png"),
                log_scale=True)
    plot_metric(runs_by_name, "val_acc", "Validation Accuracy (%)",
                f"{label} Comparison: Validation Accuracy", os.path.join(figures_dir, f"{prefix}_val_acc.png"))
    if part == 2:
        plot_lr_schedules(runs_by_name, os.path.join(figures_dir, "part2_lr_schedule.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="결과 JSON으로부터 비교 곡선 재생성")
    parser.add_argument("--part", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    plot_part(args.part)
