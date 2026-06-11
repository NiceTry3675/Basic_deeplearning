# -*- coding: utf-8 -*-
"""실험 로그(JSON)로부터 보고서용 그림과 표(markdown/CSV)를 생성한다.

실행:  python plot_results.py   (run_experiments.py 완료 후)
출력:  ../results/figures/*.png, ../results/tables/*.md
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "results"))
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")

SEEDS = [42, 123, 2025]
OPTIMIZERS = ["sgd", "sgd_momentum", "sgd_nesterov", "adagrad", "rmsprop", "adam", "adamw"]
SCHEDULERS = ["none", "step", "exponential", "cosine", "plateau", "onecycle"]

OPT_LABELS = {
    "sgd": "SGD", "sgd_momentum": "SGD+Momentum", "sgd_nesterov": "SGD+Nesterov",
    "adagrad": "Adagrad", "rmsprop": "RMSprop", "adam": "Adam", "adamw": "AdamW",
}
SCHED_LABELS = {
    "none": "None (constant)", "step": "StepLR", "exponential": "ExponentialLR",
    "cosine": "CosineAnnealing", "plateau": "ReduceLROnPlateau", "onecycle": "OneCycleLR",
}
COLORS = plt.cm.tab10.colors


def read_log(*parts):
    with open(os.path.join(LOGS_DIR, *parts)) as f:
        return json.load(f)


def load_group(subdir, names, fmt):
    """{name: [seed별 log]} 로 로드."""
    return {n: [read_log(subdir, fmt.format(n=n, s=s)) for s in SEEDS] for n in names}


def mean_curve(logs, key):
    arr = np.array([l["history"][key] for l in logs])
    return arr.mean(axis=0), arr.std(axis=0)


def epochs_to_reach(history_vals, threshold, mode="ge"):
    """threshold 도달 첫 epoch (1-base). 미도달 시 None."""
    for i, v in enumerate(history_vals):
        if (mode == "ge" and v >= threshold) or (mode == "le" and v <= threshold):
            return i + 1
    return None


def mean_epochs_to_reach(logs, key, threshold, mode="ge"):
    """seed 별 도달 epoch 평균. 한 seed 라도 미도달이면 '>E' 표기."""
    epochs_list = [epochs_to_reach(l["history"][key], threshold, mode) for l in logs]
    n_epochs = len(logs[0]["history"][key])
    if any(e is None for e in epochs_list):
        return None, f">{n_epochs}"
    m = float(np.mean(epochs_list))
    return m, f"{m:.1f}"


def curve_figure(group, labels, key, ylabel, title, path, logy=False, band=True):
    plt.figure(figsize=(8, 5))
    for i, (name, logs) in enumerate(group.items()):
        m, s = mean_curve(logs, key)
        epochs = np.arange(1, len(m) + 1)
        plt.plot(epochs, m, label=labels[name], color=COLORS[i % 10], linewidth=1.8)
        if band:
            plt.fill_between(epochs, m - s, m + s, color=COLORS[i % 10], alpha=0.15)
    if logy:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def bar_figure(group, labels, title, path, value_key="test_acc"):
    names = list(group.keys())
    means = [np.mean([l[value_key] for l in group[n]]) for n in names]
    stds = [np.std([l[value_key] for l in group[n]]) for n in names]
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(names))
    bars = plt.bar(x, means, yerr=stds, capsize=4,
                   color=[COLORS[i % 10] for i in range(len(names))], alpha=0.85)
    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f"{m:.2f}", ha="center", fontsize=9)
    plt.xticks(x, [labels[n] for n in names], rotation=20, ha="right")
    plt.ylabel("Test accuracy (%)")
    lo = min(m - s for m, s in zip(means, stds))
    plt.ylim(min(lo - 0.3, 97.0), max(means) + 0.45)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------- Stage 1
def figure_lr_sweep(decisions):
    """optimizer 별로 탐색한 LR(가변 개수)의 val acc 를 그룹 막대로 표시."""
    best_lr = decisions["stage1_best_lr"]
    plt.figure(figsize=(10, 5))
    width = 0.2
    x = np.arange(len(OPTIMIZERS))
    for i, opt in enumerate(OPTIMIZERS):
        entries = sorted(best_lr[opt]["all"].items(), key=lambda kv: -float(kv[0]))
        n = len(entries)
        for j, (lr, acc) in enumerate(entries):
            selected = float(lr) == best_lr[opt]["lr"]
            b = plt.bar(i + (j - (n - 1) / 2) * width, acc, width, alpha=0.9,
                        color="darkorange" if selected else "steelblue",
                        edgecolor="black" if selected else "none", linewidth=0.8)
            plt.text(b[0].get_x() + width / 2, acc + 0.05, f"{float(lr):g}",
                     ha="center", fontsize=7, rotation=90)
    plt.xticks(x, [OPT_LABELS[o] for o in OPTIMIZERS], rotation=20, ha="right")
    plt.ylabel("Val accuracy, mean of last 3 epochs (%)")
    plt.ylim(90, 99.5)
    plt.title("Stage 1: LR sweep (10 epochs, seed 42; orange = selected LR)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "stage1_lr_sweep.png"), dpi=150)
    plt.close()


def table_lr_sweep(decisions):
    best_lr = decisions["stage1_best_lr"]
    lines = ["| Optimizer | 탐색한 LR (val acc %) | 선택된 LR |", "|---|---|---|"]
    for opt in OPTIMIZERS:
        entries = sorted(best_lr[opt]["all"].items(), key=lambda kv: -float(kv[0]))
        cell = ", ".join(f"{float(lr):g} ({acc:.2f})" for lr, acc in entries)
        lines.append(f"| {OPT_LABELS[opt]} | {cell} | **{best_lr[opt]['lr']:g}** |")
    return "\n".join(lines)


# ---------------------------------------------------------------- Stage 2/3 표
def comparison_table(group, labels, lr_info=None):
    header = ("| 항목 | LR | Final Train Loss | Final Val Loss | Final Val Acc (%) "
              "| Test Acc (%) | Val Acc 98% 도달 epoch | epoch당 시간(s) |")
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for name, logs in group.items():
        tl = np.mean([l["history"]["train_loss"][-1] for l in logs])
        vl = np.mean([l["history"]["val_loss"][-1] for l in logs])
        va_m = np.mean([l["final_val_acc"] for l in logs])
        va_s = np.std([l["final_val_acc"] for l in logs])
        ta_m = np.mean([l["test_acc"] for l in logs])
        ta_s = np.std([l["test_acc"] for l in logs])
        _, e98 = mean_epochs_to_reach(logs, "val_acc", 98.0)
        et = np.mean([np.mean(l["history"]["epoch_time"]) for l in logs])
        lr = f"{logs[0]['config']['lr']:g}" if lr_info is None else lr_info
        lines.append(f"| {labels[name]} | {lr} | {tl:.4f} | {vl:.4f} "
                     f"| {va_m:.2f} ± {va_s:.2f} | {ta_m:.2f} ± {ta_s:.2f} | {e98} | {et:.1f} |")
    return "\n".join(lines)


def convergence_table(group, labels):
    lines = ["| 항목 | Val Acc 97% 도달 | Val Acc 98% 도달 | Train Loss 0.1 도달 | Train Loss 0.05 도달 |",
             "|---|---|---|---|---|"]
    for name, logs in group.items():
        _, a97 = mean_epochs_to_reach(logs, "val_acc", 97.0)
        _, a98 = mean_epochs_to_reach(logs, "val_acc", 98.0)
        _, l10 = mean_epochs_to_reach(logs, "train_loss", 0.10, mode="le")
        _, l05 = mean_epochs_to_reach(logs, "train_loss", 0.05, mode="le")
        lines.append(f"| {labels[name]} | {a97} | {a98} | {l10} | {l05} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- Stage 3 LR 곡선
def figure_lr_schedule(group):
    plt.figure(figsize=(8, 5))
    for i, (name, logs) in enumerate(group.items()):
        lrs = logs[0]["history"]["lr"]
        plt.plot(np.arange(1, len(lrs) + 1), lrs, label=SCHED_LABELS[name],
                 color=COLORS[i % 10], linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate (at epoch start)")
    plt.yscale("log")
    plt.title("Stage 3: LR schedule per scheduler")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "stage3_lr_schedule.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------- Stage 4
def figure_ensemble(ens):
    folds = ens["fold_accuracies"]
    names = [f"Fold {i}" for i in range(len(folds))] + \
            ["Single mean", "Hard voting", "Soft voting"]
    vals = folds + [ens["single_mean_acc"], ens["hard_voting_acc"], ens["soft_voting_acc"]]
    colors = ["lightsteelblue"] * len(folds) + ["gray", "darkorange", "crimson"]
    plt.figure(figsize=(8.5, 4.5))
    bars = plt.bar(names, vals, color=colors, alpha=0.9)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.2f}", ha="center", fontsize=9)
    plt.ylabel("Test accuracy (%)")
    plt.ylim(min(vals) - 0.3, max(vals) + 0.25)
    plt.title("Stage 4: 5-Fold ensemble vs single models")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "stage4_ensemble_acc.png"), dpi=150)
    plt.close()


def figure_confusion(ens):
    cm = np.array(ens["confusion_matrix_soft"])
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(np.log1p(cm), cmap="Blues")
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                         color="white" if i == j else "black")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (soft-voting ensemble)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "stage4_confusion.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------- Main
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "decisions.json")) as f:
        decisions = json.load(f)

    # ----- Stage 1 -----
    figure_lr_sweep(decisions)

    # ----- Stage 2 -----
    opt_group = load_group("optimizers", OPTIMIZERS, "{n}_s{s}.json")
    curve_figure(opt_group, OPT_LABELS, "train_loss", "Train loss (log scale)",
                 "Stage 2: Training loss (mean of 3 seeds)",
                 os.path.join(FIG_DIR, "stage2_train_loss.png"), logy=True)
    curve_figure(opt_group, OPT_LABELS, "val_loss", "Validation loss",
                 "Stage 2: Validation loss (mean of 3 seeds)",
                 os.path.join(FIG_DIR, "stage2_val_loss.png"))
    curve_figure(opt_group, OPT_LABELS, "val_acc", "Validation accuracy (%)",
                 "Stage 2: Validation accuracy (mean of 3 seeds)",
                 os.path.join(FIG_DIR, "stage2_val_acc.png"))
    bar_figure(opt_group, OPT_LABELS, "Stage 2: Test accuracy by optimizer (mean ± std, 3 seeds)",
               os.path.join(FIG_DIR, "stage2_test_acc.png"))

    # ----- Stage 3 -----
    sched_group = load_group("schedulers", SCHEDULERS, "{n}_s{s}.json")
    curve_figure(sched_group, SCHED_LABELS, "train_loss", "Train loss (log scale)",
                 "Stage 3: Training loss (mean of 3 seeds)",
                 os.path.join(FIG_DIR, "stage3_train_loss.png"), logy=True)
    curve_figure(sched_group, SCHED_LABELS, "val_loss", "Validation loss",
                 "Stage 3: Validation loss (mean of 3 seeds)",
                 os.path.join(FIG_DIR, "stage3_val_loss.png"))
    curve_figure(sched_group, SCHED_LABELS, "val_acc", "Validation accuracy (%)",
                 "Stage 3: Validation accuracy (mean of 3 seeds)",
                 os.path.join(FIG_DIR, "stage3_val_acc.png"))
    bar_figure(sched_group, SCHED_LABELS, "Stage 3: Test accuracy by scheduler (mean ± std, 3 seeds)",
               os.path.join(FIG_DIR, "stage3_test_acc.png"))
    figure_lr_schedule(sched_group)

    # ----- Stage 4 -----
    ens = read_log("ensemble", "ensemble_test.json")
    figure_ensemble(ens)
    figure_confusion(ens)

    # ----- 표 (markdown) -----
    md = []
    md.append("## Stage 1: LR sweep\n\n" + table_lr_sweep(decisions))
    md.append("## Stage 2: Optimizer comparison\n\n" + comparison_table(opt_group, OPT_LABELS))
    md.append("### Stage 2: Convergence speed\n\n" + convergence_table(opt_group, OPT_LABELS))
    md.append("## Stage 3: Scheduler comparison\n\n" + comparison_table(sched_group, SCHED_LABELS))
    md.append("### Stage 3: Convergence speed\n\n" + convergence_table(sched_group, SCHED_LABELS))
    ens_md = (
        "## Stage 4: Ensemble\n\n"
        "| 구분 | Test Acc (%) |\n|---|---|\n"
        + "".join(f"| Fold {i} 단일 모델 | {a:.2f} |\n" for i, a in enumerate(ens["fold_accuracies"]))
        + f"| 단일 모델 평균 | {ens['single_mean_acc']:.2f} |\n"
        + f"| 단일 모델 최고 | {ens['single_max_acc']:.2f} |\n"
        + f"| **앙상블 (Hard Voting)** | **{ens['hard_voting_acc']:.2f}** |\n"
        + f"| **앙상블 (Soft Voting)** | **{ens['soft_voting_acc']:.2f}** |"
    )
    md.append(ens_md)

    with open(os.path.join(TABLE_DIR, "summary_tables.md"), "w") as f:
        f.write("\n\n".join(md) + "\n")

    print(f"Figures  -> {FIG_DIR}")
    print(f"Tables   -> {os.path.join(TABLE_DIR, 'summary_tables.md')}")


if __name__ == "__main__":
    main()
