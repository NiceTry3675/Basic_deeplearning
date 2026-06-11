"""Part 1(optimizer 비교)·Part 2(scheduler 비교) 실험 러너.

사용법:
    # Part 1-1: optimizer별 LR 탐색 (5 epochs, seed 0)
    python run_experiments.py --part 1 --stage sweep
    # Part 1-2: 각 optimizer의 최적 LR로 본 비교 (15 epochs, 3 seeds)
    python run_experiments.py --part 1 --stage main
    # Part 2: 우승 optimizer에 scheduler 7종 적용 (15 epochs, 3 seeds)
    python run_experiments.py --part 2 --optimizer Adam --lr 1e-3
    # 저장된 JSON으로 표/플롯만 재생성
    python run_experiments.py --part 1 --stage main --summary-only

출력: results/part*/ 아래 run별 JSON + summary.csv + summary.md(보고서용 표),
      figures/ 아래 비교 곡선 PNG.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import plotting
from configs import COMMON, LR_GRID, OPTIMIZERS, SCHEDULER_NAMES, make_scheduler
from data_loader import load_mnist
from engine import train_model
from model import MLP
from utils import get_device, load_json, save_json, set_seed

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "results"))

# 수렴 속도 지표의 임계값 (summary 시점에 history로부터 계산 — 재학습 없이 조정 가능)
# 97%/0.1은 모든 optimizer가 1~2 epoch에 도달해 변별력이 없어 98%/0.05로 상향
VAL_ACC_TARGET = 98.0
TRAIN_LOSS_TARGET = 0.05


def run_one(name, optimizer_name, lr, scheduler_name, seed, epochs, device):
    """단일 학습 run을 수행하고 결과 dict를 반환한다."""
    set_seed(seed)
    train_ds, val_ds, _, in_dim, _, _ = load_mnist(
        valDB_portion=COMMON["val_portion"], split_seed=COMMON["split_seed"])
    train_loader = DataLoader(train_ds, batch_size=COMMON["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = MLP(in_dim).to(device)
    criterion = CrossEntropyLoss()
    optimizer = OPTIMIZERS[optimizer_name](model.parameters(), lr)
    scheduler, mode = make_scheduler(scheduler_name, optimizer, lr, epochs, len(train_loader))

    print(f"\n===== {name} (seed {seed}) =====", flush=True)
    history = train_model(model, train_loader, val_loader, criterion, optimizer,
                          scheduler=scheduler, scheduler_mode=mode,
                          epochs=epochs, device=device)
    return {
        "name": name,
        "config": {"optimizer": optimizer_name, "lr": lr, "scheduler": scheduler_name,
                   "seed": seed, "epochs": epochs, "batch_size": COMMON["batch_size"],
                   "val_portion": COMMON["val_portion"], "split_seed": COMMON["split_seed"]},
        "history": history,
    }


def first_epoch_reaching(history, key, target, direction):
    """지표가 임계값에 처음 도달한 epoch (미도달 시 None)."""
    for h in history:
        if direction == "ge" and h[key] >= target:
            return h["epoch"]
        if direction == "le" and h[key] <= target:
            return h["epoch"]
    return None


def run_metrics(run):
    hist = run["history"]
    epoch_times = [h["epoch_time_sec"] for h in hist[1:]] or [hist[0]["epoch_time_sec"]]
    return {
        "final_val_acc": hist[-1]["val_acc"],
        "best_val_acc": max(h["val_acc"] for h in hist),
        "final_val_loss": hist[-1]["val_loss"],
        "final_train_loss": hist[-1]["train_loss"],
        "epochs_to_val_acc": first_epoch_reaching(hist, "val_acc", VAL_ACC_TARGET, "ge"),
        "epochs_to_train_loss": first_epoch_reaching(hist, "train_loss", TRAIN_LOSS_TARGET, "le"),
        "sec_per_epoch": float(np.mean(epoch_times)),  # 1 epoch 워밍업 제외
    }


def fmt_reach(values):
    """도달 epoch 목록 → '평균 (도달seed수/전체)' 형식. 예: '4.3 (3/3)', '- (0/3)'"""
    reached = [v for v in values if v is not None]
    if not reached:
        return f"- (0/{len(values)})"
    return f"{np.mean(reached):.1f} ({len(reached)}/{len(values)})"


def write_summary(subdir, group_key_label, out_name="summary"):
    """results/<subdir>의 run JSON들을 설정 이름별로 집계해 CSV/MD 표를 만든다."""
    runs_by_name = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(subdir, "*_seed*.json"))):
        run = load_json(path)
        runs_by_name[run["name"]].append(run)

    rows = []
    for name, runs in runs_by_name.items():
        metrics = [run_metrics(r) for r in runs]
        cfg = runs[0]["config"]
        rows.append({
            "name": name,
            "lr": cfg["lr"],
            "n_seeds": len(runs),
            "final_val_acc_mean": np.mean([m["final_val_acc"] for m in metrics]),
            "final_val_acc_std": np.std([m["final_val_acc"] for m in metrics]),
            "best_val_acc_mean": np.mean([m["best_val_acc"] for m in metrics]),
            "best_val_acc_std": np.std([m["best_val_acc"] for m in metrics]),
            "final_val_loss_mean": np.mean([m["final_val_loss"] for m in metrics]),
            "epochs_to_val_acc": fmt_reach([m["epochs_to_val_acc"] for m in metrics]),
            "epochs_to_train_loss": fmt_reach([m["epochs_to_train_loss"] for m in metrics]),
            "sec_per_epoch": np.mean([m["sec_per_epoch"] for m in metrics]),
        })
    rows.sort(key=lambda r: -r["final_val_acc_mean"])

    csv_path = os.path.join(subdir, f"{out_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        f"| {group_key_label} | LR | Final Val Acc (%) | Best Val Acc (%) | Final Val Loss "
        f"| Val Acc≥{VAL_ACC_TARGET}% 도달 epoch | Train Loss≤{TRAIN_LOSS_TARGET} 도달 epoch | sec/epoch |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['name']} | {r['lr']:g} "
            f"| {r['final_val_acc_mean']:.2f} ± {r['final_val_acc_std']:.2f} "
            f"| {r['best_val_acc_mean']:.2f} ± {r['best_val_acc_std']:.2f} "
            f"| {r['final_val_loss_mean']:.4f} "
            f"| {r['epochs_to_val_acc']} | {r['epochs_to_train_loss']} "
            f"| {r['sec_per_epoch']:.1f} |")
    md_path = os.path.join(subdir, f"{out_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nsaved: {csv_path}\nsaved: {md_path}\n")
    print("\n".join(md_lines))
    return rows


def write_sweep_summary(subdir):
    """LR 탐색 결과를 optimizer×LR 표로 집계하고 최적 LR을 저장한다."""
    runs = [load_json(p) for p in sorted(glob.glob(os.path.join(subdir, "*_seed*.json")))]
    by_opt = defaultdict(list)
    for run in runs:
        by_opt[run["config"]["optimizer"]].append(run)

    best_lr = {}
    md_lines = ["| Optimizer | LR | Best Val Acc (%) | Final Val Acc (%) | 선택 |",
                "|---|---|---|---|---|"]
    for opt in OPTIMIZERS:
        if opt not in by_opt:
            continue
        candidates = []
        for run in sorted(by_opt[opt], key=lambda r: -r["config"]["lr"]):
            m = run_metrics(run)
            candidates.append((run["config"]["lr"], m["best_val_acc"], m["final_val_acc"]))
        chosen = max(candidates, key=lambda c: c[1])[0]
        best_lr[opt] = chosen
        for lr, best, final in candidates:
            mark = "✓" if lr == chosen else ""
            md_lines.append(f"| {opt} | {lr:g} | {best:.2f} | {final:.2f} | {mark} |")

    save_json(best_lr, os.path.join(subdir, "best_lr.json"))
    md_path = os.path.join(subdir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nsaved: {md_path}")
    print("\n".join(md_lines))
    print(f"\nbest LR per optimizer: {best_lr}")
    return best_lr


def main():
    parser = argparse.ArgumentParser(description="MNIST optimizer/scheduler 비교 실험")
    parser.add_argument("--part", type=int, choices=[1, 2], required=True)
    parser.add_argument("--stage", type=str, choices=["sweep", "main"], default="main",
                        help="part 1 전용: sweep(LR 탐색) / main(본 비교)")
    parser.add_argument("--optimizer", type=str, default=None,
                        help="part 2 전용: part 1에서 선택된 optimizer 이름")
    parser.add_argument("--lr", type=float, default=None, help="part 2 전용: 선택된 LR")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quick", action="store_true",
                        help="스모크 테스트: 축소 구성으로 results/quick_* 에 기록")
    parser.add_argument("--summary-only", action="store_true",
                        help="학습 없이 저장된 JSON으로 표/플롯만 재생성")
    args = parser.parse_args()

    device = get_device(args.device)
    epochs = COMMON["epochs"]
    sweep_epochs = COMMON["sweep_epochs"]
    seeds = COMMON["seeds"]
    optimizers = list(OPTIMIZERS)
    schedulers = list(SCHEDULER_NAMES)
    results_root = RESULTS_DIR

    if args.quick:
        epochs, sweep_epochs, seeds = 2, 2, [0]
        optimizers = ["SGD", "Adam"]
        schedulers = ["None", "OneCycleLR"]
        results_root = os.path.join(RESULTS_DIR, "quick")

    if args.part == 1 and args.stage == "sweep":
        subdir = os.path.join(results_root, "part1_sweep")
        if not args.summary_only:
            for opt in optimizers:
                lrs = LR_GRID[opt][:1] if args.quick else LR_GRID[opt]
                for lr in lrs:
                    name = f"{opt}_lr{lr:g}"
                    run = run_one(name, opt, lr, "None", seed=0,
                                  epochs=sweep_epochs, device=device)
                    save_json(run, os.path.join(subdir, f"{name}_seed0.json"))
        write_sweep_summary(subdir)

    elif args.part == 1:
        sweep_dir = os.path.join(results_root, "part1_sweep")
        best_lr_path = os.path.join(sweep_dir, "best_lr.json")
        if not os.path.exists(best_lr_path):
            raise SystemExit("먼저 --stage sweep을 실행하세요 (best_lr.json 없음)")
        best_lr = load_json(best_lr_path)
        subdir = os.path.join(results_root, "part1")
        if not args.summary_only:
            for opt in optimizers:
                for seed in seeds:
                    run = run_one(opt, opt, best_lr[opt], "None", seed=seed,
                                  epochs=epochs, device=device)
                    save_json(run, os.path.join(subdir, f"{opt}_seed{seed}.json"))
        write_summary(subdir, "Optimizer")
        plotting.plot_part(1, results_root,
                           plotting.FIGURES_DIR if not args.quick
                           else os.path.join(plotting.FIGURES_DIR, "quick"))

    else:  # part 2
        if not args.optimizer or args.lr is None:
            raise SystemExit("part 2에는 --optimizer와 --lr이 필요합니다 (part 1의 우승 구성)")
        subdir = os.path.join(results_root, "part2")
        if not args.summary_only:
            for sched in schedulers:
                for seed in seeds:
                    run = run_one(sched, args.optimizer, args.lr, sched, seed=seed,
                                  epochs=epochs, device=device)
                    save_json(run, os.path.join(subdir, f"{sched}_seed{seed}.json"))
        write_summary(subdir, "Scheduler")
        plotting.plot_part(2, results_root,
                           plotting.FIGURES_DIR if not args.quick
                           else os.path.join(plotting.FIGURES_DIR, "quick"))


if __name__ == "__main__":
    main()
