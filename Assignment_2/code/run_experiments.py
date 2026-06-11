# -*- coding: utf-8 -*-
"""과제 (1)(2)(3) 전체 실험 오케스트레이터

Stage 1: optimizer 별 learning rate 탐색 (10 epochs, seed 42, scheduler 없음)
         → 선택 기준: 마지막 3개 epoch 의 평균 validation accuracy
Stage 2: 각 optimizer 를 최적 LR 로 20 epochs × 3 seeds 학습 → optimizer 비교
         → 최고 optimizer 선택 기준: seed 평균 final validation accuracy
Stage 3: 최고 optimizer 에 대해 6종 scheduler × 3 seeds 비교
         → 최고 scheduler 선택 기준: seed 평균 final validation accuracy
Stage 4: 최고 optimizer+scheduler 로 5-Fold 앙상블 학습/평가/추론

- 모든 선택은 validation 성능 기준 (test 성능은 보고용으로만 사용)
- 각 실행 로그(JSON)가 이미 존재하면 건너뛰므로 중단 후 재실행 가능
- 동시에 최대 PARALLEL 개의 학습 프로세스를 실행

실행:  python run_experiments.py
"""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "results"))
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CONSOLE_DIR = os.path.join(LOGS_DIR, "console")

# ----- 공통 하이퍼파라미터 -----
BATCH_SIZE = 128
SWEEP_EPOCHS = 10
MAIN_EPOCHS = 20
SEEDS = [42, 123, 2025]
PARALLEL = 3  # 동시 학습 프로세스 수

# optimizer 별 LR 탐색 그리드 (통상적 권장 범위)
# 최초 3개 후보에서 최적 LR 이 그리드 경계였던 optimizer(sgd, adagrad, adamw)는
# 더 큰/작은 LR 을 추가하여 최적값이 그리드 내부에 오도록 확장함
LR_GRID = {
    "sgd":          [3.0, 1.0, 0.3, 0.1, 0.03, 0.01],
    "sgd_momentum": [0.1, 0.03, 0.01],
    "sgd_nesterov": [0.1, 0.03, 0.01],
    "adagrad":      [0.3, 0.1, 0.03, 0.01, 0.003],
    "rmsprop":      [0.003, 0.001, 0.0003],
    "adam":         [0.003, 0.001, 0.0003],
    "adamw":        [0.01, 0.003, 0.001, 0.0003],
}
SCHEDULERS = ["none", "step", "exponential", "cosine", "plateau", "onecycle"]


def run_one(cmd, console_path):
    """학습 프로세스 1개 실행, 콘솔 출력은 파일로 저장."""
    os.makedirs(os.path.dirname(console_path), exist_ok=True)
    with open(console_path, "w") as out:
        result = subprocess.run(cmd, cwd=CODE_DIR, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        with open(console_path) as f:
            tail = "".join(f.readlines()[-15:])
        raise RuntimeError(f"FAILED: {' '.join(cmd)}\n--- console tail ---\n{tail}")
    return cmd


def train_cmd(optimizer, scheduler, lr, epochs, seed, log_path):
    return [
        sys.executable, "train.py",
        "--optimizer", optimizer,
        "--scheduler", scheduler,
        "--lr", str(lr),
        "--batch_size", str(BATCH_SIZE),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--log_path", log_path,
    ]


def run_pool(jobs):
    """jobs: [(cmd, console_path, log_path)] — 로그가 없는 작업만 병렬 실행."""
    todo = [(c, p) for c, p, log in jobs if not os.path.isfile(log)]
    done = len(jobs) - len(todo)
    if done:
        print(f"  ({done} runs already done, skipping)")
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(run_one, c, p): c for c, p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()  # 실패 시 예외 전파
            print(f"  [{i}/{len(todo)}] done: {' '.join(futures[fut][2:8])}", flush=True)


def read_log(path):
    with open(path) as f:
        return json.load(f)


def invalidate_stale_log(log_path, **expected):
    """기존 로그의 config 가 현재 실험 설정과 다르면 삭제 (그리드 변경 후 재실행 안전장치)."""
    if not os.path.isfile(log_path):
        return
    try:
        cfg = read_log(log_path)["config"]
    except (json.JSONDecodeError, KeyError):
        os.remove(log_path)
        return
    for key, value in expected.items():
        if cfg.get(key) != value:
            print(f"  (stale log removed: {os.path.basename(log_path)} — "
                  f"{key}={cfg.get(key)} != {value})")
            os.remove(log_path)
            return


def tail_mean(values, n=3):
    return sum(values[-n:]) / len(values[-n:])


# ---------------------------------------------------------------- Stage 1
def stage1_lr_sweep():
    print("\n===== Stage 1: Learning-rate sweep =====")
    jobs = []
    for opt, lrs in LR_GRID.items():
        for lr in lrs:
            name = f"{opt}_lr{lr}"
            log = os.path.join(LOGS_DIR, "lr_sweep", f"{name}.json")
            jobs.append((train_cmd(opt, "none", lr, SWEEP_EPOCHS, 42, log),
                         os.path.join(CONSOLE_DIR, f"sweep_{name}.txt"), log))
    run_pool(jobs)

    best_lr = {}
    for opt, lrs in LR_GRID.items():
        scores = {}
        for lr in lrs:
            log = read_log(os.path.join(LOGS_DIR, "lr_sweep", f"{opt}_lr{lr}.json"))
            scores[lr] = tail_mean(log["history"]["val_acc"])
        lr_star = max(scores, key=scores.get)
        best_lr[opt] = {"lr": lr_star, "val_acc_tail3": scores[lr_star], "all": scores}
        print(f"  {opt:13s} -> best lr {lr_star} (val acc {scores[lr_star]:.2f}%)")
    return best_lr


# ---------------------------------------------------------------- Stage 2
def stage2_optimizers(best_lr):
    print("\n===== Stage 2: Optimizer comparison (3 seeds) =====")
    jobs = []
    for opt in LR_GRID:
        for seed in SEEDS:
            log = os.path.join(LOGS_DIR, "optimizers", f"{opt}_s{seed}.json")
            invalidate_stale_log(log, optimizer=opt, lr=best_lr[opt]["lr"])
            jobs.append((train_cmd(opt, "none", best_lr[opt]["lr"], MAIN_EPOCHS, seed, log),
                         os.path.join(CONSOLE_DIR, f"opt_{opt}_s{seed}.txt"), log))
    run_pool(jobs)

    summary = {}
    for opt in LR_GRID:
        logs = [read_log(os.path.join(LOGS_DIR, "optimizers", f"{opt}_s{s}.json")) for s in SEEDS]
        final_val = [l["final_val_acc"] for l in logs]
        summary[opt] = {
            "lr": best_lr[opt]["lr"],
            "mean_final_val_acc": sum(final_val) / len(final_val),
            "mean_test_acc": sum(l["test_acc"] for l in logs) / len(logs),
        }
        print(f"  {opt:13s} val {summary[opt]['mean_final_val_acc']:.2f}% | "
              f"test {summary[opt]['mean_test_acc']:.2f}%")
    best_opt = max(summary, key=lambda o: summary[o]["mean_final_val_acc"])
    print(f"  ==> Best optimizer: {best_opt}")
    return best_opt, summary


# ---------------------------------------------------------------- Stage 3
def stage3_schedulers(best_opt, lr):
    print(f"\n===== Stage 3: Scheduler comparison on {best_opt} (3 seeds) =====")
    jobs = []
    for sched in SCHEDULERS:
        for seed in SEEDS:
            log = os.path.join(LOGS_DIR, "schedulers", f"{sched}_s{seed}.json")
            invalidate_stale_log(log, optimizer=best_opt, scheduler=sched, lr=lr)
            if sched == "none" and not os.path.isfile(log):
                # 동일 설정의 Stage 2 로그 재사용 (같은 seed/조건 → 같은 결과)
                src = os.path.join(LOGS_DIR, "optimizers", f"{best_opt}_s{seed}.json")
                if os.path.isfile(src):
                    os.makedirs(os.path.dirname(log), exist_ok=True)
                    shutil.copy(src, log)
                    continue
            jobs.append((train_cmd(best_opt, sched, lr, MAIN_EPOCHS, seed, log),
                         os.path.join(CONSOLE_DIR, f"sched_{sched}_s{seed}.txt"), log))
    run_pool(jobs)

    summary = {}
    for sched in SCHEDULERS:
        logs = [read_log(os.path.join(LOGS_DIR, "schedulers", f"{sched}_s{s}.json")) for s in SEEDS]
        final_val = [l["final_val_acc"] for l in logs]
        summary[sched] = {
            "mean_final_val_acc": sum(final_val) / len(final_val),
            "mean_test_acc": sum(l["test_acc"] for l in logs) / len(logs),
        }
        print(f"  {sched:12s} val {summary[sched]['mean_final_val_acc']:.2f}% | "
              f"test {summary[sched]['mean_test_acc']:.2f}%")
    best_sched = max(summary, key=lambda s: summary[s]["mean_final_val_acc"])
    print(f"  ==> Best scheduler: {best_sched}")
    return best_sched, summary


# ---------------------------------------------------------------- Stage 4
def stage4_ensemble(best_opt, lr, best_sched):
    print(f"\n===== Stage 4: 5-Fold ensemble ({best_opt} + {best_sched}) =====")
    prefix = os.path.join(RESULTS_DIR, "models", "ensemble", "mlp_mnist")
    log_dir = os.path.join(LOGS_DIR, "ensemble")
    ens_test_log = os.path.join(log_dir, "ensemble_test.json")

    for fold in range(5):
        invalidate_stale_log(os.path.join(log_dir, f"fold{fold}.json"),
                             optimizer=best_opt, scheduler=best_sched, lr=lr)

    if not all(os.path.isfile(os.path.join(log_dir, f"fold{f}.json")) for f in range(5)):
        run_one(
            [sys.executable, "train_ensemble.py",
             "--optimizer", best_opt, "--scheduler", best_sched, "--lr", str(lr),
             "--batch_size", str(BATCH_SIZE), "--epochs", str(MAIN_EPOCHS),
             "--k_splits", "5", "--save_prefix", prefix, "--log_dir", log_dir],
            os.path.join(CONSOLE_DIR, "ensemble_train.txt"),
        )
    else:
        print("  (ensemble training already done, skipping)")

    run_one(
        [sys.executable, "test_ensemble.py",
         "--model_prefix", prefix, "--k_splits", "5", "--log_path", ens_test_log],
        os.path.join(CONSOLE_DIR, "ensemble_test.txt"),
    )
    run_one(
        [sys.executable, "inference.py",
         "--model_prefix", prefix, "--k_splits", "5", "--sample_idx", "11",
         "--fig_path", os.path.join(RESULTS_DIR, "figures", "inference_sample.png")],
        os.path.join(CONSOLE_DIR, "ensemble_inference.txt"),
    )
    return read_log(ens_test_log)


# ---------------------------------------------------------------- Main
def main():
    os.makedirs(CONSOLE_DIR, exist_ok=True)

    best_lr = stage1_lr_sweep()
    best_opt, opt_summary = stage2_optimizers(best_lr)
    best_sched, sched_summary = stage3_schedulers(best_opt, best_lr[best_opt]["lr"])
    ensemble_result = stage4_ensemble(best_opt, best_lr[best_opt]["lr"], best_sched)

    decisions = {
        "common": {"batch_size": BATCH_SIZE, "sweep_epochs": SWEEP_EPOCHS,
                   "main_epochs": MAIN_EPOCHS, "seeds": SEEDS,
                   "val_portion": 0.1, "model": "MLP 784-256-256-10 (BN, Dropout 0.2)"},
        "stage1_best_lr": best_lr,
        "stage2_optimizer_summary": opt_summary,
        "stage2_best_optimizer": best_opt,
        "stage3_scheduler_summary": sched_summary,
        "stage3_best_scheduler": best_sched,
        "stage4_ensemble": {
            k: ensemble_result[k]
            for k in ["fold_accuracies", "single_mean_acc", "single_max_acc",
                      "hard_voting_acc", "soft_voting_acc", "mcnemar_vs_best_fold"]
        },
    }
    decisions_path = os.path.join(RESULTS_DIR, "decisions.json")
    with open(decisions_path, "w") as f:
        json.dump(decisions, f, indent=2)
    print(f"\nAll experiments finished. Decisions saved to {decisions_path}")


if __name__ == "__main__":
    main()
