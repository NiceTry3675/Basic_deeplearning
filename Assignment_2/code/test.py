"""Part 3: 테스트셋(10,000장) 앙상블 평가 (강의자료 pr_14 test_ensemble.py 기반).

비교 대상:
- Fold별 단일 모델 / K-Fold 앙상블 (hard·soft voting)
- 전체 데이터로 학습한 단일 모델(full-train, seed0)
- 시드 앙상블: full-train 모델 5개(시드만 다름)의 hard·soft voting
- Hard voting: 다수결, 동률 시 가장 작은 라벨 (강의 방식)
- Soft voting: 모델들의 softmax 확률 평균 후 argmax

사용법: python test.py
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import load_mnist
from model import MLP
from utils import get_device, save_json

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CODE_DIR, "models")
RESULTS_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "results", "part3"))
FIGURES_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "figures"))


def load_model(path, in_dim, device):
    model = MLP(in_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict_probs(model, loader, device):
    """테스트셋 전체에 대한 softmax 확률 (N, 10) 반환."""
    probs = []
    with torch.no_grad():
        for data, _ in loader:
            output = model(data.to(device))
            probs.append(F.softmax(output, dim=1).cpu())
    return torch.cat(probs).numpy()


def hard_voting(fold_preds, num_classes=10):
    """다수결 투표. 동률이면 가장 작은 라벨 선택 (강의자료 방식)."""
    votes = np.stack(fold_preds)                       # (k, N)
    counts = np.stack([(votes == c).sum(axis=0) for c in range(num_classes)], axis=1)
    return counts.argmax(axis=1)                       # argmax는 동률 시 최소 인덱스


def main(args):
    device = get_device(args.device)
    _, _, test_ds, in_dim, x_test_raw, y_test = load_mnist(valDB_portion=0.0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    y_true = np.asarray(y_test)

    # ----- Fold 모델별 예측 -----
    fold_probs, fold_preds, fold_accs = [], [], []
    for fold in range(args.k_splits):
        path = os.path.join(args.model_dir, f"mlp_fold{fold}.pth")
        probs = predict_probs(load_model(path, in_dim, device), test_loader, device)
        preds = probs.argmax(axis=1)
        acc = 100.0 * (preds == y_true).mean()
        fold_probs.append(probs)
        fold_preds.append(preds)
        fold_accs.append(acc)
        print(f"Fold {fold} Test Accuracy: {acc:.2f}%")

    # ----- K-Fold 앙상블 -----
    kf_hard_acc = 100.0 * (hard_voting(fold_preds) == y_true).mean()
    kf_soft_acc = 100.0 * (np.mean(fold_probs, axis=0).argmax(axis=1) == y_true).mean()

    # ----- Full-train 모델들 (seed0 = 단일 모델 baseline) -----
    full_probs, full_preds, full_accs = [], [], []
    for i in range(args.n_full):
        path = os.path.join(args.model_dir, f"mlp_full_seed{i}.pth")
        probs = predict_probs(load_model(path, in_dim, device), test_loader, device)
        preds = probs.argmax(axis=1)
        acc = 100.0 * (preds == y_true).mean()
        full_probs.append(probs)
        full_preds.append(preds)
        full_accs.append(acc)
        print(f"Full-train seed{i} Test Accuracy: {acc:.2f}%")
    single_acc = full_accs[0]

    # ----- 시드 앙상블 -----
    seed_hard_acc = 100.0 * (hard_voting(full_preds) == y_true).mean()
    seed_soft_preds = np.mean(full_probs, axis=0).argmax(axis=1)
    seed_soft_acc = 100.0 * (seed_soft_preds == y_true).mean()

    # ----- 통합 앙상블 (fold 5개 + full-train 5개, soft voting) -----
    all_soft_acc = 100.0 * (np.mean(fold_probs + full_probs, axis=0).argmax(axis=1) == y_true).mean()

    print("\n----- 테스트셋 결과 요약 -----")
    print(f"Fold 단일 모델 평균:            {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f} "
          f"(min {min(fold_accs):.2f} / max {max(fold_accs):.2f})")
    print(f"K-Fold 앙상블 (Hard voting):    {kf_hard_acc:.2f}%")
    print(f"K-Fold 앙상블 (Soft voting):    {kf_soft_acc:.2f}%")
    print(f"Full-train 단일 모델 (seed0):   {single_acc:.2f}%")
    print(f"Full-train 모델 평균:           {np.mean(full_accs):.2f}% ± {np.std(full_accs):.2f}")
    print(f"시드 앙상블 (Hard voting):      {seed_hard_acc:.2f}%")
    print(f"시드 앙상블 (Soft voting):      {seed_soft_acc:.2f}%")
    print(f"통합 앙상블 (10모델, Soft):     {all_soft_acc:.2f}%")

    results = {
        "fold_accs": fold_accs,
        "fold_acc_mean": float(np.mean(fold_accs)),
        "fold_acc_std": float(np.std(fold_accs)),
        "kfold_hard_voting_acc": float(kf_hard_acc),
        "kfold_soft_voting_acc": float(kf_soft_acc),
        "full_train_accs": full_accs,
        "full_train_single_acc": float(single_acc),
        "seed_hard_voting_acc": float(seed_hard_acc),
        "seed_soft_voting_acc": float(seed_soft_acc),
        "all_soft_voting_acc": float(all_soft_acc),
    }
    save_json(results, os.path.join(RESULTS_DIR, "test_results.json"))

    # ----- 정확도 비교 막대그래프 -----
    os.makedirs(FIGURES_DIR, exist_ok=True)
    labels = ([f"Fold {f}" for f in range(args.k_splits)]
              + ["KFold\nHard", "KFold\nSoft", "Full-train\nsingle", "Seed-Ens\nHard", "Seed-Ens\nSoft",
                 "All-10\nSoft"])
    values = fold_accs + [kf_hard_acc, kf_soft_acc, single_acc, seed_hard_acc, seed_soft_acc,
                          all_soft_acc]
    colors = (["#9ecae1"] * args.k_splits + ["#74c476", "#31a354", "#fdae6b", "#6baed6", "#3182bd",
                                             "#08519c"])
    plt.figure(figsize=(11, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.bar_label(bars, fmt="%.2f", fontsize=9)
    plt.ylim(min(values) - 0.5, max(values) + 0.3)
    plt.ylabel("Test Accuracy (%)")
    plt.title("Single Models vs Ensemble (MNIST Test Set)")
    plt.grid(True, axis="y", alpha=0.3)
    fig_path = os.path.join(FIGURES_DIR, "part3_accuracy.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {fig_path}")

    # ----- 앙상블이 단일 모델 오류를 바로잡은 예시 시각화 -----
    single_preds = full_preds[0]
    fixed = np.where((single_preds != y_true) & (seed_soft_preds == y_true))[0][:8]
    if len(fixed) > 0:
        plt.figure(figsize=(2 * len(fixed), 2.6))
        for i, idx in enumerate(fixed):
            plt.subplot(1, len(fixed), i + 1)
            plt.imshow(x_test_raw[idx], cmap="gray")
            plt.title(f"true {y_true[idx]}\nsingle {single_preds[idx]} / ens {seed_soft_preds[idx]}",
                      fontsize=9)
            plt.axis("off")
        plt.suptitle("Samples corrected by the ensemble", y=1.02)
        fig_path = os.path.join(FIGURES_DIR, "part3_corrected_samples.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="앙상블 테스트셋 평가")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--k_splits", type=int, default=5)
    parser.add_argument("--n_full", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    main(parser.parse_args())
