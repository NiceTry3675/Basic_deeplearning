# -*- coding: utf-8 -*-
"""앙상블 테스트 스크립트 (과제 (3))

Fold 별 단일 모델 정확도와 앙상블(Hard/Soft Voting) 정확도를 비교 평가한다.
- Hard Voting: fold 별 예측 클래스의 다수결 (동률 시 평균 확률이 높은 클래스)
- Soft Voting: fold 별 softmax 확률 평균 후 argmax

예시:
    python test_ensemble.py --model_prefix ../results/models/ensemble/mlp_mnist --k_splits 5
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import load_mnist
from model import MLP


def collect_probs(model, test_loader, device):
    """테스트셋 전체에 대한 softmax 확률 (N,10) 반환."""
    model.eval()
    probs = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            probs.append(F.softmax(output, dim=1).cpu())
    return torch.cat(probs, dim=0).numpy()


def mcnemar_exact(preds_a, preds_b, y_true):
    """두 분류기의 paired 예측에 대한 exact McNemar 검정.

    b = A만 맞힌 샘플 수, c = B만 맞힌 샘플 수.
    귀무가설(두 분류기의 오류율 동일) 하에서 p = 2 * P[X <= min(b,c)], X~Bin(b+c, 0.5).
    """
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return b, c, min(1.0, p)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    _, _, test_ds, in_dim, _, y_test = load_mnist(val_portion=0.0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    y_test = np.asarray(y_test)

    # 1) Fold 별 확률 수집 및 단일 모델 정확도
    all_probs = []          # (K, N, 10)
    fold_accuracies = []
    for fold in range(args.k_splits):
        model_path = f"{args.model_prefix}_fold{fold}.pth"
        model = MLP(in_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        probs = collect_probs(model, test_loader, device)
        preds = probs.argmax(axis=1)
        acc = 100.0 * (preds == y_test).mean()
        print(f"Fold {fold} Test Accuracy: {acc:.2f}%")
        fold_accuracies.append(acc)
        all_probs.append(probs)

    all_probs = np.stack(all_probs)            # (K, N, 10)
    all_preds = all_probs.argmax(axis=2)       # (K, N)
    mean_probs = all_probs.mean(axis=0)        # (N, 10)

    # 2) Hard Voting (다수결, 동률 시 평균 확률이 높은 클래스 선택)
    n_test = all_preds.shape[1]
    hard_preds = np.empty(n_test, dtype=np.int64)
    for i in range(n_test):
        counts = np.bincount(all_preds[:, i], minlength=10)
        winners = np.flatnonzero(counts == counts.max())
        hard_preds[i] = winners[np.argmax(mean_probs[i, winners])]
    hard_acc = 100.0 * (hard_preds == y_test).mean()

    # 3) Soft Voting (확률 평균)
    soft_preds = mean_probs.argmax(axis=1)
    soft_acc = 100.0 * (soft_preds == y_test).mean()

    print(f"\nSingle-model mean accuracy : {np.mean(fold_accuracies):.2f}% "
          f"(min {min(fold_accuracies):.2f}%, max {max(fold_accuracies):.2f}%)")
    print(f"## Ensemble (Hard Voting) Test Accuracy: {hard_acc:.2f}%")
    print(f"## Ensemble (Soft Voting) Test Accuracy: {soft_acc:.2f}%")

    # 앙상블(soft) vs 최고 단일 fold 모델의 paired 비교 (exact McNemar test)
    best_fold = int(np.argmax(fold_accuracies))
    mc_b, mc_c, mc_p = mcnemar_exact(soft_preds, all_preds[best_fold], np.asarray(y_test))
    print(f"McNemar (soft ensemble vs best fold {best_fold}): "
          f"ensemble-only-correct={mc_b}, fold-only-correct={mc_c}, p={mc_p:.4f}")

    # 4) 결과 저장 (보고서/그림 생성용)
    if args.log_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
        # 혼동 행렬 (soft voting 기준)
        confusion = np.zeros((10, 10), dtype=int)
        for t, p in zip(y_test, soft_preds):
            confusion[t, p] += 1
        log = {
            "config": vars(args),
            "fold_accuracies": fold_accuracies,
            "single_mean_acc": float(np.mean(fold_accuracies)),
            "single_max_acc": float(np.max(fold_accuracies)),
            "hard_voting_acc": float(hard_acc),
            "soft_voting_acc": float(soft_acc),
            "mcnemar_vs_best_fold": {"best_fold": best_fold, "ensemble_only_correct": mc_b,
                                     "fold_only_correct": mc_c, "p_value": mc_p},
            "confusion_matrix_soft": confusion.tolist(),
        }
        with open(args.log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Log saved to {args.log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate K-Fold ensemble on MNIST test set")
    parser.add_argument("--model_prefix", type=str, default="../results/models/ensemble/mlp_mnist")
    parser.add_argument("--k_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_path", type=str, default="../results/logs/ensemble/ensemble_test.json")
    args = parser.parse_args()

    main(args)
