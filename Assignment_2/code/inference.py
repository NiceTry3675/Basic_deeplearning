# -*- coding: utf-8 -*-
"""앙상블 추론(inference) 스크립트 (과제 (3))

테스트셋의 단일 샘플에 대해 fold 별 예측과 앙상블(Soft Voting) 예측을 출력하고
시각화 이미지를 저장한다.

예시:
    python inference.py --model_prefix ../results/models/ensemble/mlp_mnist \
        --k_splits 5 --sample_idx 11 \
        --fig_path ../results/figures/inference_sample.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")  # GUI 없는 환경에서도 동작
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_mnist
from model import MLP


def load_fold_models(model_prefix, k_splits, in_dim, device):
    models = []
    for fold in range(k_splits):
        model_path = f"{model_prefix}_fold{fold}.pth"
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{model_path} 파일을 찾을 수 없습니다.")
        model = MLP(in_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models


def infer_ensemble(models, x_sample, device):
    """단일 샘플에 대한 fold별 예측과 Soft Voting 앙상블 예측."""
    tensor = x_sample.unsqueeze(0).to(device)  # (1, 784)
    fold_preds = []
    probs_sum = torch.zeros(1, 10, device=device)
    with torch.no_grad():
        for model in models:
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            probs_sum += probs
            fold_preds.append(int(output.argmax(dim=1).item()))
    mean_probs = (probs_sum / len(models)).squeeze(0).cpu().numpy()
    ensemble_pred = int(mean_probs.argmax())
    return fold_preds, ensemble_pred, mean_probs


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 1) 데이터 로드 및 샘플 전처리 (학습과 동일한 정규화는 test_ds 텐서에 이미 적용됨)
    _, _, test_ds, in_dim, x_test_raw, y_test = load_mnist(val_portion=0.0)
    x_sample = test_ds.tensors[0][args.sample_idx]

    # 2) 앙상블 추론
    models = load_fold_models(args.model_prefix, args.k_splits, in_dim, device)
    fold_preds, ensemble_pred, mean_probs = infer_ensemble(models, x_sample, device)

    print(f"Sample Index: {args.sample_idx} (True Label: {y_test[args.sample_idx]})")
    for fold, p in enumerate(fold_preds):
        print(f"  Fold {fold} Prediction: {p}")
    print(f"Ensemble Prediction (soft voting): {ensemble_pred}")
    print("Mean probabilities:", np.round(mean_probs, 4))

    # 3) 시각화 저장
    if args.fig_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.fig_path)), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(x_test_raw[args.sample_idx], cmap="gray")
        axes[0].set_title(f"Ensemble Pred: {ensemble_pred} / True: {y_test[args.sample_idx]}")
        axes[0].axis("off")

        axes[1].bar(range(10), mean_probs, color="steelblue")
        axes[1].set_xticks(range(10))
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Mean probability")
        axes[1].set_title(f"Soft-voting probabilities (K={args.k_splits})")
        fig.tight_layout()
        fig.savefig(args.fig_path, dpi=150)
        print(f"Figure saved to {args.fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble inference on a single MNIST sample")
    parser.add_argument("--model_prefix", type=str, default="../results/models/ensemble/mlp_mnist")
    parser.add_argument("--k_splits", type=int, default=5)
    parser.add_argument("--sample_idx", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fig_path", type=str, default="../results/figures/inference_sample.png")
    args = parser.parse_args()

    main(args)
