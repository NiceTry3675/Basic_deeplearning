"""Part 3: 단일 샘플 앙상블 추론 (강의자료 pr_14 inference_ensemble.py 기반).

테스트셋에서 한 장을 골라 fold 모델별 예측과 앙상블(soft voting) 예측을 출력하고
이미지 + 클래스별 평균 확률을 PNG로 저장한다.

사용법: python inference.py --sample_idx 11
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_mnist
from model import MLP
from utils import get_device

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CODE_DIR, "models")
FIGURES_DIR = os.path.normpath(os.path.join(CODE_DIR, "..", "figures"))


def main(args):
    device = get_device(args.device)
    _, _, test_ds, in_dim, x_test_raw, y_test = load_mnist(valDB_portion=0.0)
    x_sample = test_ds[args.sample_idx][0].unsqueeze(0).to(device)
    true_label = int(y_test[args.sample_idx])

    # ----- Fold 모델별 예측 -----
    fold_probs = []
    print(f"샘플 #{args.sample_idx} (정답: {true_label})")
    for fold in range(args.k_splits):
        path = os.path.join(args.model_dir, f"mlp_fold{fold}.pth")
        model = MLP(in_dim).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(x_sample), dim=1).squeeze(0).cpu().numpy()
        fold_probs.append(probs)
        print(f"  Fold {fold}: 예측 {probs.argmax()} (확률 {probs.max():.4f})")

    # ----- Soft voting 앙상블 -----
    mean_probs = np.mean(fold_probs, axis=0)
    ens_pred = int(mean_probs.argmax())
    print(f"앙상블 (Soft voting): 예측 {ens_pred} (평균 확률 {mean_probs.max():.4f})")
    print("정답" if ens_pred == true_label else "오답")

    # ----- 시각화 저장 -----
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].imshow(x_test_raw[args.sample_idx], cmap="gray")
    axes[0].set_title(f"true: {true_label} / ensemble: {ens_pred}")
    axes[0].axis("off")
    axes[1].bar(range(10), mean_probs, color="#3182bd")
    axes[1].set_xticks(range(10))
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Mean Softmax Probability")
    axes[1].set_title("Ensemble class probabilities")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig_path = os.path.join(FIGURES_DIR, f"part3_inference_sample{args.sample_idx}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단일 샘플 앙상블 추론")
    parser.add_argument("--sample_idx", type=int, default=11)
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--k_splits", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    main(parser.parse_args())
