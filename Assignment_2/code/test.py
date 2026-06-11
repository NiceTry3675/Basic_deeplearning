# -*- coding: utf-8 -*-
"""학습된 단일 모델의 테스트셋 평가 스크립트

예시:
    python test.py --model_path ../results/models/ensemble/mlp_mnist_fold0.pth
"""
import argparse

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from data_loader import load_mnist
from model import MLP
from engine import evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    _, _, test_ds, in_dim, _, _ = load_mnist(val_portion=0.0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MLP(in_dim)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    criterion = CrossEntropyLoss()
    test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)
    print(f"Model: {args.model_path}")
    print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained MLP on MNIST")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args)
