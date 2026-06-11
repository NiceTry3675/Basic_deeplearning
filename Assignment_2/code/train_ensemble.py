# -*- coding: utf-8 -*-
"""K-Fold 앙상블 학습 스크립트 (과제 (3))

(1)에서 선택된 optimizer, (2)에서 선택된 scheduler 를 적용하여
전체 학습 데이터(60,000장)를 K개 fold 로 나누고 fold 별 모델을 학습/저장한다.

예시:
    python train_ensemble.py --optimizer adam --scheduler cosine --lr 1e-3 \
        --k_splits 5 --epochs 20 --batch_size 128 \
        --save_prefix ../results/models/ensemble/mlp_mnist \
        --log_dir ../results/logs/ensemble
"""
import argparse
import json
import os
import time

import torch
from sklearn.model_selection import KFold
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset

from data_loader import load_mnist
from model import MLP
from engine import (
    OPTIMIZER_NAMES,
    SCHEDULER_NAMES,
    set_seed,
    make_optimizer,
    make_scheduler,
    train_model,
)


def main(args):
    # 1) 전체 학습 데이터 로드 (fold 분할은 KFold 가 담당하므로 val 분할 없음)
    train_full, _, _, in_dim, _, _ = load_mnist(val_portion=0.0)

    kfold = KFold(n_splits=args.k_splits, shuffle=True, random_state=42)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    criterion = CrossEntropyLoss()

    os.makedirs(os.path.dirname(os.path.abspath(args.save_prefix)), exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    # 2) Fold 별 학습
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(train_full)))):
        print(f"\n========== Fold {fold + 1}/{args.k_splits} ==========", flush=True)
        set_seed(args.seed + fold)  # fold 마다 다른 초기화 → 앙상블 다양성 확보

        train_loader = DataLoader(
            Subset(train_full, train_ids.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + fold),
        )
        val_loader = DataLoader(
            Subset(train_full, val_ids.tolist()), batch_size=512, shuffle=False
        )

        model = MLP(in_dim).to(device)
        optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr)
        scheduler, step_mode = make_scheduler(
            args.scheduler, optimizer, args.lr, args.epochs, len(train_loader)
        )

        t0 = time.monotonic()
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler=scheduler, step_mode=step_mode,
            epochs=args.epochs, device=device,
        )
        elapsed = time.monotonic() - t0

        model_path = f"{args.save_prefix}_fold{fold}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Fold {fold} model saved to {model_path} ({elapsed:.1f}s)")

        if args.log_dir:
            log = {
                "config": vars(args),
                "fold": fold,
                "history": history,
                "final_val_acc": history["val_acc"][-1],
                "total_time_sec": elapsed,
            }
            with open(os.path.join(args.log_dir, f"fold{fold}.json"), "w") as f:
                json.dump(log, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold ensemble training on MNIST")
    parser.add_argument("--optimizer", type=str, default="adam", choices=OPTIMIZER_NAMES)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=SCHEDULER_NAMES)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--k_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_prefix", type=str, default="../results/models/ensemble/mlp_mnist")
    parser.add_argument("--log_dir", type=str, default="../results/logs/ensemble")
    args = parser.parse_args()

    main(args)
