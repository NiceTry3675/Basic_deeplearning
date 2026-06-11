# -*- coding: utf-8 -*-
"""단일 설정 학습 스크립트 (과제 (1), (2) 실험의 기본 단위)

예시:
    python train.py --optimizer adam --scheduler cosine --lr 1e-3 \
        --batch_size 128 --epochs 20 --seed 42 \
        --save_path ../results/models/adam_cosine.pth \
        --log_path  ../results/logs/adam_cosine_s42.json

학습 종료 후 테스트셋 성능까지 평가하여 JSON 로그로 저장한다.
(단, optimizer/scheduler "선택"은 validation 성능으로만 수행하고,
 테스트 성능은 최종 보고용으로만 사용한다.)
"""
import argparse
import json
import os
import time

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from data_loader import load_mnist
from model import MLP
from engine import (
    OPTIMIZER_NAMES,
    SCHEDULER_NAMES,
    set_seed,
    make_optimizer,
    make_scheduler,
    train_model,
    evaluate,
)


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 1) 데이터 준비 (train/val 분할은 val_seed 로 고정 → seed 가 달라도 동일 분할)
    train_ds, val_ds, test_ds, in_dim, _, _ = load_mnist(
        val_portion=args.val_portion, seed=args.val_seed
    )
    loader_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=loader_gen
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    # 2) 모델·손실·최적화기·스케줄러
    model = MLP(in_dim).to(device)
    criterion = CrossEntropyLoss()
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr)
    scheduler, step_mode = make_scheduler(
        args.scheduler, optimizer, args.lr, args.epochs, len(train_loader)
    )

    print(f"=== {args.optimizer} / {args.scheduler} | lr={args.lr} | "
          f"batch={args.batch_size} | epochs={args.epochs} | seed={args.seed} ===", flush=True)

    # 3) 학습
    t_start = time.monotonic()
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler=scheduler, step_mode=step_mode,
        epochs=args.epochs, device=device,
    )
    total_time = time.monotonic() - t_start

    # 4) 최종 테스트 평가 (보고용)
    test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)
    print(f"## Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.2f}% "
          f"(total {total_time:.1f}s)", flush=True)

    # 5) 모델 저장
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")

    # 6) 로그 저장
    if args.log_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
        log = {
            "config": vars(args),
            "history": history,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_val_acc": max(history["val_acc"]),
            "best_val_acc_epoch": history["val_acc"].index(max(history["val_acc"])) + 1,
            "final_val_acc": history["val_acc"][-1],
            "final_val_loss": history["val_loss"][-1],
            "total_time_sec": total_time,
        }
        with open(args.log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Log saved to {args.log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on MNIST (single config)")
    parser.add_argument("--optimizer", type=str, default="adam", choices=OPTIMIZER_NAMES)
    parser.add_argument("--scheduler", type=str, default="none", choices=SCHEDULER_NAMES)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="모델 초기화/셔플 seed")
    parser.add_argument("--val_seed", type=int, default=42, help="train/val 분할 seed (고정 권장)")
    parser.add_argument("--val_portion", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    main(args)
