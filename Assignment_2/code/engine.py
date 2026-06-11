# -*- coding: utf-8 -*-
"""학습/평가 공통 엔진

- Optimizer / LR Scheduler 팩토리
- 에폭 단위 학습 루프 (per-epoch metric 기록)
- 재현성을 위한 seed 고정 유틸
"""
import random
import time

import numpy as np
import torch
from torch.optim import SGD, Adagrad, RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
)

OPTIMIZER_NAMES = ["sgd", "sgd_momentum", "sgd_nesterov", "adagrad", "rmsprop", "adam", "adamw"]
SCHEDULER_NAMES = ["none", "step", "exponential", "cosine", "plateau", "onecycle"]


def set_seed(seed: int):
    """모든 난수원(seed) 고정 — 동일 조건 재실행 시 동일 결과 보장."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(name: str, params, lr: float):
    """이름으로 optimizer 생성. 비교 대상 7종."""
    name = name.lower()
    if name == "sgd":
        return SGD(params, lr=lr)
    if name == "sgd_momentum":
        return SGD(params, lr=lr, momentum=0.9)
    if name == "sgd_nesterov":
        return SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == "adagrad":
        return Adagrad(params, lr=lr)
    if name == "rmsprop":
        return RMSprop(params, lr=lr, alpha=0.99)
    if name == "adam":
        return Adam(params, lr=lr)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=0.01)
    raise ValueError(f"Unknown optimizer: {name}")


def make_scheduler(name: str, optimizer, lr: float, epochs: int, steps_per_epoch: int):
    """이름으로 scheduler 생성.

    Returns:
        (scheduler, step_mode)
        step_mode: "epoch"   — 매 에폭 후 scheduler.step()
                   "batch"   — 매 미니배치 후 scheduler.step() (OneCycle)
                   "plateau" — 매 에폭 후 scheduler.step(val_loss)
                   None      — scheduler 없음
    """
    name = name.lower()
    if name == "none":
        return None, None
    if name == "step":
        return StepLR(optimizer, step_size=5, gamma=0.5), "epoch"
    if name == "exponential":
        return ExponentialLR(optimizer, gamma=0.9), "epoch"
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01), "epoch"
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2), "plateau"
    if name == "onecycle":
        # momentum 을 실제로 사용하는 optimizer 에서만 momentum cycling 활성화
        # (momentum 미사용 optimizer 에 켜면 오류가 나거나 optimizer 동작이 바뀜)
        defaults = optimizer.defaults
        cycle_momentum = ("betas" in defaults) or (defaults.get("momentum", 0) != 0)
        return (
            OneCycleLR(
                optimizer,
                max_lr=lr * 3,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                cycle_momentum=cycle_momentum,
            ),
            "batch",
        )
    raise ValueError(f"Unknown scheduler: {name}")


def evaluate(model, loader, criterion, device):
    """평가: (평균 loss, 정확도(%), 예측 리스트) 반환."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    all_preds = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
            all_preds.extend(pred.cpu().tolist())
    return loss_sum / total, 100.0 * correct / total, all_preds


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    step_mode=None,
    epochs: int = 20,
    device=None,
    verbose: bool = True,
):
    """학습 루프. 에폭별 train/val loss·acc, lr, 소요시간을 기록해 반환."""
    history = {
        "epoch": [], "lr": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epoch_time": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()
        epoch_lr = optimizer.param_groups[0]["lr"]  # 이 에폭 시작 시점의 LR

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if step_mode == "batch":
                scheduler.step()

            train_loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += data.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = 100.0 * train_correct / train_total

        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)

        if step_mode == "epoch":
            scheduler.step()
        elif step_mode == "plateau":
            scheduler.step(val_loss)

        elapsed = time.monotonic() - t0
        history["epoch"].append(epoch)
        history["lr"].append(epoch_lr)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)

        if verbose:
            print(
                f"[Epoch {epoch:2d}] LR: {epoch_lr:.6f} | "
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}% | "
                f"{elapsed:.1f}s",
                flush=True,
            )

    return history
