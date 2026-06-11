"""공용 학습/평가 루프 (강의자료 pr_14 train_optimizer_scheduler.py의 train()을 일반화).

- per-epoch 지표(train/val loss·acc, lr, 소요 시간)를 history 리스트로 반환
- scheduler_mode:
    "epoch"   : 매 epoch 종료 후 scheduler.step()        (StepLR, ExponentialLR, CosineAnnealingLR 등)
    "batch"   : 매 batch(optimizer.step()) 후 step()     (CyclicLR, OneCycleLR)
    "plateau" : 매 epoch 종료 후 step(val_loss)          (ReduceLROnPlateau)
- loss는 표본 단위 평균(배치 크기 가중)으로 집계해 마지막 부분 배치 편향을 제거
"""
import time

import torch


def evaluate(model, loader, criterion, device):
    """주어진 데이터셋에 대한 (평균 loss, 정확도 %)를 반환한다."""
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += criterion(output, target).item() * data.size(0)
            correct += (output.argmax(dim=1) == target).sum().item()
    n = len(loader.dataset)
    return loss_sum / n, 100.0 * correct / n


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, scheduler_mode="epoch", epochs=15, device=None):
    history = []
    lr_log_every = max(1, len(train_loader) // 10)

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        lr_samples = []

        for it, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # batch 모드 scheduler의 LR 궤적 시각화를 위해 epoch당 ~10회 샘플링
            if it == 0 or (scheduler_mode == "batch" and it % lr_log_every == 0):
                lr_samples.append(optimizer.param_groups[0]["lr"])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_mode == "batch":
                scheduler.step()

            loss_sum += loss.item() * data.size(0)
            correct += (output.argmax(dim=1) == target).sum().item()
            seen += data.size(0)

        train_loss = loss_sum / seen
        train_acc = 100.0 * correct / seen
        epoch_time = time.time() - start

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = None, None

        if scheduler is not None:
            if scheduler_mode == "epoch":
                scheduler.step()
            elif scheduler_mode == "plateau":
                # full-train(검증셋 없음)에서는 train_loss로 대체
                scheduler.step(val_loss if val_loss is not None else train_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr_samples[0],
            "lr_samples": lr_samples,
            "epoch_time_sec": epoch_time,
        })

        val_str = (f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%"
                   if val_loss is not None else "Val: -")
        print(f"[Epoch {epoch:2d}] LR: {lr_samples[0]:.6f} | "
              f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}% | "
              f"{val_str} | {epoch_time:.1f}s", flush=True)

    return history
