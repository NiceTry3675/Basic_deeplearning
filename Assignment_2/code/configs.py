"""실험 설정의 단일 출처: 공통 하이퍼파라미터, optimizer/scheduler 팩토리, LR 탐색 그리드."""
from torch.optim import SGD, Adagrad, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (CosineAnnealingLR, CyclicLR, ExponentialLR,
                                      OneCycleLR, ReduceLROnPlateau, StepLR)

COMMON = {
    "batch_size": 64,
    "epochs": 15,
    "sweep_epochs": 5,
    "val_portion": 0.1,   # train 60,000 -> train 54,000 / val 6,000
    "seeds": [0, 1, 2],
    "split_seed": 42,
}

# Part 1: 강의자료 5종(SGD, SGD+momentum, Adagrad, RMSprop, Adam) + AdamW(확장)
OPTIMIZERS = {
    "SGD":          lambda params, lr: SGD(params, lr=lr),
    "SGD-Momentum": lambda params, lr: SGD(params, lr=lr, momentum=0.9),
    "Adagrad":      lambda params, lr: Adagrad(params, lr=lr),
    "RMSprop":      lambda params, lr: RMSprop(params, lr=lr, alpha=0.99),
    "Adam":         lambda params, lr: Adam(params, lr=lr),
    "AdamW":        lambda params, lr: AdamW(params, lr=lr, weight_decay=1e-2),
}

# optimizer 계열별 LR 탐색 범위 (적응형 계열은 더 작은 LR이 적합)
LR_GRID = {
    "SGD":          [1e-1, 1e-2, 1e-3],
    "SGD-Momentum": [1e-1, 1e-2, 1e-3],
    "Adagrad":      [1e-1, 1e-2, 1e-3],
    "RMSprop":      [1e-2, 1e-3, 1e-4],
    "Adam":         [1e-2, 1e-3, 1e-4],
    "AdamW":        [1e-2, 1e-3, 1e-4],
}

# Part 2: 상수 LR baseline + 강의 4종(설정 보정) + 확장 2종
SCHEDULER_NAMES = ["None", "StepLR", "ExponentialLR", "CosineAnnealingLR",
                   "CyclicLR", "ReduceLROnPlateau", "OneCycleLR"]


def make_scheduler(name, optimizer, base_lr, epochs, iters_per_epoch):
    """(scheduler, step 모드)를 반환한다. 모드: 'epoch' | 'batch' | 'plateau'

    강의 기본값 대비 보정: CosineAnnealingLR의 T_max는 전체 epoch 수에 맞추고
    (강의 T_max=100은 10 epoch 학습 시 코사인 곡선의 10%만 사용),
    CyclicLR/OneCycleLR은 batch 단위로 step해 주기가 실제로 동작하도록 함.
    """
    if name == "None":
        return None, "epoch"
    if name == "StepLR":
        return StepLR(optimizer, step_size=5, gamma=0.5), "epoch"
    if name == "ExponentialLR":
        return ExponentialLR(optimizer, gamma=0.9), "epoch"
    if name == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr / 100), "epoch"
    if name == "CyclicLR":
        return CyclicLR(optimizer, base_lr=base_lr / 10, max_lr=base_lr,
                        step_size_up=int(iters_per_epoch * 2.5), mode="triangular",
                        cycle_momentum=False), "batch"
    if name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2), "plateau"
    if name == "OneCycleLR":
        # cycle_momentum=False: momentum/betas가 없는 optimizer(Adagrad 등)에서도 동작
        return OneCycleLR(optimizer, max_lr=base_lr * 3,
                          total_steps=epochs * iters_per_epoch,
                          cycle_momentum=False), "batch"
    raise ValueError(f"unknown scheduler: {name}")


# Part 1/2 실험으로 결정된 최종 구성 — train.py/test.py/inference.py의 기본값.
# Part 1: Adagrad(lr=0.1)가 최종 검증 정확도 98.40%로 1위 (수렴 속도도 최고)
# Part 2: OneCycleLR가 98.56%로 baseline(98.40%) 및 나머지 scheduler 대비 우위
BEST = {
    "optimizer": "Adagrad",
    "lr": 0.1,
    "scheduler": "OneCycleLR",
}
