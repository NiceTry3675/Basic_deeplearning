"""
공통 유틸: GPU 상주 MNIST 로딩, train/val 분할, EMA, 평가, 복잡도(Params/MACs) 측정
- 테스트 데이터(x_test)는 학습에 절대 사용하지 않는다. (평가 전용)
- 학습 데이터 증강은 사용하지 않는다. (student·teacher 모두 원본 학습 데이터만 사용)
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "data", "mnist.npz")

# 검증용 분할: 학습 60,000장 중 고정 시드(12345)로 5,000장을 떼어낸다.
VAL_SIZE = 5000
SPLIT_SEED = 12345


def load_mnist(device, split: str = "dev"):
    """
    split='dev'  : train 55,000 / val 5,000 (하이퍼파라미터 탐색용, test는 반환하지만 사용 안 함)
    split='full' : train 60,000 / val 없음 (최종 학습용)
    반환: dict(x_tr, y_tr, idx_tr, x_val, y_val, x_te, y_te) — 모두 GPU 텐서, x는 (N,1,28,28) 정규화 완료
    idx_tr: 원본 60,000장 기준 인덱스 (KD teacher logits 조회용)
    """
    d = np.load(DATA_PATH)
    x_train = d["x_train"].astype(np.float32) / 255.0
    y_train = d["y_train"].astype(np.int64)
    x_test = d["x_test"].astype(np.float32) / 255.0
    y_test = d["y_test"].astype(np.int64)

    # 정규화 통계는 학습 데이터에서만 계산 (MNIST 표준값과 동일: 0.1307 / 0.3081)
    mean, std = 0.1307, 0.3081
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    perm = np.random.RandomState(SPLIT_SEED).permutation(len(x_train))
    if split == "dev":
        idx_val = np.sort(perm[:VAL_SIZE])
        idx_tr = np.sort(perm[VAL_SIZE:])
    elif split == "full":
        idx_val = np.array([], dtype=np.int64)
        idx_tr = np.arange(len(x_train))
    else:
        raise ValueError(split)

    t = lambda a: torch.from_numpy(a).to(device)
    out = dict(
        x_tr=t(x_train[idx_tr]).unsqueeze(1), y_tr=t(y_train[idx_tr]), idx_tr=t(idx_tr),
        x_val=t(x_train[idx_val]).unsqueeze(1) if len(idx_val) else None,
        y_val=t(y_train[idx_val]) if len(idx_val) else None,
        x_te=t(x_test).unsqueeze(1), y_te=t(y_test),
    )
    return out


@torch.no_grad()
def predict_logits(model, x, bs=2000):
    model.eval()
    outs = [model(x[i:i + bs]).float() for i in range(0, len(x), bs)]
    return torch.cat(outs)


@torch.no_grad()
def accuracy(model, x, y, bs=2000):
    logits = predict_logits(model, x, bs)
    return (logits.argmax(1) == y).float().mean().item() * 100.0


class EMA:
    """가중치 지수이동평균 (BN running stats 포함 buffer도 복사)"""

    def __init__(self, model, decay=0.999, warmup=True):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.n = 0
        esd, msd = self.ema.state_dict(), model.state_dict()
        self.e_f = [v for k, v in esd.items() if v.dtype.is_floating_point]
        self.m_f = [msd[k] for k, v in esd.items() if v.dtype.is_floating_point]
        self.e_i = [v for k, v in esd.items() if not v.dtype.is_floating_point]
        self.m_i = [msd[k] for k, v in esd.items() if not v.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = self.decay
        if self.warmup:  # 초기에는 짧은 평균 (timm 방식)
            d = min(d, (1 + self.n) / (10 + self.n))
        torch._foreach_lerp_(self.e_f, self.m_f, 1 - d)  # e = d*e + (1-d)*m
        for e, m in zip(self.e_i, self.m_i):
            e.copy_(m)


# ---------------------------------------------------------------- complexity
def count_complexity(model, verbose=False):
    """thop / ptflops 두 도구로 MACs·Params 측정 (교수님이 어느 툴을 쓰셔도 통과하도록)"""
    import io
    import contextlib
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 1, 28, 28)
    res = {}
    n_params = sum(p.numel() for p in m.parameters())
    res["params"] = n_params
    try:
        from thop import profile
        with contextlib.redirect_stdout(io.StringIO()):
            macs, params = profile(copy.deepcopy(m), inputs=(x,), verbose=False)
        res["thop_macs"] = int(macs)
        res["thop_params"] = int(params)
    except Exception as e:  # pragma: no cover
        res["thop_err"] = str(e)
    try:
        from ptflops import get_model_complexity_info
        with contextlib.redirect_stdout(io.StringIO()):
            macs, params = get_model_complexity_info(copy.deepcopy(m), (1, 28, 28), as_strings=False,
                                                     print_per_layer_stat=verbose, verbose=False)
        res["ptflops_macs"] = int(macs)
        res["ptflops_params"] = int(params)
    except Exception as e:  # pragma: no cover
        res["ptflops_err"] = str(e)
    res["max_macs"] = max(res.get("thop_macs", 0), res.get("ptflops_macs", 0))
    res["max_params"] = max(n_params, res.get("thop_params", 0), res.get("ptflops_params", 0))
    res["ok"] = res["max_macs"] <= 10_000_000 and res["max_params"] <= 30_000
    return res
