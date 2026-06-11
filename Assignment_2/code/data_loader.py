"""MNIST 데이터 로더 (강의자료 pr_14 data_loader.py 기반).

강의 버전 대비 변경점:
- mnist.npz 경로를 스크립트 위치 기준으로 해석 (실행 위치 무관)
- train/val 분할에 시드 고정 generator 사용 → 모든 실험이 동일한 검증셋을 공유
- valDB_portion=0이면 val_ds=None 반환 (random_split 호출 안 함)
"""
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist.npz")


def load_mnist(use_small: bool = False, valDB_portion: float = 0.0, split_seed: int = 42):
    # 1) MNIST npz 로드 및 z-정규화 (train 통계 사용)
    mnist = np.load(DATA_PATH)
    mean = np.mean(mnist["x_train"])
    std = np.std(mnist["x_train"])
    x_train = (mnist["x_train"] - mean) / std
    y_train = mnist["y_train"]
    x_test = (mnist["x_test"] - mean) / std
    y_test = mnist["y_test"]

    # 2) 다운사이징(7x7) 혹은 원본(28x28)
    if use_small:
        x_train_in = x_train[:, ::4, ::4].reshape(-1, 7 * 7)
        x_test_in = x_test[:, ::4, ::4].reshape(-1, 7 * 7)
    else:
        x_train_in = x_train.reshape(-1, 28 * 28)
        x_test_in = x_test.reshape(-1, 28 * 28)

    input_dim = x_train_in.shape[1]

    train_ds = TensorDataset(torch.FloatTensor(x_train_in), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(x_test_in), torch.LongTensor(y_test))

    # 3) Train/Validation 분할 — 분할 시드는 실험 시드와 독립적으로 고정
    val_size = int(valDB_portion * len(train_ds))
    if val_size > 0:
        generator = torch.Generator().manual_seed(split_seed)
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [len(train_ds) - val_size, val_size], generator=generator
        )
    else:
        val_ds = None

    # 시각화용 원본(정규화 전) 테스트 이미지도 함께 반환
    return train_ds, val_ds, test_ds, input_dim, mnist["x_test"], y_test
