# -*- coding: utf-8 -*-
"""MNIST 데이터 로더 (강의자료 pr_14 data_loader.py 기반, 재현성 보강)

- mnist.npz 를 로드하여 train 통계(mean/std)로 표준화
- train/val 분할 시 seed 고정 generator 사용 (재현성)
- 시각화를 위해 정규화 전 원본 x_test 도 함께 반환
"""
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist.npz")


def load_mnist(val_portion: float = 0.1, seed: int = 42, data_path: str = DEFAULT_DATA_PATH):
    """MNIST 로드 및 전처리.

    Returns:
        train_ds  : 학습용 TensorDataset (val 분할 후)
        val_ds    : 검증용 TensorDataset (val_portion=0 이면 None)
        test_ds   : 테스트용 TensorDataset
        input_dim : 입력 차원 (28*28=784)
        x_test_raw: 정규화 전 테스트 이미지 (N,28,28) — 시각화용
        y_test    : 테스트 라벨 (N,)
    """
    mnist = np.load(data_path)
    mean = np.mean(mnist["x_train"])
    std = np.std(mnist["x_train"])

    x_train = (mnist["x_train"] - mean) / std
    x_test = (mnist["x_test"] - mean) / std
    y_train = mnist["y_train"]
    y_test = mnist["y_test"]

    x_train_in = x_train.reshape(-1, 28 * 28)
    x_test_in = x_test.reshape(-1, 28 * 28)
    input_dim = x_train_in.shape[1]

    train_full = TensorDataset(torch.FloatTensor(x_train_in), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(x_test_in), torch.LongTensor(y_test))

    if val_portion > 0:
        val_size = int(val_portion * len(train_full))
        train_size = len(train_full) - val_size
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=gen)
    else:
        train_ds, val_ds = train_full, None

    return train_ds, val_ds, test_ds, input_dim, mnist["x_test"], y_test
