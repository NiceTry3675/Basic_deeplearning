"""MLP 분류 모델 (강의자료 pr_14 model_v2.py와 동일 구조).

784 -> 256 -> 256 -> 10, BatchNorm + ReLU + Dropout(0.2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)

        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout2(x2)

        x3 = self.fc3(x2)
        return x3


if __name__ == "__main__":
    x = torch.randn(8, 784)
    model = MLP(784)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {model(x).shape}")
