import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_size: int=28):
        """
        in_size: 입력 이미지 한 변의 길이 (원본 28, 다운사이징 7)
        """
        super().__init__()
        self.in_size = in_size
        # conv(k=3, s=1, p=1)은 공간 크기를 유지하고, pool(2×2)은 절반(내림)으로 줄임
        #   28×28 -> conv1 -> 28×28 -> pool1 -> 14×14 -> conv2 -> 14×14 -> pool2 -> 7×7
        #    7×7  -> conv1 ->  7×7  -> pool1 ->  3×3  -> conv2 ->  3×3  -> pool2 -> 1×1
        self.feat_size = (in_size // 2) // 2
        self.flat_dim  = 64 * self.feat_size * self.feat_size   # 28×28: 64*7*7 = 3136

        # 첫 번째 Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 두 번째 Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Input shape: (batch_size, 784) -> (batch_size, 1, 28, 28)
        # Input shape: (batch_size, 49)  -> (batch_size, 1, 7, 7)
        # 배치 차원은 -1 대신 명시: 입력 크기가 맞지 않으면 배치가 바뀌는 대신 즉시 오류
        x = x.view(x.size(0), 1, self.in_size, self.in_size)

        # Conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Flatten: 배치 차원은 유지하고 나머지만 펼침
        # (view(-1, 상수)는 채널/크기가 어긋나면 배치 크기를 조용히 바꿔버리므로 사용하지 않음)
        x = x.flatten(1)

        # FC layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    from thop import profile

    batch_size = 64
    for in_size in (28, 7):
        x = torch.randn(batch_size, in_size * in_size)
        model = CNN(in_size)
        model.eval()
        output = model(x)
        print(f"===== in_size = {in_size}×{in_size} =====")

        # 모델 파라미터 수 계산 (레이어별)
        for name, m in model.named_children():
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            if n:
                print(f"  {name:8s}: {n:>9,}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params:,}")

        # 샘플 1장 기준 연산량(MACs) 계산 pip install thop
        macs, params = profile(model, inputs=(x[:1], ), verbose=False)
        print(f'MACs: {macs / 1e6:.3f} M (per sample), Params: {params / 1e6:.5f} M')

        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}\n")
